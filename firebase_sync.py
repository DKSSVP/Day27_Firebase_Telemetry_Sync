import copy
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
import os


class FirebaseTelemetrySync:
    """
    Background Firestore writer for normalized proctor alert events.

    Events are queued by the alert-processing path and uploaded by a
    dedicated worker so Firestore network calls do not block camera or
    microphone capture threads.
    """

    def __init__(
        self,
        project_id: str,
        credentials_path: str,
        collection_name: str = "proctor_tracking",
        queue_size: int = 1000,
        max_retries: int = 3,
    ):
        if not project_id:
            raise ValueError("FIREBASE_PROJECT_ID is required.")
        if not collection_name:
            raise ValueError("FIREBASE_COLLECTION is required.")

        self.project_id = project_id
        self.collection_name = collection_name
        self.credentials_path = Path(credentials_path).expanduser().resolve()
        self.max_retries = max(1, int(max_retries))

        self.logger = logging.getLogger("FirebaseTelemetrySync")
        self.event_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.worker: Optional[threading.Thread] = None
        self.stop_token = object()
        self.db = None
        self._started = False
        self._closed = False

    @classmethod
    def from_env(cls, base_dir: Path) -> "FirebaseTelemetrySync":
        """Load Firebase settings from the project's .env file."""
        base_dir = Path(base_dir).resolve()
        load_dotenv(base_dir / ".env")

        project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
        credentials_value = os.getenv(
            "FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json"
        ).strip()
        collection_name = os.getenv(
            "FIREBASE_COLLECTION", "proctor_tracking"
        ).strip()

        credentials_path = Path(credentials_value)
        if not credentials_path.is_absolute():
            credentials_path = base_dir / credentials_path

        return cls(
            project_id=project_id,
            credentials_path=str(credentials_path),
            collection_name=collection_name,
        )

    def connect(self) -> None:
        """Initialize Firebase Admin and obtain a Firestore client."""
        if not self.credentials_path.is_file():
            raise FileNotFoundError(
                "Firebase service-account JSON was not found at the configured path."
            )

        try:
            app = firebase_admin.get_app()
            if app.project_id != self.project_id:
                raise RuntimeError(
                    "An existing Firebase Admin app uses a different project ID."
                )
        except ValueError:
            certificate = credentials.Certificate(str(self.credentials_path))
            app = firebase_admin.initialize_app(
                certificate,
                {"projectId": self.project_id},
            )

        self.db = firestore.client(app=app)
        # A harmless server-side read verifies credentials, project and database access.
        self.db.collection(self.collection_name).limit(1).get()
        self.logger.info(
            "Firestore connection verified | project=%s | collection=%s",
            self.project_id,
            self.collection_name,
        )

    def start(self) -> None:
        """Connect and start the background upload worker."""
        if self._started:
            return
        if self._closed:
            raise RuntimeError("FirebaseTelemetrySync has already been closed.")

        self.connect()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="Firebase-Telemetry-Worker",
            daemon=True,
        )
        self.worker.start()
        self._started = True

    def enqueue(self, event: Dict[str, Any]) -> bool:
        """Queue a copy of one normalized event for background upload."""
        if not self._started or self._closed:
            self.logger.error("Cannot queue telemetry: Firebase worker is not running.")
            return False

        try:
            self.event_queue.put_nowait(copy.deepcopy(event))
            return True
        except queue.Full:
            self.logger.error(
                "Firebase telemetry queue is full; event was not queued."
            )
            return False

    def write_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Write one event directly; intended for connection/write testing."""
        if self.db is None:
            raise RuntimeError("Firestore is not connected.")

        payload = copy.deepcopy(event)
        started = time.perf_counter()
        document_ref = self.db.collection(self.collection_name).document()
        document_ref.set(payload)
        sync_latency_ms = round((time.perf_counter() - started) * 1000, 2)

        self.logger.info(
            "Firestore event uploaded | document=%s | sync_latency_ms=%.2f",
            document_ref.id,
            sync_latency_ms,
        )
        return {
            "document_id": document_ref.id,
            "sync_latency_ms": sync_latency_ms,
        }

    def _worker_loop(self) -> None:
        while True:
            item = self.event_queue.get()
            try:
                if item is self.stop_token:
                    return

                for attempt in range(1, self.max_retries + 1):
                    try:
                        self.write_event(item)
                        break
                    except Exception:
                        if attempt >= self.max_retries:
                            self.logger.exception(
                                "Firestore upload failed after %d attempts.",
                                self.max_retries,
                            )
                        else:
                            delay = min(2 ** (attempt - 1), 4)
                            self.logger.warning(
                                "Firestore upload attempt %d/%d failed; retrying in %ds.",
                                attempt,
                                self.max_retries,
                                delay,
                            )
                            time.sleep(delay)
            finally:
                self.event_queue.task_done()

    def close(self, timeout: float = 30.0) -> None:
        """Flush queued events and stop the worker cleanly."""
        if self._closed:
            return
        self._closed = True

        if self.worker is not None and self.worker.is_alive():
            self.event_queue.join()
            self.event_queue.put(self.stop_token)
            self.worker.join(timeout=timeout)
            if self.worker.is_alive():
                self.logger.warning(
                    "Firebase worker did not stop within %.1f seconds.", timeout
                )

        self.logger.info("Firebase telemetry worker stopped.")
