import json
import logging
import threading
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class ProctorAlertEngine:
    """
    Centralized engine for normalizing, recording, and preparing proctoring alerts for telemetry sync.

    Alert sources:
        1. Gaze deviation
        2. Acoustic anomaly
        3. Speech quality assessment

    Severity policy:
        WARNING  - One or more alert types are triggered.
        CRITICAL - Both gaze deviation and acoustic anomaly
                   are triggered in the same event.
        NORMAL   - No alerts are triggered.
    """

    def __init__(
        self,
        candidate_id: str,
        session_id: str,
        log_file: str = "logs/proctor_alerts.jsonl",
    ):
        if not candidate_id or not session_id:
            raise ValueError("candidate_id and session_id are required.")

        self.candidate_id = candidate_id
        self.session_id = session_id

        self.log_path = Path(log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger("ProctorAlertEngine")

    @staticmethod
    def _timestamp() -> str:
        """Return a UTC ISO-8601 timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_gaze(
        triggered: bool = False,
        duration_seconds: float = 0.0,
        vector_shift: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        vector_shift = vector_shift or {"x": 0.0, "y": 0.0}

        return {
            "triggered": bool(triggered),
            "duration_seconds": round(float(duration_seconds), 3),
            "vector_shift": {
                "x": round(float(vector_shift.get("x", 0.0)), 4),
                "y": round(float(vector_shift.get("y", 0.0)), 4),
            },
        }

    @staticmethod
    def _normalize_acoustic(
        triggered: bool = False,
        avg_logprob: Optional[float] = None,
        flag_type: str = "NONE",
    ) -> Dict[str, Any]:

        return {
            "triggered": bool(triggered),
            "avg_logprob": (
                round(float(avg_logprob), 4)
                if avg_logprob is not None
                else None
            ),
            "flag_type": str(flag_type),
        }

    @staticmethod
    def _normalize_speech_quality(
        triggered: bool = False,
        word_count: int = 0,
        duration_seconds: float = 0.0,
        words_per_minute: float = 0.0,
        flag_type: str = "NONE",
    ) -> Dict[str, Any]:

        return {
            "triggered": bool(triggered),
            "word_count": int(word_count),
            "duration_seconds": round(float(duration_seconds), 3),
            "words_per_minute": round(float(words_per_minute), 2),
            "flag_type": str(flag_type),
        }

    @staticmethod
    def _calculate_severity(alerts: Dict[str, Any]) -> str:
        """
        CRITICAL when gaze and acoustic alerts occur together.
        WARNING when any individual alert is triggered.
        NORMAL when no alerts are triggered.
        """

        gaze_triggered = alerts["gaze_deviation"]["triggered"]
        acoustic_triggered = alerts["acoustic_anomaly"]["triggered"]

        speech_triggered = alerts["speech_quality"]["triggered"]

        if gaze_triggered and acoustic_triggered:
            return "CRITICAL"

        if gaze_triggered or acoustic_triggered or speech_triggered:
            return "WARNING"

        return "NORMAL"

    def create_event(
        self,
        gaze: Optional[Dict[str, Any]] = None,
        acoustic: Optional[Dict[str, Any]] = None,
        speech_quality: Optional[Dict[str, Any]] = None,
        processing_start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Create a normalized proctoring event.

        processing_start_time should be a time.perf_counter()
        value captured when processing the event began.
        """

        start_time = (
            processing_start_time
            if processing_start_time is not None
            else time.perf_counter()
        )

        gaze = gaze or {}
        acoustic = acoustic or {}
        speech_quality = speech_quality or {}

        alerts = {
            "gaze_deviation": self._normalize_gaze(
                triggered=gaze.get("triggered", False),
                duration_seconds=gaze.get("duration_seconds", 0.0),
                vector_shift=gaze.get("vector_shift"),
            ),
            "acoustic_anomaly": self._normalize_acoustic(
                triggered=acoustic.get("triggered", False),
                avg_logprob=acoustic.get("avg_logprob"),
                flag_type=acoustic.get("flag_type", "NONE"),
            ),
            "speech_quality": self._normalize_speech_quality(
                triggered=speech_quality.get("triggered", False),
                word_count=speech_quality.get("word_count", 0),
                duration_seconds=speech_quality.get(
                    "duration_seconds", 0.0
                ),
                words_per_minute=speech_quality.get(
                    "words_per_minute", 0.0
                ),
                flag_type=speech_quality.get("flag_type", "NONE"),
            ),
        }

        latency_ms = round(
            max(0.0, time.perf_counter() - start_time) * 1000,
            2,
        )

        return {
            "candidate_id": self.candidate_id,
            "session_id": self.session_id,
            "timestamp": self._timestamp(),
            "alerts": alerts,
            "severity_level": self._calculate_severity(alerts),
            "latency_ms": latency_ms,
        }

    def publish_event(self, event: Dict[str, Any]) -> None:
        """Append one normalized event to the JSONL log."""

        with self.lock:
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

        self.logger.info(
            "Proctor alert published | Severity: %s | Candidate: %s",
            event["severity_level"],
            event["candidate_id"],
        )

    def process_alerts(
        self,
        gaze: Optional[Dict[str, Any]] = None,
        acoustic: Optional[Dict[str, Any]] = None,
        speech_quality: Optional[Dict[str, Any]] = None,
        processing_start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create, normalize, and publish one alert event."""

        event = self.create_event(
            gaze=gaze,
            acoustic=acoustic,
            speech_quality=speech_quality,
            processing_start_time=processing_start_time,
        )

        self.publish_event(event)

        return event


if __name__ == "__main__":
    engine = ProctorAlertEngine(
        candidate_id="TEST_CANDIDATE_001",
        session_id="TEST_SESSION_001",
    )

    test_event = engine.process_alerts(
        gaze={
            "triggered": True,
            "duration_seconds": 2.4,
            "vector_shift": {"x": -0.35, "y": 0.12},
        },
        acoustic={
            "triggered": True,
            "avg_logprob": -1.34,
            "flag_type": "BACKGROUND_WHISPER",
        },
        speech_quality={
            "triggered": False,
            "word_count": 120,
            "duration_seconds": 60,
            "words_per_minute": 120,
        },
    )

    print("\nGenerated Proctor Alert Event:")
    print(json.dumps(test_event, indent=4))