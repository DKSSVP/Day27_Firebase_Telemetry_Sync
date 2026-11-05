import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from firebase_sync import FirebaseTelemetrySync


BASE_DIR = Path(__file__).resolve().parent


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    sync = FirebaseTelemetrySync.from_env(BASE_DIR)
    try:
        sync.connect()

        test_event = {
            "candidate_id": "DAY27_TEST_CANDIDATE",
            "session_id": "DAY27_FIREBASE_CONNECTION_TEST",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts": {
                "gaze_deviation": {
                    "triggered": False,
                    "duration_seconds": 0.0,
                    "vector_shift": {"x": 0.0, "y": 0.0},
                },
                "acoustic_anomaly": {
                    "triggered": False,
                    "avg_logprob": None,
                    "flag_type": "NONE",
                },
                "speech_quality": {
                    "triggered": False,
                    "word_count": 0,
                    "duration_seconds": 0.0,
                    "words_per_minute": 0.0,
                    "flag_type": "TEST_EVENT",
                },
            },
            "severity_level": "NORMAL",
            "latency_ms": 0.0,
        }

        result = sync.write_event(test_event)
        print("\nFirebase connection and write test successful.")
        print(json.dumps(result, indent=4))
        print(f"Project ID: {sync.project_id}")
        print(f"Collection: {sync.collection_name}")
    finally:
        # This test writes directly and does not start the background worker.
        pass


if __name__ == "__main__":
    main()
