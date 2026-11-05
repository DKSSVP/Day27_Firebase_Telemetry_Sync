import json
import logging
import queue
import threading
import time
import wave
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pyaudio

from alert_engine import ProctorAlertEngine
from audio_analyzer import (
    AudioAnalyzer,
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_CHUNK,
    MICROPHONE_INDEX,
)
from gaze_analyzer import GazeAnalyzer
from firebase_sync import FirebaseTelemetrySync


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0
CANDIDATE_ID = "CANDIDATE_001"
SESSION_ID = "DAY27_SESSION_001"

VIDEO_OUTPUT = OUTPUT_DIR / "day27_video.mp4"
AUDIO_OUTPUT = OUTPUT_DIR / "day27_audio.wav"

ACTIVITY_LOG = LOG_DIR / "proctor_activity.jsonl"
ALERT_LOG = LOG_DIR / "proctor_alerts.jsonl"

VIDEO_FPS_FALLBACK = 30.0
VIDEO_QUEUE_SIZE = 500
AUDIO_QUEUE_SIZE = 1000
ANALYSIS_QUEUE_SIZE = 12
ANALYSIS_WINDOW_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(threadName)s | %(message)s",
)
logger = logging.getLogger("Day27Main")


# ---------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------
stop_event = threading.Event()
start_event = threading.Event()

video_queue = queue.Queue(maxsize=VIDEO_QUEUE_SIZE)
audio_record_queue = queue.Queue(maxsize=AUDIO_QUEUE_SIZE)
audio_analysis_queue = queue.Queue(maxsize=ANALYSIS_QUEUE_SIZE)

state_lock = threading.Lock()
engine_lock = threading.Lock()
activity_lock = threading.Lock()

latest_gaze_result = {
    "face_detected": False,
    "triggered": False,
    "duration_seconds": 0.0,
    "vector_shift": {"x": 0.0, "y": 0.0},
    "status": "INITIALIZING",
}

event_count = 0
audio_windows_processed = 0
gaze_alert_count = 0
processed_frames = 0
processed_audio_chunks = 0
analysis_windows_dropped = 0
video_info = {"fps": VIDEO_FPS_FALLBACK}


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def write_activity(event_type, severity="INFO", details=None):
    """Append one structured activity record; never write raw exceptions."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID,
        "session_id": SESSION_ID,
        "event_type": event_type,
        "severity_level": severity,
        "details": details or {},
    }
    with activity_lock:
        with ACTIVITY_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


alert_engine = ProctorAlertEngine(
    candidate_id=CANDIDATE_ID,
    session_id=SESSION_ID,
    log_file=str(ALERT_LOG),
)
audio_analyzer = AudioAnalyzer()
firebase_sync = None


def publish_central_alert(gaze=None, acoustic=None, speech_quality=None):
    global event_count
    started = time.perf_counter()
    try:
        with engine_lock:
            event = alert_engine.process_alerts(
                gaze=gaze,
                acoustic=acoustic,
                speech_quality=speech_quality,
                processing_start_time=started,
            )
        event_count += 1
        if firebase_sync is not None:
            queued = firebase_sync.enqueue(event)
            if not queued:
                write_activity("FIREBASE_EVENT_QUEUE_FAILED", "ERROR")
            else:
                write_activity(
                    "FIREBASE_EVENT_QUEUED",
                    details={"severity_level": event.get("severity_level")},
                )
        write_activity(
            "ALERT_GENERATED",
            severity=event.get("severity_level", "WARNING"),
            details={
                "alert_log": str(ALERT_LOG),
                "severity_level": event.get("severity_level"),
                "latency_ms": event.get("latency_ms"),
            },
        )
        return event
    except Exception:
        logger.exception("Central alert processing failed")
        write_activity("ALERT_PROCESSING_FAILED", "ERROR")
        return None


# ---------------------------------------------------------
# QUEUE HELPERS
# ---------------------------------------------------------
def enqueue_with_stop(data_queue, item):
    while not stop_event.is_set():
        try:
            data_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def enqueue_latest_video(frame):
    """Keep the video queue bounded; preserve recording frames when possible."""
    while not stop_event.is_set():
        try:
            video_queue.put(frame, timeout=0.1)
            return True
        except queue.Full:
            # The writer/main loop is behind. Wait rather than silently discard.
            continue
    return False


# ---------------------------------------------------------
# CAPTURE WORKERS
# ---------------------------------------------------------
def webcam_capture(camera):
    global latest_gaze_result, gaze_alert_count

    analyzer = None
    previous_triggered = False
    try:
        start_event.wait()
        analyzer = GazeAnalyzer()
        write_activity("CAMERA_CAPTURE_STARTED")
        logger.info("Webcam capture and gaze analysis started.")

        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                logger.error("Webcam frame capture failed.")
                write_activity("CAMERA_CAPTURE_FAILED", "ERROR")
                stop_event.set()
                break

            frame = cv2.flip(frame, 1)
            gaze = analyzer.analyze_frame(frame)

            with state_lock:
                latest_gaze_result = dict(gaze)

            status = gaze.get("status", "UNKNOWN")
            triggered = bool(gaze.get("triggered", False))
            duration = float(gaze.get("duration_seconds", 0.0))
            color = (0, 0, 255) if triggered else (0, 255, 0)

            cv2.putText(
                frame, f"Gaze: {status}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.putText(
                frame, f"Deviation: {duration:.2f}s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.putText(
                frame, "Q / ESC: Stop", (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

            if triggered and not previous_triggered:
                gaze_alert_count += 1
                gaze_payload = {
                    "triggered": True,
                    "duration_seconds": duration,
                    "vector_shift": gaze.get(
                        "vector_shift", {"x": 0.0, "y": 0.0}
                    ),
                }
                write_activity(
                    "GAZE_DEVIATION",
                    "WARNING",
                    gaze_payload,
                )
                publish_central_alert(
                    gaze=gaze_payload,
                    acoustic={
                        "triggered": False,
                        "avg_logprob": None,
                        "flag_type": "NONE",
                    },
                    speech_quality={
                        "triggered": False,
                        "flag_type": "NOT_ANALYZED",
                    },
                )
            previous_triggered = triggered

            if not enqueue_latest_video(frame.copy()):
                break

    except Exception:
        logger.exception("Webcam worker failed")
        write_activity("CAMERA_WORKER_FAILED", "ERROR")
        stop_event.set()
    finally:
        if analyzer is not None:
            analyzer.close()
        logger.info("Webcam capture worker stopped.")
        write_activity("CAMERA_CAPTURE_STOPPED")


def microphone_capture(stream, audio_interface):
    """Capture continuously; duplicate chunks for file writing and analysis."""
    try:
        start_event.wait()
        write_activity("MICROPHONE_CAPTURE_STARTED")
        logger.info("Continuous microphone capture started.")

        window_frames = []
        frames_in_window = 0
        required_frames = AUDIO_RATE * ANALYSIS_WINDOW_SECONDS

        while not stop_event.is_set():
            data = stream.read(
                AUDIO_CHUNK,
                exception_on_overflow=False,
            )

            if not enqueue_with_stop(audio_record_queue, data):
                break

            window_frames.append(data)
            frames_in_window += AUDIO_CHUNK

            if frames_in_window >= required_frames:
                payload = b"".join(window_frames)
                try:
                    audio_analysis_queue.put_nowait(payload)
                except queue.Full:
                    global analysis_windows_dropped
                    analysis_windows_dropped += 1
                    write_activity(
                        "AUDIO_ANALYSIS_WINDOW_DROPPED",
                        "WARNING",
                        {"reason": "analysis_queue_full"},
                    )
                window_frames = []
                frames_in_window = 0

        # Do not enqueue an incomplete analysis window at shutdown.
    except Exception:
        logger.exception("Microphone capture worker failed")
        write_activity("MICROPHONE_CAPTURE_FAILED", "ERROR")
        stop_event.set()
    finally:
        write_activity("MICROPHONE_CAPTURE_STOPPED")
        logger.info("Microphone capture worker stopped.")


# ---------------------------------------------------------
# AUDIO ANALYSIS WORKER
# ---------------------------------------------------------
def audio_analysis_worker():
    global audio_windows_processed

    write_activity("GROQ_ANALYSIS_WORKER_STARTED")
    while not stop_event.is_set() or not audio_analysis_queue.empty():
        try:
            pcm_bytes = audio_analysis_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="day26_analysis_",
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(AUDIO_CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(AUDIO_RATE)
                wav_file.writeframes(pcm_bytes)

            result = audio_analyzer.transcribe_audio(temp_path)
            audio_windows_processed += 1

            acoustic = result.get("acoustic_anomaly", {})
            speech_quality = result.get("speech_quality", {})

            with state_lock:
                gaze_snapshot = dict(latest_gaze_result)

            gaze_payload = {
                "triggered": bool(gaze_snapshot.get("triggered", False)),
                "duration_seconds": gaze_snapshot.get("duration_seconds", 0.0),
                "vector_shift": gaze_snapshot.get(
                    "vector_shift", {"x": 0.0, "y": 0.0}
                ),
            }

            if acoustic.get("triggered"):
                write_activity(
                    "ACOUSTIC_ANOMALY",
                    "WARNING",
                    {
                        "avg_logprob": acoustic.get("avg_logprob"),
                        "flag_type": acoustic.get("flag_type"),
                        "low_confidence_segment_count": acoustic.get(
                            "low_confidence_segment_count"
                        ),
                    },
                )

            write_activity(
                "SPEECH_ANALYSIS_COMPLETED",
                "INFO",
                {
                    "word_count": result.get("word_count"),
                    "words_per_minute": result.get("words_per_minute"),
                    "api_latency_ms": result.get("api_latency_ms"),
                    "acoustic_anomaly": acoustic.get("triggered", False),
                },
            )

            publish_central_alert(
                gaze=gaze_payload,
                acoustic=acoustic,
                speech_quality=speech_quality,
            )

            logger.info(
                "Groq window %d | words=%s | WPM=%s | anomaly=%s",
                audio_windows_processed,
                result.get("word_count"),
                result.get("words_per_minute"),
                acoustic.get("triggered"),
            )
        except Exception:
            logger.exception("Groq audio analysis failed")
            write_activity("GROQ_ANALYSIS_FAILED", "ERROR")
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary analysis WAV.")
            audio_analysis_queue.task_done()

    write_activity("GROQ_ANALYSIS_WORKER_STOPPED")


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
def main():
    global processed_frames, processed_audio_chunks, firebase_sync

    camera = None
    audio_interface = None
    audio_stream = None
    webcam_thread = None
    microphone_thread = None
    analysis_thread = None
    video_writer = None
    audio_writer = None
    latest_frame = None
    recording_started = None
    fatal_error = None

    # Replace only the previous session's two recording outputs.
    for old_file in (VIDEO_OUTPUT, AUDIO_OUTPUT):
        old_file.unlink(missing_ok=True)

    write_activity("SESSION_STARTED")
    try:
        firebase_sync = FirebaseTelemetrySync.from_env(BASE_DIR)
        firebase_sync.start()
        write_activity(
            "FIREBASE_CONNECTED",
            details={
                "project_id": firebase_sync.project_id,
                "collection": firebase_sync.collection_name,
            },
        )
    except Exception:
        logger.exception("Firebase initialization failed; stopping Day27 startup.")
        write_activity("FIREBASE_CONNECTION_FAILED", "ERROR")
        if firebase_sync is not None:
            firebase_sync.close()
        raise
    logger.info("=" * 64)
    logger.info("DAY27 FIREBASE TELEMETRY SYNC")
    logger.info("Candidate: %s | Session: %s", CANDIDATE_ID, SESSION_ID)

    try:
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            raise RuntimeError(f"Unable to open webcam index {CAMERA_INDEX}")

        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        detected_fps = camera.get(cv2.CAP_PROP_FPS)
        if 1 <= detected_fps <= 120:
            video_info["fps"] = detected_fps

        audio_interface = pyaudio.PyAudio()
        audio_stream = audio_interface.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=MICROPHONE_INDEX,
            frames_per_buffer=AUDIO_CHUNK,
        )

        # Writers are initialized from the first frame/audio block.
        webcam_thread = threading.Thread(
            target=webcam_capture,
            args=(camera,),
            name="Webcam-Gaze-Capture",
        )
        microphone_thread = threading.Thread(
            target=microphone_capture,
            args=(audio_stream, audio_interface),
            name="Microphone-Capture",
        )
        analysis_thread = threading.Thread(
            target=audio_analysis_worker,
            name="Groq-Analysis",
        )

        webcam_thread.start()
        microphone_thread.start()
        analysis_thread.start()

        recording_started = time.perf_counter()
        start_event.set()
        write_activity(
            "RECORDING_STARTED",
            details={
                "audio_file": str(AUDIO_OUTPUT),
                "video_file": str(VIDEO_OUTPUT),
                "video_resolution": [width, height],
                "video_fps": video_info["fps"],
                "audio_sample_rate": AUDIO_RATE,
            },
        )
        logger.info("Audio and video recording started together.")
        logger.info("Press Q or ESC to stop.")

        while not stop_event.is_set():
            # Drain video frames into the MP4 writer and display the latest.
            try:
                while True:
                    frame = video_queue.get_nowait()
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(VIDEO_OUTPUT),
                            fourcc,
                            video_info["fps"],
                            (frame.shape[1], frame.shape[0]),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError("Could not create MP4 video.")
                        write_activity("VIDEO_FILE_CREATED")
                    video_writer.write(frame)
                    processed_frames += 1
                    latest_frame = frame
            except queue.Empty:
                pass

            # Drain captured PCM blocks into the one continuous WAV file.
            try:
                while True:
                    audio_data = audio_record_queue.get_nowait()
                    if audio_writer is None:
                        audio_writer = wave.open(str(AUDIO_OUTPUT), "wb")
                        audio_writer.setnchannels(AUDIO_CHANNELS)
                        audio_writer.setsampwidth(
                            audio_interface.get_sample_size(AUDIO_FORMAT)
                        )
                        audio_writer.setframerate(AUDIO_RATE)
                        write_activity("AUDIO_FILE_CREATED")
                    audio_writer.writeframes(audio_data)
                    processed_audio_chunks += 1
            except queue.Empty:
                pass

            if latest_frame is not None:
                display = latest_frame.copy()
                elapsed = time.perf_counter() - recording_started
                cv2.putText(
                    display,
                    "DAY27 - FIREBASE PROCTOR TELEMETRY",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    f"Recording: {elapsed:.1f}s | Frames: {processed_frames}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display,
                    "Press Q or ESC to stop",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Day27 - Firebase Proctor Telemetry", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                write_activity("STOP_REQUESTED_BY_USER")
                break

            if not webcam_thread.is_alive() or not microphone_thread.is_alive():
                logger.warning("A capture worker stopped unexpectedly.")
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
        write_activity("STOP_REQUESTED_BY_KEYBOARD")
    except Exception:
        logger.exception("Day26 main application failed.")
        write_activity("APPLICATION_FAILED", "ERROR")
        fatal_error = True
    finally:
        stop_event.set()
        start_event.set()

        # Stop device capture before draining the final queued data.
        if audio_stream is not None:
            try:
                if audio_stream.is_active():
                    audio_stream.stop_stream()
            except Exception:
                logger.exception("Error stopping microphone stream.")

        if webcam_thread is not None:
            webcam_thread.join(timeout=10)
        if microphone_thread is not None:
            microphone_thread.join(timeout=10)

        # Drain remaining captured video frames.
        while not video_queue.empty():
            try:
                frame = video_queue.get_nowait()
            except queue.Empty:
                break
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(VIDEO_OUTPUT),
                    fourcc,
                    video_info["fps"],
                    (frame.shape[1], frame.shape[0]),
                )
            video_writer.write(frame)
            processed_frames += 1

        # Drain remaining captured audio blocks.
        while not audio_record_queue.empty():
            try:
                audio_data = audio_record_queue.get_nowait()
            except queue.Empty:
                break
            if audio_writer is None and audio_interface is not None:
                audio_writer = wave.open(str(AUDIO_OUTPUT), "wb")
                audio_writer.setnchannels(AUDIO_CHANNELS)
                audio_writer.setsampwidth(
                    audio_interface.get_sample_size(AUDIO_FORMAT)
                )
                audio_writer.setframerate(AUDIO_RATE)
            if audio_writer is not None:
                audio_writer.writeframes(audio_data)
                processed_audio_chunks += 1

        # Allow queued analysis windows to finish, then stop the worker.
        if analysis_thread is not None:
            analysis_thread.join(timeout=60)
            if analysis_thread.is_alive():
                logger.warning("Groq analysis worker did not finish in time.")

        if camera is not None:
            camera.release()
        if audio_stream is not None:
            try:
                audio_stream.close()
            except Exception:
                logger.exception("Error closing microphone stream.")
        if audio_interface is not None:
            audio_interface.terminate()

        if video_writer is not None:
            video_writer.release()
            write_activity("VIDEO_RECORDING_SAVED", details={"file": str(VIDEO_OUTPUT)})
        if audio_writer is not None:
            audio_writer.close()
            write_activity("AUDIO_RECORDING_SAVED", details={"file": str(AUDIO_OUTPUT)})

        if firebase_sync is not None:
            try:
                firebase_sync.close()
                write_activity("FIREBASE_SYNC_SHUTDOWN_COMPLETED")
            except Exception:
                logger.exception("Firebase sync shutdown failed.")
                write_activity("FIREBASE_SYNC_SHUTDOWN_FAILED", "ERROR")

        cv2.destroyAllWindows()

        duration = (
            round(time.perf_counter() - recording_started, 2)
            if recording_started is not None else 0.0
        )
        write_activity(
            "SESSION_STOPPED",
            details={
                "recording_duration_seconds": duration,
                "video_frames": processed_frames,
                "audio_chunks": processed_audio_chunks,
                "audio_analysis_windows": audio_windows_processed,
                "analysis_windows_dropped": analysis_windows_dropped,
                "audio_file_exists": AUDIO_OUTPUT.exists(),
                "video_file_exists": VIDEO_OUTPUT.exists(),
            },
        )

        logger.info("=" * 64)
        logger.info("DAY27 SESSION SUMMARY")
        logger.info("Duration: %.2f seconds", duration)
        logger.info("Video frames written: %d", processed_frames)
        logger.info("Audio chunks written: %d", processed_audio_chunks)
        logger.info("Groq analysis windows: %d", audio_windows_processed)
        logger.info("Gaze alerts: %d | Central events: %d", gaze_alert_count, event_count)
        logger.info("Audio output: %s", AUDIO_OUTPUT)
        logger.info("Video output: %s", VIDEO_OUTPUT)
        logger.info("Activity log: %s", ACTIVITY_LOG)
        logger.info("Alert log: %s", ALERT_LOG)
        logger.info("DAY27 STOPPED")
        logger.info("=" * 64)


if __name__ == "__main__":
    main()