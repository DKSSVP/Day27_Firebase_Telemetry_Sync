import json
import logging
import os
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict

import pyaudio
from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "whisper-large-v3-turbo"

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024
MICROPHONE_INDEX = 1

ACOUSTIC_LOGPROB_THRESHOLD = -1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("AudioAnalyzer")


class AudioAnalyzer:
    """Groq transcription and lightweight speech metrics for temporary WAV chunks."""

    def __init__(self, api_key=GROQ_API_KEY):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL
        logger.info("Groq Audio Analyzer initialized using %s", self.model)

    def transcribe_audio(self, audio_path: Path) -> Dict[str, Any]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        request_start = time.perf_counter()
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                temperature=0.0,
            )

        latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
        return self._process_response(response, audio_path, latency_ms)

    def _process_response(self, response, audio_path, api_latency_ms):
        transcript = (getattr(response, "text", "") or "").strip()
        segments = getattr(response, "segments", None) or []
        word_count = len(re.findall(r"\b[\w']+\b", transcript))

        duration = getattr(response, "duration", None)
        if duration is None:
            with wave.open(str(audio_path), "rb") as wav_file:
                duration = wav_file.getnframes() / float(wav_file.getframerate())
        duration = float(duration)

        wpm = (word_count / duration) * 60 if duration > 0 else 0.0
        segment_results = []
        low_confidence = []
        logprobs = []

        for segment in segments:
            if isinstance(segment, dict):
                get_value = segment.get
            else:
                get_value = lambda key, default=None: getattr(segment, key, default)

            avg_logprob = get_value("avg_logprob")
            item = {
                "start": get_value("start"),
                "end": get_value("end"),
                "text": (get_value("text", "") or "").strip(),
                "avg_logprob": float(avg_logprob) if avg_logprob is not None else None,
                "no_speech_prob": (
                    float(get_value("no_speech_prob"))
                    if get_value("no_speech_prob") is not None else None
                ),
            }
            segment_results.append(item)

            if avg_logprob is not None:
                score = float(avg_logprob)
                logprobs.append(score)
                if score < ACOUSTIC_LOGPROB_THRESHOLD:
                    low_confidence.append(item)

        triggered = bool(low_confidence)
        overall_logprob = (
            sum(logprobs) / len(logprobs) if logprobs else None
        )

        if triggered:
            flag_type = "BACKGROUND_WHISPER"
        elif logprobs:
            flag_type = "NONE"
        else:
            flag_type = "METADATA_UNAVAILABLE"

        return {
            "transcript": transcript,
            "word_count": word_count,
            "duration_seconds": round(duration, 3),
            "words_per_minute": round(wpm, 2),
            "acoustic_anomaly": {
                "triggered": triggered,
                "avg_logprob": (
                    round(overall_logprob, 4)
                    if overall_logprob is not None else None
                ),
                "flag_type": flag_type,
                "low_confidence_segment_count": len(low_confidence),
            },
            "speech_quality": {
                "triggered": False,
                "word_count": word_count,
                "duration_seconds": round(duration, 3),
                "words_per_minute": round(wpm, 2),
                "flag_type": "METRICS_ONLY",
            },
            "segments": segment_results,
            "api_latency_ms": api_latency_ms,
        }

    @staticmethod
    def analyze_pcm_window(pcm_bytes: bytes) -> Dict[str, Any]:
        """Write a temporary WAV for Groq and always remove it after analysis."""
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

            return self.transcribe_audio(temp_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def record_and_analyze_once(self):
        """Optional standalone ten-second test; temporary WAV is deleted."""
        interface = pyaudio.PyAudio()
        stream = None
        frames = []
        try:
            stream = interface.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=MICROPHONE_INDEX,
                frames_per_buffer=AUDIO_CHUNK,
            )
            required_frames = AUDIO_RATE * 10
            captured = 0
            while captured < required_frames:
                count = min(AUDIO_CHUNK, required_frames - captured)
                frames.append(
                    stream.read(count, exception_on_overflow=False)
                )
                captured += count
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            interface.terminate()

        return self.analyze_pcm_window(b"".join(frames))


if __name__ == "__main__":
    try:
        analyzer = AudioAnalyzer()
        result = analyzer.record_and_analyze_once()
        print("\nDAY26 AUDIO ANALYSIS RESULT")
        print("=" * 55)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("\nTemporary audio was removed after analysis.")
    except KeyboardInterrupt:
        print("\nAudio recording interrupted.")
    except Exception:
        logger.exception("Audio analysis failed.")
