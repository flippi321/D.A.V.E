import numpy as np
import pyaudio
import time
from faster_whisper import WhisperModel
import collections

# === CONFIGURATION ===
TRIGGER_WORDS = ["dave", "hey dave"]
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5  # seconds
SILENCE_TIMEOUT = 2.0  # seconds of silence to end a command
MODEL_SIZE = "tiny.en"

# === SETUP ===
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Load model (CPU-only, int8 quantized)
model = WhisperModel(MODEL_SIZE, compute_type="int8", device="cpu")

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

def is_silent(audio_chunk, threshold=0.01):
    return np.abs(audio_chunk).mean() < threshold

print("Listening...")

try:
    buffer = []
    triggered = False
    silent_chunks = 0
    last_trigger_time = 0

    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        buffer.append(audio_chunk)

        # Join small chunks into 2s chunks for trigger word detection
        if not triggered:
            if len(buffer) >= int(2 / CHUNK_DURATION):  # 2s of audio
                audio_window = np.concatenate(buffer[-int(2 / CHUNK_DURATION):])
                segments, _ = model.transcribe(audio_window, beam_size=1)
                for seg in segments:
                    text = seg.text.lower().strip()
                    if any(w in text for w in TRIGGER_WORDS):
                        print(">>> Hello.")
                        triggered = True
                        buffer = []  # clear buffer to start clean
                        break
        else:
            # Record until we get 2 seconds of silence
            if is_silent(audio_chunk):
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks >= int(SILENCE_TIMEOUT / CHUNK_DURATION):
                # Stop recording and transcribe
                print("Transcribing your request...")
                full_audio = np.concatenate(buffer)
                segments, _ = model.transcribe(full_audio, beam_size=1)
                for seg in segments:
                    print(f">>> {seg.text.strip()}")
                print("Listening again...\n")
                buffer = []
                triggered = False
                silent_chunks = 0
            else:
                buffer.append(audio_chunk)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
