import numpy as np
import pyaudio
import time
from faster_whisper import WhisperModel

# Load the model — "base.en" is fast and reasonably accurate
# Options: tiny.en, base.en, small.en, etc.
model_size = "base.en"
model = WhisperModel(model_size, compute_type="int8", device="cpu")

# Audio config
CHUNK_DURATION = 10       # seconds
SAMPLE_RATE = 16000      # required for Whisper
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open microphone stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024)

print("Listening... (press Ctrl+C to stop)")

try:
    while True:
        frames = []
        print("\nRecording chunk...")
        for _ in range(0, int(SAMPLE_RATE / 1024 * CHUNK_DURATION)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0

        print("Transcribing...")
        segments, _ = model.transcribe(audio_np, beam_size=1)

        for segment in segments:
            print(f">>> {segment.text.strip()}")

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
