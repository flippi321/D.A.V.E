import numpy as np
import pyaudio
from faster_whisper import WhisperModel

class dave_speech_recognition:
    def __init__(self):
        self.TRIGGER_WORDS    = ["hey", "Hey", "hi", "Hi", "hello", "Hello"]
        self.SAMPLE_RATE      = 16000
        self.CHUNK_DURATION   = 0.5      # seconds
        self.SPEECH_GRACE     = 4.0      # seconds of silence allowed before timeout applies
        self.SILENCE_TIMEOUT  = 2.0      # seconds of silence to end a command
        self.SILENCE_TRESHOLD = 0.01    # threshold for silence detection
        self.WHISPER_MODEL    = "base.en"

        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.FORMAT     = pyaudio.paInt16
        self.CHANNELS   = 1

    def setup(self):
        # Load model (CPU-only, int8 quantized)
        self.model = WhisperModel(self.WHISPER_MODEL, compute_type="int8", device="cpu")

        # PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.SAMPLE_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )

    def is_silent(self, audio_chunk):
        return np.abs(audio_chunk).mean() < self.SILENCE_TRESHOLD

    def run(self):
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("Listening...")

        try:
            buffer = []
            triggered = False
            total_chunks = 0
            silent_chunks = 0

            while True:
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer.append(audio_chunk)

                # Join small chunks into 2s chunks for trigger word detection
                if not triggered:
                    if len(buffer) >= int(2 / self.CHUNK_DURATION):  # 2s of audio
                        audio_window = np.concatenate(buffer[-int(2 / self.CHUNK_DURATION):])
                        segments, _ = self.model.transcribe(audio_window, beam_size=1)
                        for seg in segments:
                            text = seg.text.lower().strip()
                            if any(w in text for w in self.TRIGGER_WORDS):
                                print(">>> Hello.")
                                triggered = True
                                buffer.clear()  # clear buffer to start clean
                                break
                else:
                    # Record until we get 2 seconds of silence
                    if self.is_silent(audio_chunk):
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    total_chunks += 1
                    
                    # If we have enough silence, stop recording and transcribe
                    if silent_chunks >= int(self.SILENCE_TIMEOUT / self.CHUNK_DURATION) and total_chunks >= int(self.SPEECH_GRACE / self.CHUNK_DURATION):
                        # Stop recording and transcribe
                        print("Transcribing your request...")
                        full_audio = np.concatenate(buffer)
                        segments, _ = self.model.transcribe(full_audio, beam_size=1)
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
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
