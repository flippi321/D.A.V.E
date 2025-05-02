import numpy as np
import pyaudio
from faster_whisper import WhisperModel

class dave_speech_recognition:
    def __init__(self):
        self.TRIGGER_WORDS    = ["hey", "Hey", "hi", "Hi", "hello", "Hello", "Test", "test"]
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
    
    def read_query(self, buffer, audio_chunk, total_chunks, silent_chunks):
        print("Recording your request...")
        # Record until we get 2 seconds of silence
        if self.is_silent(audio_chunk):
            silent_chunks += 1
        else:
            silent_chunks = 0
        total_chunks += 1

        # Check if we have enough silence and are past the speech grace period
        silence_cap_reached = silent_chunks > int(self.SILENCE_TIMEOUT / self.CHUNK_DURATION)
        past_speech_grace = total_chunks >= int(self.SPEECH_GRACE / self.CHUNK_DURATION)
        
        # If we have enough silence and we are past the speech grace period, transcribe
        if silence_cap_reached and past_speech_grace:
            # Stop recording and transcribe
            print("Transcribing your request...")
            full_audio = np.concatenate(buffer)
            segments, _ = self.model.transcribe(full_audio, beam_size=1)
            
            return segments
        else:
            buffer.append(audio_chunk)

    def listen(self):
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print("Listening...")

        try:
            buffer = []
            total_chunks = 0
            silent_chunks = 0

            # TODO CHANGE
            while True:
                data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                buffer.append(audio_chunk)

                # Join small chunks into 2s chunks for trigger word detection
                if len(buffer) >= int(2 / self.CHUNK_DURATION):
                    audio_window = np.concatenate(buffer[-int(2 / self.CHUNK_DURATION):])
                    segments, _ = self.model.transcribe(audio_window, beam_size=1)
                    for seg in segments:
                        text = seg.text.lower().strip()
                        if any(w in text for w in self.TRIGGER_WORDS):

                            # If trigger word detected, start recording
                            query = self.read_query(buffer, audio_chunk, total_chunks, silent_chunks)
                            print(query)
                            # Reset the buffer and counters
                            buffer = []
                            silent_chunks = 0
                    

        except KeyboardInterrupt:
            print("\nExiting...")

        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
