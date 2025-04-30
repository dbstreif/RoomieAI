import os
import sys
import json
from time import sleep
from collections.abc import Callable
from threading import Thread, Lock, Event
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
import logging
from wake_word_detector import WakeWordDetector
import numpy as np
import librosa

logger = logging.getLogger("transcription_logger")
# logging.basicConfig(level=logging.DEBUG)
logging.disable(logging.CRITICAL)
SetLogLevel(-1)

class Transcriber:
    def __init__(self, model_path, wakewords, wakeword_exclusions) -> None:
        self.wakewords: list[str] = wakewords 
        self.wakeword_exclusions: list[str] = wakeword_exclusions
        self.chunk_size: int = 4096 # Adjust chunk size for performance (larger = faster but less responsive)
        self.formatting = pyaudio.paInt16
        self.channels: int = 1
        self.rate: int = 48000# Vosk models typically expect 16KHz audio
        self.data_callback: Callable = None
        self.err_callback: Callable = None
        self.command_callback: Callable = None

        self.p = pyaudio.PyAudio()
        self.stream = self.__init_audio_stream()
        if self.stream is None:
            raise ValueError("Stream initialization failed")

        self.model = self.__load_vosk_model(model_path)
        if self.model is None:
            raise ValueError("Model initialization failed")

        self.recognizer = self.__load_recognizer()
        if self.recognizer is None:
            raise ValueError("Recognizer initialization failed")

        self.stream_bool = False
        self.pause_bool = False
        self.stop_event = Event()
        self.pause_event = Event()
        self.stream_lock = Lock()


    def __init_audio_stream(self):
        # Initialize PyAudio stream
        try:
            stream = self.p.open(
                format=self.formatting,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                start=False
            )
            return stream

        except OSError as e:
            logger.error("Failed to open audio stream: %s", e)
            return None


    def __transcribe(self) -> None:
        logger.info("Transcription thread started.")
        try:
            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    self.stream.read(self.chunk_size, exception_on_overflow=False)
                    sleep(0.3)
                    continue

                # Read audio data from the microphone
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) 
                    resampled_audio = librosa.resample( 
                        audio_array, 
                        orig_sr=self.rate,
                        target_sr=16000
                    ).astype(np.int16)
                    data = resampled_audio.tobytes()
                except Exception as e:
                    logger.error("Error reading audio data: %s", e)
                    sleep(1)
                    continue

                # Continue the loop if no data is received
                if len(data) == 0:
                    logger.warning("No audio data received. Retrying...")
                    continue

                # Process the audio data
                if self.recognizer.AcceptWaveform(data):
                    result = self.recognizer.Result()

                    # Validate and parse the result
                    try:
                        result_dict = json.loads(result)
                        result_dict['command'] = False
                        text = None

                        if len(result_dict['text']) != 0:
                            text = result_dict['text'].lower().split()
                        else:
                            continue

                        # Post processing wake word detection
                        wake_word_detector = WakeWordDetector(text[0], self.wakewords, self.wakeword_exclusions, 65)
                        if wake_word_detector.process_result():
                            result_dict['command'] = True
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse JSON result: %s", e)
                        continue

                    # Call the callback function with the transcription result
                    try:
                        if self.data_callback is not None:
                            self.data_callback(result_dict)
                    except Exception as e:
                        logger.error("Error in data callback: %s", e)

                    # Call the command callback function with the transcription result
                    try:
                        if result_dict['command'] and self.command_callback is not None:
                            self.command_callback(result_dict)
                    except Exception as e:
                        logger.error("Error in command callback: %s", e)

                    # Log the transcribed text
                    text = result_dict.get("text", "")
                    if text:
                        logger.debug("You said: %s", text)

        except Exception as e:
            logger.error("Unexpected error in transcription thread: %s", e)
            self.err_callback("Unexpected error in transcription thread: " + str(e))

        finally:
            logger.debug("Transcription Halted")
            self.stream.stop_stream()
            self.__close()

        """
        else:
            # Handle partial results for real-time feedback
            partial_result = recognizer.PartialResult()
            partial_result_dict = json.loads(partial_result)
            partial_text = partial_result_dict.get("partial", "")
            if partial_text:
                print(f"Partial: {partial_text}", end="\r")
        """


    def __load_recognizer(self) -> KaldiRecognizer:
        if self.model is not None and self.rate != 0:
            recognizer = KaldiRecognizer(self.model, self.rate)
            logger.debug("Recognizer registered successfully")
        else:
            recognizer = None
            logger.error("Recognizer failed to register!")
        return recognizer


    def __load_vosk_model(self, model_path) -> Model:
        # Ensure the model path exists
        if not os.path.exists(model_path):
            logger.error("Model path '%s' does not exist.", model_path)
            return None

        # Load the Vosk model
        try:
            model = Model(model_path)
            logger.debug("Vosk Model loaded successfully")
        except Exception as e:
            logger.error("Vosk Model failed to load: %s", e)
            model = None
        return model


    def register_data_callback(self, func: Callable) -> None:
        if callable(func):
            self.data_callback = func
            logger.debug("Data callback registered successfully")
        else:
            self.data_callback = None
            logger.error("Data callback function not of callable type!")


    def register_err_callback(self, func: Callable) -> None:
        if callable(func):
            self.err_callback = func
            logger.debug("Error callback registered successfully")
        else:
            self.err_callback = None
            logger.error("Error callback function not of callable type!")

    def register_command_callback(self, func: Callable) -> None:
        if callable(func):
            self.command_callback = func
            logger.debug("Command callback registered successfully")
        else:
            self.command_callback = None
            logger.error("Command callback function not of callable type!")


    def start_transcription(self) -> None:
        """
        Perform real-time speech-to-text transcription using Vosk and PyAudio.
        """
        if self.err_callback is None:
            raise ValueError("Error callback function must be set to begin transcription")
        if self.stream_bool:
            logger.error("Cannot launch new transcriber. Transcription active already.")
            self.err_callback("Cannot launch new transcriber. Transcription active already.")
            return

        print("Listening...")
        with self.stream_lock:
            self.stream_bool = True

        self.stop_event.clear()
        self.stream.start_stream()

        trans_thread = Thread(target=self.__transcribe, name="TranscriptionThread")
        trans_thread.start()


    def pause_transcription(self) -> None:
        if not self.pause_bool:
            self.pause_event.set()
            self.pause_bool = True
            logger.info("Audio capture paused")
        else:
            logger.warning("Audio capture already paused!")
            self.err_callback("Audio capture already paused!")


    def resume_transcription(self) -> None:
        if self.pause_bool:
            self.pause_event.clear()
            self.pause_bool = False
            print("Listening...")
            logger.info("Audio capture resumed")
        else:
            logger.warning("Audio capture already active!")
            self.err_callback("Audio capture already active!")


    def stop_transcription(self) -> None:
        if not self.stream_bool:
            logger.error("Transcriber already in stopped state!")
            self.err_callback("Transcriber already in stopped state!")
            return

        with self.stream_lock:
            self.stream_bool = False
            self.stop_event.set()
        logger.debug("Transcription halt requested")


    def __close(self) -> None:
        with self.stream_lock:
            while not self.stream.is_stopped():
                sleep(0.1)
            self.stream.close()
            self.p.terminate()
        logger.debug("All streams terminated!")
        logger.debug("Exiting Gracefully...")
