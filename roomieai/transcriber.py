import os
import sys
import json
from collections.abc import Callable
from threading import Thread, Lock
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel
import logging

logger = logging.getLogger("transcription_logger")
# logging.basicConfig(level=logging.DEBUG)
logging.disable(logging.CRITICAL)
SetLogLevel(-1)

class Transcriber:
    def __init__(self, model_path) -> None:
        self.chunk_size: int = 8192 # Adjust chunk size for performance (larger = faster but less responsive)
        self.formatting = pyaudio.paInt16
        self.channels: int = 1
        self.rate: int = 16000 # Vosk models typically expect 16KHz audio
        self.data_callback: Callable = None
        self.err_callback: Callable = None

        self.p = pyaudio.PyAudio()
        self.stream = self.__init_audio_stream()

        self.model = self.__load_vosk_model(model_path)
        if self.model is None:
            raise ValueError("Model initialization failed")

        self.recognizer = self.__load_recognizer()
        if self.recognizer is None:
            raise ValueError("Recognizer initialization failed")

        self.stream_bool = False
        self.stream_lock = Lock()


    def __init_audio_stream(self):
        # Initialize PyAudio stream
        stream = self.p.open(
            format=self.formatting,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            start=False
        )
        return stream


    def __transcribe(self) -> None:
        logger.info("Transcription thread started.")
        try:
            while True:
                # Check if transcription should stop
                with self.stream_lock:
                    if not self.stream_bool:
                        logger.info("Transcription thread stopping.")
                        break

                # Read audio data from the microphone
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    logger.error("Error reading audio data: %s", e)
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
                    except json.JSONDecodeError as e:
                        logger.error("Failed to parse JSON result: %s", e)
                        continue

                    # Call the callback function with the transcription result
                    try:
                        self.data_callback(result_dict)
                    except Exception as e:
                        logger.error("Error in data callback: %s", e)

                    # Log the transcribed text
                    text = result_dict.get("text", "")
                    if text:
                        logger.debug("You said: %s", text)

        except Exception as e:
            logger.error("Unexpected error in transcription thread: %s", e)
            self.err_callback("Unexpected error in transcription thread: %s", e)

        finally:
            self.stream.stop_stream()
            logger.info("Transcription thread has stopped.")

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


    def start_transcription(self) -> None:
        """
        Perform real-time speech-to-text transcription using Vosk and PyAudio.
        """
        with self.stream_lock:
            if self.stream_bool:
                logger.error("Cannot launch new transcriber. Transcription active already.")
                self.err_callback("Cannot launch new transcriber. Transcription active already.")
                return

            print("Listening...")
            self.stream_bool = True
            self.stream.start_stream()

            trans_thread = Thread(target=self.__transcribe, name="TranscriptionThread")
            trans_thread.start()


    def stop_transcription(self) -> None:
        with self.stream_lock:
            if not self.stream_bool:
                logger.error("Transcriber already in stopped state!")
                self.err_callback("Transcriber already in stopped state!")
                return

            self.stream_bool = False
            self.stream.stop_stream()


    def close(self) -> None:
        self.stream.close()
        self.p.terminate()
