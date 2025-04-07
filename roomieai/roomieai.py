from transcriber import Transcriber
from time import sleep
from TTS.api import TTS
import numpy as np
import sounddevice as sd
from threading import Event
from ai_brain import openai_call 

# Get device
DEVICE = "cpu"

# Init TTS
tts = TTS("tts_models/en/ljspeech/glow-tts").to(DEVICE)
sample_rate = tts.synthesizer.output_sample_rate


def strip_rumi(data):
    # Extract the text
    text = data.get("text", "")

    # Remove the first word
    if text.strip():  # Ensure the text is not empty
        words = text.split(" ")
        if len(words) > 1:  # Check if there are multiple words
            result = " ".join(words[1:])
        else:
            result = ""  # If there's only one word, return an empty string
    else:
        result = ""  # Handle empty input

    return result


def data_callback_func(data):
    text = data.get("text", "")
    # TODO: Send text data to a chatbot API


def command_callback_func(data):
    transcriber.pause_transcription()
    text = strip_rumi(data)
    response = openai_call(text)
    if response.get("status_code", "") == 200:
        waveform = tts.tts(response.get("response", ""))
        waveform = np.array(waveform)
        sd.play(waveform, sample_rate)
        sd.wait()
    else:
        waveform = tts.tts("Sorry, something went wrong.")
        waveform = np.array(waveform)
        sd.play(waveform, sample_rate)
        sd.wait()
    transcriber.resume_transcription()


def err_callback_func(e):
    print("Error msg: %s", e)


# Init Global Transcriber(model_path, wakewords, wakeword_exclusions)
transcriber = Transcriber("small_model", ["roomie", "roomy", "rumi"], ["room", "rum", "roam"])
transcriber.register_data_callback(data_callback_func)
transcriber.register_err_callback(err_callback_func)
transcriber.register_command_callback(command_callback_func)



def main():
    transcriber.start_transcription()
    try:
        while True:
            sleep(20 * 60)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully.")
        transcriber.stop_transcription()


if __name__ == "__main__":
    main()
