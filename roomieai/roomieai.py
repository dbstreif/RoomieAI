from time import sleep
import threading
import queue
import re
from threading import Event
import random
from transcriber import Transcriber
from TTS.api import TTS
import numpy as np
import sounddevice as sd
from ai_brain import openai_call 
import beepy
from devtext import DEV3

# Get device
DEVICE = "cpu"

# Init TTS
tts = TTS("tts_models/en/ljspeech/vits").to(DEVICE)
sample_rate = tts.synthesizer.output_sample_rate


# Shared queue for waveforms
waveform_queue = queue.Queue()

def generate_waveforms(sentences):
    """
    Producer: Generate waveforms for each sentence and add them to the queue.
    """
    for sentence in sentences:
        waveform = tts.tts(sentence)
        waveform_queue.put(waveform)
    waveform_queue.put(None)  # Sentinel value to signal end of generation

def play_waveforms():
    """
    Consumer: Play waveforms from the queue as they become available.
    """
    while True:
        waveform = waveform_queue.get()
        if waveform is None:  # End of generation
            break
        sd.play(np.array(waveform), samplerate=22050)
        sd.wait()


def process_flag(flag: str, sentences: list[str]) -> None:
    if flag == "yes":
        sentences = sentences.append(random.choice(DEV3))


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
    beepy.beep(sound=1)
    transcriber.pause_transcription()
    text = strip_rumi(data)
    response = openai_call(text)
    if response.get("status_code", "") == 200:
        text = response.get("response", "")
        sentences = re.findall(r'[^.,]+[.,]?', text)
        sentences = [s.strip() for s in sentences]
        process_flag(response.get("flag", ""), sentences)

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=generate_waveforms, args=(sentences,))
        consumer_thread = threading.Thread(target=play_waveforms)

        producer_thread.start()
        consumer_thread.start()

        # Wait for threads to finish
        producer_thread.join()
        consumer_thread.join()
    else:
        waveform = tts.tts("Sorry, something went wrong.")
        waveform = np.array(waveform)
        beepy.beep(sound='error')
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
    beepy.beep(sound=1)
    transcriber.start_transcription()
    try:
        while True:
            sleep(20 * 60)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully.")
        transcriber.stop_transcription()


if __name__ == "__main__":
    main()
