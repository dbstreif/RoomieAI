from transcriber import Transcriber
from time import sleep

def data_callback_func(data):
    text = data.get("text", "")
    # TODO: Send text data to a chatbot API


def command_callback_func(data):
    text = data.get("text", "")
    print(text)


def err_callback_func(e):
    print("Error msg: %s", e)


def main():
    transcriber = Transcriber("small_model", ["roomie", "roomy", "rumi"], ["room", "rum", "roam"])
    transcriber.register_data_callback(data_callback_func)
    transcriber.register_err_callback(err_callback_func)
    transcriber.register_command_callback(command_callback_func)
    transcriber.start_transcription()
    try:
        while True:
            sleep(20 * 60)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully.")
        transcriber.stop_transcription()
        transcriber.close()

if __name__ == "__main__":
    main()
