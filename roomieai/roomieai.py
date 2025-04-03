from transcriber import Transcriber
from time import sleep

def data_callback_func(data):
    print(data)
    text = data.get("text", "")
    # TODO: Send text data to a chatbot API


def err_callback_func(e):
    print("Error msg: %s", e)


def main():
    transcriber = Transcriber("model")
    transcriber.register_data_callback(data_callback_func)
    transcriber.register_err_callback(err_callback_func)
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
