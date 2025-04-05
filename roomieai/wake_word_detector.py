from fuzzywuzzy import fuzz

class WakeWordDetector:
    def __init__(self, word, wake_word_list, wake_word_excl_list, threshold=80):
        self.word = word.lower()
        self.wake_word_list = [s.lower() for s in wake_word_list]
        self.wake_word_excl_list = [s.lower() for s in wake_word_excl_list]
        self.threshold = threshold  # Minimum similarity score for a match

    def process_result(self):
        # Use fuzzy matching to detect the wake word
        if self.word in self.wake_word_excl_list:
            return False
        for wake_word in self.wake_word_list:
            similarity = fuzz.ratio(self.word.lower(), wake_word)
            if similarity >= self.threshold:
                return True
        return False
