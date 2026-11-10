
import re


class BasicTokenizer:

    def __init__(self):
        self.vocab = {}
        self.id_to_token = {}

    def train(self, text):
        # Split text into words and punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)

        # Create a vocabulary
        unique_tokens = sorted(set(tokens))

        self.vocab = {
            "<UNK>": 0
        }

        for token in unique_tokens:
            self.vocab[token] = len(self.vocab)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)

        return [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

    def decode(self, token_ids):
        return " ".join(
            self.id_to_token.get(token_id, "<UNK>")
            for token_id in token_ids
        )


if __name__ == "__main__":

    tokenizer = BasicTokenizer()

    training_text = "Python is powerful. Python is simple!"

    tokenizer.train(training_text)

    sentence = "Python is powerful!"

    encoded = tokenizer.encode(sentence)

    decoded = tokenizer.decode(encoded)

    print("Vocabulary:", tokenizer.vocab)
    print("Original Text:", sentence)
    print("Token IDs:", encoded)
    print("Decoded Text:", decoded)
