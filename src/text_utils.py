import re
from collections import Counter


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(texts):
    counts = Counter()

    for text in texts:
        counts.update(tokenize(text))

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word in sorted(counts):
        vocab[word] = len(vocab)

    return vocab


def encode(text, vocab):
    return [
        vocab.get(word, vocab[UNK_TOKEN])
        for word in tokenize(text)
    ]


def pad_batch(sequences):
    max_length = max(len(seq) for seq in sequences)

    return [
        seq + [0] * (max_length - len(seq))
        for seq in sequences
    ]