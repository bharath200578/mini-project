from collections import Counter

def build_vocab(texts):
    words = []
    for text in texts:
        words.extend(text.lower().split())

    vocab = {word: i+1 for i, (word, _) in enumerate(Counter(words).items())}
    vocab["<PAD>"] = 0
    return vocab


def encode(text, vocab, max_len=10):
    tokens = [vocab.get(word, 0) for word in text.lower().split()]
    tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]
