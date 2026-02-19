from collections import Counter
import re
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def build_vocab(texts):
    words = []
    for text in texts:
        text = clean_text(text)
        words.extend(text.split())

    vocab = {word: i+1 for i, (word, _) in enumerate(Counter(words).items())}
    vocab["<PAD>"] = 0
    return vocab


def encode(text, vocab, max_len=20):
    text = clean_text(text)
    tokens = [vocab.get(word, 0) for word in text.split()]
    tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]


def load_data(csv_path, max_len=20):
    df = pd.read_csv(csv_path)
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(str).tolist()

    vocab = build_vocab(texts)

    X = [encode(t, vocab, max_len) for t in texts]
    X = torch.tensor(X, dtype=torch.long)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    y = torch.tensor(y, dtype=torch.long)

    return X, y, vocab, le

