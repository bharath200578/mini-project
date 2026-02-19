import streamlit as st
import torch
import pickle

from models.transformer_model import TinyTransformer
from utils.preprocessing import encode

MAX_LEN = 20

# Load model and files
vocab = torch.load("vocab.pth")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = TinyTransformer(len(vocab))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# UI
st.title("🧠 Tiny Transformer Sentiment Analyzer")
st.write("Enter a sentence to predict sentiment")

user_input = st.text_input("Type your sentence here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        encoded = encode(user_input, vocab, MAX_LEN)
        x = torch.tensor([encoded])

        with torch.no_grad():
            output = model(x).float()
            probs = torch.softmax(output, dim=1).squeeze().tolist()
            pred = int(torch.argmax(output, dim=1).item())

        label = label_encoder.inverse_transform([pred])[0]

        st.write(f"**Prediction:** {label}")
     
        if label == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

# Quick sample test to reproduce reported issue
if st.button("Try sample: 'i gave this is bad as sentence'"):
    sample = "i gave this is bad as sentence"
    encoded = encode(sample, vocab, MAX_LEN)
    x = torch.tensor([encoded])
    with torch.no_grad():
        output = model(x).float()
        probs = torch.softmax(output, dim=1).squeeze().tolist()
        pred = int(torch.argmax(output, dim=1).item())
    label = label_encoder.inverse_transform([pred])[0]
    st.write(f"Sample prediction: **{label}** — probs: {probs}")
