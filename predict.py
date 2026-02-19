import torch
from models.transformer_model import TinyTransformer
from utils.preprocessing import encode

MAX_LEN = 20  # MUST match training length

# Load saved files
vocab = torch.load("vocab.pth")
import pickle

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)



model = TinyTransformer(len(vocab))
model.load_state_dict(torch.load("model.pth"))
model.eval()

print("\nTiny Transformer Ready!")
print("Type a sentence to predict sentiment.")
print("Type 'quit' to exit.\n")

while True:
    text = input("Enter sentence: ")

    if text.lower() == "quit":
        print("Exiting...")
        break

    # Encode input text
    encoded = encode(text, vocab, MAX_LEN)
    x = torch.tensor([encoded])

    # Predict with probabilities
    with torch.no_grad():
        output = model(x).float()
        probs = torch.softmax(output, dim=1).squeeze().tolist()
        pred = int(torch.argmax(output, dim=1).item())

    label = label_encoder.inverse_transform([pred])[0]

    print("Prediction:", label)
    print(f"Probabilities: {probs}\n")
