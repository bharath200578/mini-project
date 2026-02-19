import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer_model import TinyTransformer
from utils.preprocessing import load_data

# Load dataset
X, y, vocab, label_encoder = load_data("data/dataset.csv")

# Initialize model
model = TinyTransformer(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
torch.save(vocab, "vocab.pth")
import pickle

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


print("Training Complete. Model saved.")
