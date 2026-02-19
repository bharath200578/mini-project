import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=64
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.fc(x)
