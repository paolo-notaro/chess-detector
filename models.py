import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNNEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # [B, 32, 56, 56]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # [B, 64, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # [B, 64, 28, 28]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # [B, 128, 28, 28]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                           # [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.encoder(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [B, 128]
        x = self.fc(x)             # [B, output_dim]
        return x

class ChessMovePredictor(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.encoder = SimpleCNNEncoder(output_dim=embedding_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Output heads
        self.from_head = nn.Linear(256, 64)
        self.to_head = nn.Linear(256, 64)
        self.promotion_head = nn.Linear(256, 5)

    def forward(self, before, after):
        emb_before = self.encoder(before)  # [B, D]
        emb_after = self.encoder(after)    # [B, D]
        combined = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D]

        x = self.mlp(combined)  # [B, 256]

        from_logits = self.from_head(x)       # [B, 64]
        to_logits = self.to_head(x)           # [B, 64]
        promotion_logits = self.promotion_head(x)  # [B, 5]

        return from_logits, to_logits, promotion_logits
