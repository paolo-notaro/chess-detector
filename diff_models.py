import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        # Load a pre-trained ResNet18 model
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer to accept `in_channels` input channels
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace the fully connected layer to output `embed_dim` features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_dim)

    def forward(self, patches):
        # patches: [B, 64, C, H, W]
        B, N, C, H, W = patches.shape
        # Reshape to process one patch at a time through ResNet18
        patches = patches.view(B * N, C, H, W)  # [B * 64, C, H, W]
        embeddings = self.resnet(patches)  # [B * 64, embed_dim]
        embeddings = embeddings.view(B, N, -1)  # [B, 64, embed_dim]
        return embeddings


class MoveScorer(nn.Module):
    def __init__(self, embed_dim=128, proj_size=32):
        super().__init__()
        self.from_proj = nn.Linear(embed_dim, proj_size)
        self.to_proj = nn.Linear(embed_dim, proj_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings):
        # embeddings: [B, 64, embed_dim]
        from_vecs = self.from_proj(embeddings) # [B, 64, proj_size]
        to_vecs = self.to_proj(embeddings)  # [B, 64, proj_size]

        # Compute pairwise move scores using batched matrix multiplication / normalize by learned temperature
        scores = torch.matmul(from_vecs, to_vecs.transpose(1, 2)) * self.temperature  # [B, 64, 64]
        return scores


class ChessMoveModel(nn.Module):
    def __init__(self, patch_size=32, in_channels=1, embed_dim=128):
        super().__init__()
        self.encoder = PatchEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.scorer = MoveScorer(embed_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(64, embed_dim))

    def forward(self, patches):
        # patches: [B, 64, C, H, W]
        embeddings = self.encoder(patches)  # [B, 64, embed_dim]
        embeddings = embeddings + self.positional_encoding.unsqueeze(
            0
        )  # [B, 64, embed_dim]
        scores = self.scorer(embeddings)  # [B, 64, 64]
        return scores
