import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        # Load a pretrained ResNet18 model
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first convolutional layer to accept the desired input channels
        self.resnet_features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            *list(resnet.children())[1:-2]  # Use all layers except the last two
        )
        # Add a projection layer to match the desired embedding dimension
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, embed_dim),
        )

    def forward(self, patches):
        # patches: [B, 64, C, H, W]
        B, S, C, H, W = patches.shape
        patches = patches.view(B * S, C, H, W)  # Flatten the square dimension
        features = self.resnet_features(patches)  # Extract features using ResNet
        embeddings = self.projection(
            features
        )  # Project to the desired embedding dimension
        embeddings = embeddings.view(B, S, -1)  # Reshape back to [B, 64, embed_dim]
        return embeddings


class MoveScorer(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.from_proj = nn.Linear(embed_dim, 64)
        self.to_proj = nn.Linear(embed_dim, 64)

    def forward(self, embeddings):
        # embeddings: [B, 64, D]
        from_vecs = F.normalize(self.from_proj(embeddings), dim=-1)  # [B, 64, 64]
        to_vecs = F.normalize(self.to_proj(embeddings), dim=-1)  # [B, 64, 64]

        # Compute pairwise move scores using batched matrix multiplication
        scores = torch.matmul(from_vecs, to_vecs.transpose(1, 2))  # [B, 64, 64]
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
