import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter

class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        # Load pretrained resnet18 and adapt to grayscale input
        base_model = resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = IntermediateLayerGetter(base_model, return_layers={'layer2': 'feat'})
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(128, embed_dim)  # layer2 output channels = 128

    def forward(self, x):
        feats = self.backbone(x)['feat']  # [B * 64, 128, H, W]
        pooled = self.pool(feats).squeeze(-1).squeeze(-1)  # [B * 64, 128]
        return self.project(pooled)  # [B * 64, embed_dim]


class MoveScorer(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.from_proj = nn.Linear(embed_dim, 64)
        self.to_proj = nn.Linear(embed_dim, 64)

    def forward(self, embeddings):
        # embeddings: [B, 64, D]
        from_vecs = F.normalize(self.from_proj(embeddings), dim=-1)  # [B, 64, 64]
        to_vecs = F.normalize(self.to_proj(embeddings), dim=-1)      # [B, 64, 64]

        # Compute pairwise move scores using batched matrix multiplication
        scores = torch.matmul(from_vecs, to_vecs.transpose(1, 2))  # [B, 64, 64]
        return scores

class MoveScorer(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, embeddings):
        # embeddings: [B, 64, D]
        B, N, D = embeddings.shape

        # Expand from and to square embeddings for all pair combinations
        from_expand = embeddings.unsqueeze(2).expand(B, N, N, D)  # [B, 64, 64, D]
        to_expand = embeddings.unsqueeze(1).expand(B, N, N, D)    # [B, 64, 64, D]

        # Concatenate from and to embeddings
        pairwise = torch.cat([from_expand, to_expand], dim=-1)  # [B, 64, 64, 2D]

        # Flatten to apply MLP
        pairwise_flat = pairwise.view(B * N * N, 2 * D)  # [B*64*64, 2D]
        scores_flat = self.mlp(pairwise_flat).view(B, N, N)  # [B, 64, 64]

        scores = scores_flat.view(B, N, N)

        return scores

class ChessMoveModel(nn.Module):
    def __init__(self, patch_size=32, in_channels=1, embed_dim=128):
        super().__init__()
        self.encoder = PatchEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.scorer = MoveScorer(embed_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(64, embed_dim))

    def forward(self, patches):
        # patches: [B, 64, C, H, W]
        B, N, C, H, W = patches.size()
        patches = patches.view(B * N, C, H, W)  # [B*64, C, H, W]
        embeddings = self.encoder(patches)      # [B*64, D]
        embeddings = embeddings.view(B, N, -1)  # [B, 64, D]

        embeddings = embeddings + self.positional_encoding.unsqueeze(0)  # [B, 64, D]
        scores = self.scorer(embeddings)        # [B, 64, 64]

        return scores
