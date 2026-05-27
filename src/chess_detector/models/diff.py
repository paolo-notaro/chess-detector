import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResnetPatchEncoder(nn.Module):
    """
    A ResNet18-based encoder for processing patches of chessboard images.
    It modifies the first convolutional layer to accept a variable number of input channels
    and replaces the final fully connected layer to output a specified embedding dimension.
    """

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

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet18 model.
        The input is expected to be a batch of patches with shape [B, 64, C, H, W],
        where B is the batch size, 64 is the number of patches, C is the number of channels,
        and H and W are the height and width of each patch.
        The output is a tensor of shape [B, 64, embed_dim].

        Args:
            patches (torch.Tensor): Input tensor of shape [B, 64, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, 64, embed_dim].
        """
        # patches: [B, 64, C, H, W]
        B, N, C, H, W = patches.shape
        # Reshape to process one patch at a time through ResNet18
        patches = patches.view(B * N, C, H, W)  # [B * 64, C, H, W]
        embeddings = self.resnet(patches)  # [B * 64, embed_dim]
        embeddings = embeddings.view(B, N, -1)  # [B, 64, embed_dim]
        return embeddings


class ConvPatchEncoder(nn.Module):
    """
    A convolutional encoder for processing patches of chessboard images.
    It consists of several convolutional layers followed by batch normalization and ReLU
      activations.
    The final output is projected to a specified embedding dimension.
    """

    def __init__(self, in_channels=1, embed_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 4, 4]
        )

        # Final projection to embed_dim
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional encoder.
        The input is expected to be a batch of patches with shape [B, 64, C, H, W],
        where B is the batch size, 64 is the number of patches, C is the number of channels,
        and H and W are the height and width of each patch.
        The output is a tensor of shape [B, 64, embed_dim].

        Args:
            patches (torch.Tensor): Input tensor of shape [B, 64, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, 64, embed_dim].
        """
        B, N, C, H, W = patches.shape
        patches = patches.view(B * N, C, H, W)  # [B*64, 1, 32, 32]
        features = self.cnn(patches)  # [B*64, 128, 4, 4]
        features = features.view(B * N, -1)  # [B*64, 2048]
        embeddings = self.fc(features)  # [B*64, embed_dim]
        embeddings = embeddings.view(B, N, -1)  # [B, 64, embed_dim]
        return embeddings


class MoveScorer(nn.Module):
    """
    A module to compute move scores based on embeddings of chessboard patches.
    It uses two linear projections to transform the embeddings into a lower-dimensional space
    and computes pairwise scores between them.
    The scores are normalized by a learned temperature parameter.
    """

    def __init__(self, embed_dim=128, proj_size=32):
        super().__init__()
        self.from_proj = nn.Linear(embed_dim, proj_size)
        self.to_proj = nn.Linear(embed_dim, proj_size)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute move scores.
        The input is expected to be a tensor of shape [B, 64, embed_dim],
        where B is the batch size and 64 is the number of patches.
        The output is a tensor of shape [B, 64, 64], representing the pairwise move scores.

        Args:
            embeddings (torch.Tensor): Input tensor of shape [B, 64, embed_dim].
        Returns:
            torch.Tensor: Output tensor of shape [B, 64, 64], representing move scores.
        """
        # embeddings: [B, 64, embed_dim]
        from_vecs = self.from_proj(embeddings)  # [B, 64, proj_size]
        to_vecs = self.to_proj(embeddings)  # [B, 64, proj_size]

        # Compute pairwise move scores using batched matrix multiplication and
        # normalize by learned temperature
        scores = torch.matmul(from_vecs, to_vecs.transpose(1, 2)) * self.temperature  # [B, 64, 64]
        return scores


class ChessMoveModel(nn.Module):
    """
    A model for predicting chess moves based on patches of chessboard images.
    It consists of an encoder (either ResNet18 or a custom convolutional encoder)
    followed by a move scorer.
    The model takes a batch of patches as input and outputs pairwise move scores.
    """

    def __init__(
        self,
        patch_size=32,
        in_channels=1,
        embed_dim=128,
        encoder_class=ResnetPatchEncoder,
    ):
        super().__init__()
        self.encoder = encoder_class(in_channels=in_channels, embed_dim=embed_dim)
        self.scorer = MoveScorer(embed_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(64, embed_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        The input is expected to be a batch of patches with shape [B, 64, C, H, W],
        where B is the batch size, 64 is the number of patches, C is the number of channels,
        and H and W are the height and width of each patch.
        The output is a tensor of shape [B, 64, 64], representing pairwise move scores.

        Args:
            patches (torch.Tensor): Input tensor of shape [B, 64, C, H, W].
        Returns:
            torch.Tensor: Output tensor of shape [B, 64, 64], representing move scores.
        """
        # patches: [B, 64, C, H, W]
        embeddings = self.encoder(patches)  # [B, 64, embed_dim]
        embeddings = embeddings + self.positional_encoding.unsqueeze(0)  # [B, 64, embed_dim]
        scores = self.scorer(embeddings)  # [B, 64, 64]
        return scores
