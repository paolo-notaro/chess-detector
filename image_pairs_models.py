import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Type


# Count parameters for each encoder alone
def count_params(model, trainable_only=True):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


class HybridLargeChessboardEncoder(nn.Module):
    def __init__(
        self, output_dim=256, num_transformer_layers=1, num_heads=8, dropout=0.1
    ):
        super(HybridLargeChessboardEncoder, self).__init__()

        # Load pre-trained ResNet model
        resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1-channel input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Average the weights of the original conv1 layer across the RGB channels
        self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        # Extract ResNet layers up to the adaptive pooling layer
        self.resnet_layers = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=resnet.fc.in_features,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Fully connected layer to project to desired output dimension
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # ResNet feature extraction
        x = self.resnet_layers(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, features]

        # Add sequence dimension for Transformer (sequence length = 1)
        x = x.unsqueeze(1)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Remove sequence dimension
        x = x.squeeze(1)

        # Fully connected layer
        x = self.fc(x)

        return x


def make_grayscale_conv(weight):
    """Convert a 3-channel conv weight to 1-channel by averaging."""
    return weight.mean(dim=1, keepdim=True)


class SmallCNNEncoder(nn.Module):
    """For ~6k samples: small model, ~400k total with move network."""

    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 16, kernel_size=5, stride=2, padding=2
            ),  # -> [B, 16, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 16, 56, 56]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> [B, 32, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> [B, 32, 28, 28]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> [B, 64, 28, 28]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> [B, 64, 1, 1]
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MidCNNEncoder(nn.Module):
    """For ~40k samples: ResNet18 variant, ~2-3M total with move network."""

    def __init__(self, output_dim=256):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = make_grayscale_conv(base.conv1.weight.data)

        self.backbone = nn.Sequential(
            self.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,  # stop here, reduce total params
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LargeCNNEncoder(nn.Module):
    """For ~100k samples: ResNet34 backbone, ~4-5M total with move network."""

    def __init__(self, output_dim=256):
        super().__init__()
        base = models.resnet34(pretrained=True)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = make_grayscale_conv(base.conv1.weight.data)

        self.backbone = nn.Sequential(
            self.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNNEncoder(nn.Module):
    def __init__(self, output_dim: int = 256):
        super().__init__()
        # Assuming input is of size [B, 1, 224, 224] (grayscale image)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 56, 56]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 28, 28]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [B, 128, 28, 28]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1]
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.encoder(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten: [B, 128]
        x = self.fc(x)  # [B, output_dim]
        return x


class ChessMovePredictor(nn.Module):
    def __init__(
        self, embedding_dim=256, encoder_class: Type[nn.Module] = SmallCNNEncoder
    ):
        """
        Args:
            embedding_dim (int): Dimension of the embedding.
            encoder_class (Type[nn.Module]): Class of the encoder to use.
        """
        super().__init__()
        assert encoder_class in [
            SmallCNNEncoder,
            MidCNNEncoder,
            LargeCNNEncoder,
            SimpleCNNEncoder,
        ], "Invalid encoder class"
        # self.encoder = SimpleCNNEncoder(output_dim=embedding_dim)
        self.encoder = encoder_class(output_dim=embedding_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Output heads
        self.from_head = nn.Linear(256, 64)
        self.to_head = nn.Linear(256, 64)
        self.promotion_head = nn.Linear(256, 5)

    def forward(self, before, after):
        emb_before = self.encoder(before)  # [B, D]
        emb_after = self.encoder(after)  # [B, D]
        combined = torch.cat([emb_before, emb_after], dim=1)  # [B, 2D]

        x = self.mlp(combined)  # [B, 256]

        from_logits = self.from_head(x)  # [B, 64]
        to_logits = self.to_head(x)  # [B, 64]
        promotion_logits = self.promotion_head(x)  # [B, 5]

        return from_logits, to_logits, promotion_logits

