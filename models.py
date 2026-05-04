import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Shallow Multi-Layer Perceptron for MNIST.
    Input image: 1 x 28 x 28
    Output: 10 digit classes
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    """
    Small Convolutional Neural Network for MNIST.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(),
            nn.MaxPool2d(2)                               # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class PatchEmbedding(nn.Module):
    """
    Splits image into patches and projects each patch into embedding dimension.
    MNIST image size: 28x28
    Patch size: 7x7
    Number of patches: 16
    """
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)          # [batch, embed_dim, 4, 4]
        x = x.flatten(2)                # [batch, embed_dim, 16]
        x = x.transpose(1, 2)           # [batch, 16, embed_dim]
        return x


class TransformerEncoderMNIST(nn.Module):
    """
    Small Transformer Encoder model for MNIST.
    This is like a tiny Vision Transformer.
    """
    def __init__(
        self,
        img_size=28,
        patch_size=7,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=10
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.position_embedding
        x = self.transformer(x)

        cls_output = x[:, 0]
        return self.classifier(cls_output)


def get_model(model_name):
    model_name = model_name.lower()

    if model_name == "mlp":
        return MLP()
    elif model_name == "cnn":
        return CNN()
    elif model_name in ["transformer", "vit", "encoder"]:
        return TransformerEncoderMNIST()
    else:
        raise ValueError("Invalid model name. Choose from: mlp, cnn, transformer.")
