import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetVisionEncoder(nn.Module):
    """use resnet as a vision encoder"""
    def __init__(self, model_name: str='resnet50', embed_dim: int=512, pretrained: bool=True):
        super().__init__()

        # load a pre-trained ResNet model
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f'Cannot support {model_name}')

        # remove the final fcl
        self.backbone.fc = nn.Identity()

        # need flatten in the forward manually
        # self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        feature_dim = 2048

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x):
        """
        Args:
            :param x: image.shape = [B, 3, H, W]
            :return: torch.Tensor, shape = [B, embed_dim]
        """

        # extract features + flatten
        features = self.backbone(x)
        # features = features.flatten(start_dim=1)  # flatten the features if not use nn.Identity(

        # projection + normalization
        features = self.ln(self.projection(features))
        return features

class ViTEncoder(nn.Module):
    """use ViT as a vision encoder"""
    def __init__(
            self,
            image_size: int=224,
            patch_size: int=16,
            embed_dim: int=768,
            depth: int=12,
            num_heads: int=12,
            mlp_ratio: float=4.0,
            dropout: float=0.1,
    ):
        """
        Args:
            image_size: the size of the input image
            patch_size: the size of each patch
            embed_dim: the dimension of the embedding
            depth: number of transformer blocks
            num_heads: number of attention heads
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            dropout: dropout rate
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size
        )

        # position embedding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim)) # self.num_patches + 1 for the [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # transformer blocks
        self.transformer_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=depth
        )

        # layer norm
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: image.shape = [B, 3, H, W]
        :return: image fetures -> shape = [B, embed_dim]
        """

        batch_size = x.shape[0]

        # patch embedding
        x = self.patch_embedding(x) # shape = [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)

        # add cls token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1) # shape = [B, num_patches + 1, embed_dim]

        # add position embedding
        x += self.pos_embed

        # go through transformer blocks
        x = self.transformer_blocks(x)

        # take the cls token as the output
        output = x[:, 0] # shape = [B, embed_dim]

        # layer norm
        output = self.ln(output)

        return output


if __name__ == '__main__':
    # Test ViTEncoder is working
    vit_encoder = ViTEncoder()
    vit_encoder.eval()
    example_input = torch.randn(10, 3, 224, 224)

    with torch.no_grad():
        output = vit_encoder(example_input)
    print("shape: ", output.shape)  # [10, 768]
