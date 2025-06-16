import os
import torch
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configure hyperparameters for the CLIP model."""
    # basic parameters
    embed_dim: int = 512
    vision_width: int = 768
    text_width: int = 512

    # parameters for vision encoder
    vision_model: str = "resnet50"
    image_size: int = 224
    vision_layers: int = 12
    vision_heads: int = 12

    # parameters for text encoder
    text_model: str = "bert-base"
    vocab_size: int = 30522  # bert-base vocab size
    max_text_length: int = 256
    text_layers: int = 6
    text_heads: int = 8

    # training parameters
    temperature: float = 0.07
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    # data configs
    data_root: str = r'C:\D\Programming\Datasets\rocov2'
    train_split: str = 'train'
    val_split: str = 'valid'
    test_split: str = 'test'


    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'