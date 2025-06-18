import os
import torch
from dataclasses import dataclass
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

@dataclass
class ModelConfig:
    """Configuration for the CLIP model."""
    # basic parameters
    embed_dim: int = 512
    vision_width: int = 768
    text_width: int = 512

    # parameters for vision encoder
    # vision_model: str = "resnet50"
    vision_encoder: str = "resnet50"
    image_size: int = 224
    vision_layers: int = 12
    vision_heads: int = 12

    # parameters for text encoder
    text_model: str = "bert-base"
    # vocab_size: int = 30522  # bert-base vocab size
    vocab_size = tokenizer.vocab_size
    max_text_length: int = 256
    text_layers: int = 6
    text_heads: int = 8

    # training parameters
    temperature: float = 0.07
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    # data_root: str = r'C:\D\Programming\Datasets\rocov2'
    data_root: str = '/scratch/sc232jl/rocov2'
    train_split: str = 'train'
    val_split: str = 'valid'
    test_split: str = 'test'

    # training parameters
    batch_size: int = 128
    num_workers: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_epochs: int = 30

    # optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # lr scheduler parameters
    min_lr: float = 1e-6

    # saving and logging records
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_freq: int = 1
    eval_freq: int = 1
    log_freq: int = 1

    mixed_precision: bool=True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42