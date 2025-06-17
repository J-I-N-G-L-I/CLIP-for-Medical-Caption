import argparse
import torch
import random
import numpy as np

from config.config import ModelConfig, TrainingConfig
from src.dataset.dataset import ROCOv2Dataset
from src.dataset.transforms import ImageTransform, TextTransform
from src.models.clip_model import CLIPModel
from training.trainer import CLIPTrainer

def set_seed(seed):
    """ Set random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Training CLIP for Medical Images Captions')
    parser.add_argument('--data_root', type=str, default='/scratch/sc232jl/rocov2')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    # configurations
    model_config = ModelConfig()
    train_config = TrainingConfig()

    # update configurations
    train_config.data_root = args.data_root
    train_config.save_dir = args.save_dir
    train_config.batch_size = args.batch_size
    train_config.max_epochs = args.epochs
    train_config.learning_rate = args.lr

    # set random seed
    set_seed(train_config.seed)

    # data transformations
    train_image_transform = ImageTransform(
        size=model_config.image_size,
        is_training=True
    )
    val_image_transform = ImageTransform(
        size=model_config.image_size,
        is_training=False
    )

    text_transform = TextTransform(max_length=model_config.max_text_length)

    # create datasets
    train_dataset = ROCOv2Dataset(
        data_root=train_config.data_root,
        split=train_config.train_split,
        image_transform=train_image_transform,
        text_transform=text_transform
    )
    val_dataset = ROCOv2Dataset(
        data_root=train_config.data_root,
        split=train_config.val_split,
        image_transform=val_image_transform,
        text_transform=text_transform
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # create model
    model = CLIPModel(model_config)
    print(f"Parameters in model: {sum(p.numel() for p in model.parameters()):,}")

    # create trainer
    trainer = CLIPTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=train_config
    )

    # resume training if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # start training
    trainer.train()

if __name__ == '__main__':
    main()