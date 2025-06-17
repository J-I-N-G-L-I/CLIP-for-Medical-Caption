import argparse
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import ModelConfig, TrainingConfig
from src.dataset.dataset import ROCOv2Dataset
from src.dataset.transforms import ImageTransform, TextTransform
from src.models.clip_model import CLIPModel
from evaluation.evaluator import CLIPEvaluator


def main():
    """Evaluate the model on test dataset."""

    parser = argparse.ArgumentParser(description='Testing CLIP for Medical Images Captions')
    # parser.add_argument('--data_root', type=str, default=r'C:\D\Programming\Datasets\rocov2')
    parser.add_argument('--data_root', type=str, default='/scratch/sc232jl/rocov2')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()

    model_config = ModelConfig()

    image_transform = ImageTransform(
        size=model_config.image_size,
        is_training=False
    )
    text_transform = TextTransform(max_length=model_config.max_text_length)

    test_dataset = ROCOv2Dataset(
        data_root=args.data_root,
        split=args.split,
        image_transform=image_transform,
        text_transform=text_transform
    )

    print(f"Test dataset size: {len(test_dataset)}")

    model = CLIPModel(model_config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loading model checkpoint: {args.checkpoint}")

    evaluator = CLIPEvaluator(model)

    results = evaluator.evaluate(test_dataset, batch_size=args.batch_size)

    print("\n=== Evaluate Result ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()