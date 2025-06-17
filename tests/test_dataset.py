# test for dataset.py
import os
import sys

import torch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.dataset.dataset import ROCOv2Dataset

from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def visualize_sample(sample):
    image = sample['image']
    caption = sample['caption']
    image_id = sample['image_id']

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    elif hasattr(image, 'permute'):
        image = transforms.ToPILImage()(image)

    plt.imshow(image)
    plt.title(f"{image_id}\n{caption}", fontsize=12)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # data_root = r'C:\D\Programming\Datasets\rocov2'
    data_root = '/scratch/sc232jl/rocov2'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ROCOv2Dataset(
        data_root=data_root,
        split='train',
        image_transform=transform,
        # image_transform=None,
        text_transform=str.strip
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print('Number of samples:', len(dataset))
    for batch in dataloader:
        print(f"image shape: {batch['image'].shape}")
        print(f"caption: {batch['caption']}")
        visualize_sample({
            'image': batch['image'][0],
            'caption': batch['caption'][0],
            'image_id': batch['image_id'][0]
        })
        break