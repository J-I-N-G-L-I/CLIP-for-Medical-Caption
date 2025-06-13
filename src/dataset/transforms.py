import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageTransform:
    """Image transformation class for ROCOv2 dataset."""
    def __init__(self, size=224, is_training=True):
        self.size = size
        self.is_training = is_training

        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # for test only resizing and normalizing
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        return self.transform(image)

class TextTransform:
    """Text transformation class for ROCOv2 dataset."""
    def __init__(self, max_length=128, tokenizer=None):
        self.max_length = max_length
        self.tokenizer = tokenizer

        if tokenizer is None:
            self._build_vocab()

    def _build_vocab(self):
        # Build a simple character-level vocabulary
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-'
        self.vocab = {char: i+1 for i, char in enumerate(chars)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = len(self.vocab)

    def __call__(self, text):
        if self.tokenizer:
            # Use the provided tokenizer
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens['input_ids'].squeeze(0)
        else:
            tokens = []
            for char in text[:self.max_length]:
                tokens.append(self.vocab.get(char, self.vocab['<UNK>']))

            while len(tokens) < self.max_length:
                tokens.append(self.vocab['<PAD>'])

            return torch.tensor(tokens, dtype=torch.long)

if __name__ == '__main__':
    # Test the transforms

    # apply image transform
    example_image = Image.new('RGB', (256, 256), color='blue')
    image_transform = ImageTransform(size=256, is_training=True)

    image_tensor = image_transform(example_image)
    print("image_tensor shape: ", image_tensor.shape) # torch.Size([3, 256, 256])
    print("image_tensor dtype: ", image_tensor.dtype) # torch.float32
    print(image_tensor)

    # apply text transform
    example_image = "Chest X-ray shows no acute cardiopulmonary disease."
    text_transform = TextTransform(tokenizer=None, max_length=128)

    token_ids = text_transform(example_image)
    print("token_ids shape: ", token_ids.shape) # torch.Size([128])
    print("token_ids dtype: ", token_ids.dtype) # torch.int64
    print(token_ids)
