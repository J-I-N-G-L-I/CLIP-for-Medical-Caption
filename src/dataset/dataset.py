import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ROCOv2Dataset(Dataset):
    """Custom dataset for ROCOv2 dataset"""
    def __init__(self, data_root, split='train', image_transform=None, text_transform=None):
        """
        Args:
            data_root (str): Root directory of the dataset.
            split (str): Dataset split, ['train', 'val', 'test']
            image_transform: Transform to apply to images.
            text_transform: Transform to apply to text.
        """
        self.data_root = data_root
        self.split = split
        self.image_transform = image_transform
        self.text_transform = text_transform

        self.data = self._load_data()

    def _load_data(self):
        csv_path = os.path.join(self.data_root, f"{self.split}_captions.csv")
        image_dir = os.path.join(self.data_root, f"{self.split}_images", f"{self.split}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find csv file: {csv_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Cannot find image directory: {image_dir}")

        df = pd.read_csv(csv_path)
        data =[]

        for _, row in df.iterrows():
            image_id = row['ID']
            # caption = row['Caption']
            text = row['Caption']
            image_path = os.path.join(image_dir, f"{image_id}.jpg")

            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    # 'caption': caption,
                    'text': text,
                    'image_id': image_id
                })
            else:
                raise FileNotFoundError(f"Cannot find image: {image_path}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # load images
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Cannot load image {item['image_path']}")

        if self.image_transform:
            image = self.image_transform(image)

        # caption = item['caption']
        text = item['text']
        # if self.text_transform:
        #     # caption = self.text_transform(caption)
        #     text = self.text_transform(text)


        # use openai tokenizer
        text = self.text_transform(text)

        return {
            'image': image,
            # 'caption': caption,
            # 'text': text,
            "input_ids": text["input_ids"],
            "attention_mask": text["attention_mask"],
            'image_id': item['image_id']
        }