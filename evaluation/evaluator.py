import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import RetrievalMetrics
from config.config import TrainingConfig
import torch.nn.functional as F

class CLIPEvaluator:
    def __init__(self, model, device=TrainingConfig.device):
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.model.eval()
        self.metrics = RetrievalMetrics()

    def evaluate(self, dataset, batch_size=128, num_workers=4):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="feature extraction"):
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)

                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)

                # for efficient gpu memory usage
                image_features.append(image_features.cpu())
                text_features.append(text_features.cpu())

            image_features = torch.cat(image_features)
            text_features = torch.cat(text_features)

            results = self.metrics.compute_metrics(image_features, text_features)
            return results

    def compute_similarity(self, images, texts):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)

            # normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            similarity = torch.matmul(image_features, text_features.T)
            return similarity