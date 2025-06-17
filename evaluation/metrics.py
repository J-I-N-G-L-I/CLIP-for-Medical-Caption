import torch
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

class RetrievalMetrics:
    """ metrics for retrieval tasks"""
    def __init__(self, k_value=[1, 5, 10]):
        self.k_value = k_value

    def compute_recall_k(self, similarities, k):
        """

        :param similarities: similarity matrix [N, N]
        :param k:
        :return: float recall@k
        """
        N = similarities.shape[0]
        recall_k = 0

        for i in range(N):
            sorted_indices = torch.argsort(similarities[i], descending=True)
            if i in sorted_indices[:k]:
                recall_k += 1

        return recall_k / N

    def compute_metrics(self, image_features, text_features):
        """
        Compute all metrics
        :param image_features: [N, embed_dim]
        :param text_features:  [N, embed_dim]
        :return:
        """
        # normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # compute similarity matrix
        image2text_sim = torch.matmul(image_features, text_features.T)
        text2image_sim = torch.matmul(text_features, image_features.T)

        metrics = {}

        # compute each recall@k
        for k in self.k_value:
            image2text_recall = self.compute_recall_k(text2image_sim, k)
            metrics[f'image2text_recall@{k}'] = image2text_recall

            text2image_recall = self.compute_recall_k(image2text_sim, k)
            metrics[f'text2image_recall@{k}'] = text2image_recall

            average_recall = (metrics[f'image2text_recall@{k}'] + metrics[f'text2image_recall@{k}']) / 2
            metrics[f'average_recall@{k}'] = average_recall

        image2text_mrr = self.compute_mrr(image2text_sim)
        text2image_mrr = self.compute_mrr(text2image_sim)
        metrics['image2text_mrr'] = image2text_mrr
        metrics['text2image_mrr'] = text2image_mrr
        metrics['average_mrr'] = (image2text_mrr + text2image_mrr) / 2

        return metrics

    # for computing mean reciprocal rank (MRR)
    def compute_mrr(self, similarities):
        N = similarities.shape[0]
        reciprocal_ranks = []
        for i in range(N):
            sorted_indices = torch.argsort(similarities[i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            reciprocal_ranks.append(1.0 / rank)

        return np.mean(reciprocal_ranks)


if __name__ == '__main__':

    # mock data for testing
    image_features = torch.randn(100, 512)
    text_features = torch.randn(100, 512)


    metrics_calculator = RetrievalMetrics() # k = [1, 5, 10]
    metrics = metrics_calculator.compute_metrics(image_features, text_features)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")