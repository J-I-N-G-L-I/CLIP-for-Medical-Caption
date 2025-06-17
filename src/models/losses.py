import torch
import torch.nn as nn
import torch.nn.functional as F

# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#
#     def forward(self, image_feature, text_feature):
#         """
#         :param image_feature: [B, embed_dim]
#         :param text_feature: [B, embed_dim]
#         :return: total_loss + dict
#         """
#         batch_size = image_feature.shape[0]
#
#         # L2 norm
#         image_feature = F.normalize(image_feature, dim=-1)
#         text_feature = F.normalize(text_feature, dim=-1)
#
#         similarity_matrix = torch.matmul(image_feature, text_feature.T) / self.temperature
#
#         # create labels (The diagonal represents the positive samples.)
#         labels = torch.arange(batch_size, device=image_feature.device)
#
#         # calculate final loss
#         image2text_loss = F.cross_entropy(similarity_matrix, labels)
#         text2image_loss = F.cross_entropy(similarity_matrix.T, labels)
#         total_loss = (image2text_loss + text2image_loss) / 2
#
#         return total_loss, {
#             'image2text_loss': image2text_loss.item(),
#             'text2image_loss': text2image_loss.item(),
#
#             # only for tests in the trainer.py
#             # 'image2text_loss': image2text_loss,
#             # 'text2image_loss': text2image_loss,
#             # similarity_matrix: similarity_matrix.detach(),
#         }

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, query, key):
        """
        :param query: [B, embed_dim]
        :param key: [B, embed_dim]
        :return:
        """
        # Norm
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        # similarity
        logits = torch.matmul(query, key.T) / self.temperature

        labels = torch.arange(logits.shape[0], device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss