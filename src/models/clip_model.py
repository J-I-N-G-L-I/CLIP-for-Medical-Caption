import torch
import torch.nn as nn
from .vision_encoder import ResNetVisionEncoder, ViTEncoder
from .text_encoder import TextEncoder
from .losses import ContrastiveLoss
import torch.nn.functional as F

class CLIPModel(nn.Module):
    """the main CLIP model"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # init vision encoder
        if config.vision_encoder == 'resnet50':
            self.vision_encoder = ResNetVisionEncoder(
                model_name='resnet50',
                embed_dim=config.embed_dim,
                pretrained=True
            )
        elif config.vision_encoder == 'vit':
            self.vision_encoder = ViTEncoder(
                image_size=config.image_size,
                patch_size=config.patch_size,
                embed_dim=config.embed_dim,
                depth=config.vit_depth,
                num_heads=config.vit_num_heads,
                mlp_ratio=config.vit_mlp_ratio,
                dropout=config.dropout
            )
        else:
            raise ValueError(f'Unsupported vision encoder: {config.vision_encoder}')

        # init text encoder
        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            max_length=config.max_text_length,
            num_heads=config.text_heads,
            dropout=config.dropout,
            num_layers=config.text_layers
        )

        # init contrastive loss
        self.loss_fn = ContrastiveLoss(temperature=config.temperature)

        # init learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) *
                                        torch.log(torch.tensor(1 / config.temperature)))

    def encode_image(self, images):
        """
        Encode images to feature vectors.
        :param images: torch.Tensor, shape = [B, 3, H, W]
        :return: torch.Tensor, shape = [B, embed_dim]
        """
        return self.vision_encoder(images)

    def encode_text(self, input_ids, attention_mask=None):
        """
        Encode tokenized text.
        :param input_ids: torch.Tensor, shape = [B, seq_len]
        :param attention_mask: torch.Tensor, shape = [B, seq_len]
        :return: torch.Tensor, shape = [B, embed_dim]
        """
        return self.text_encoder(input_ids, attention_mask)

    # def forward(self, images, texts, attention_mask=None, return_loss=True):
    #     """
    #     Forward pass of the CLIP model.
    #     :param images: torch.Tensor, shape = [B, 3, H, W]
    #     :param texts: torch.Tensor, shape = [B, sequence_len]
    #     :param attention_mask: optional attention mask for text encoder
    #     :param return_loss: whether to return loss
    #     :return: if return_loss is True, return loss; otherwise return image and text features
    #     """
    #     image_features = self.encode_image(images)
    #     text_features = self.encode_text(texts)

    def forward(self,
                images,
                input_ids,
                attention_mask=None,
                return_loss=True):
        """
        :param images:       torch.Tensor, shape = [B,3,H,W]
        :param input_ids:    torch.Tensor, shape = [B, sequence_len]
        :param attention_mask: optional torch.Tensor, shape = [B, sequence_len]
        :param return_loss:  bool
        """

        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # calculate similarity
        image_features_norm = torch.nn.functional.normalize(image_features, dim=-1)
        text_features_norm = torch.nn.functional.normalize(text_features, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features_norm @ text_features_norm.T
        logits_per_text = logits_per_image.T

        output = {
            'image_features': image_features,
            'text_features': text_features,
            'logit_scale': logit_scale,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
        if return_loss:
            # the following line is commented out because the model not learning
            # loss, loss_dict = self.loss_fn(image_features, text_features)
            # output['loss'] = loss
            # # add other loss information from loss_dict
            # output.update(loss_dict)

            # labels = torch.arange(images.size(0), device=images.device)
            # loss_i2t = F.cross_entropy(logits_per_image, labels)
            # loss_t2i = F.cross_entropy(logits_per_text, labels)
            # loss = (loss_i2t + loss_t2i) / 2
            #
            # output["loss"] = loss
            # output["image2text_loss"] = loss_i2t.item()
            # output["text2image_loss"] = loss_t2i.item()

            # 让 logit_scale 真正参与梯度：用 logits 计算交叉熵
            labels = torch.arange(images.size(0), device=images.device)
            loss_i2t = F.cross_entropy(logits_per_image, labels)
            loss_t2i = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i2t + loss_t2i) / 2

            output.update({
                'loss': loss,
                'image2text_loss': loss_i2t,
                'text2image_loss': loss_t2i,
            })
            # if return_loss:
            #     loss, loss_dict = self.loss_fn(image_features, text_features)
            #     output['loss'] = loss
            #     output['image2text_loss'] = loss_dict['image2text_loss']
            #     output['text2image_loss'] = loss_dict['text2image_loss']

        return output