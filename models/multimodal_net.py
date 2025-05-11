# models/multimodal_net.py
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import FusionModel

"""
MultimodalNet：整合多模態子模型（文字編碼器、圖像編碼器、融合層），
將不同來源的資訊融合並預測貼文的熱門程度（regression）。
"""

class MultimodalNet(nn.Module):
    def __init__(self, feature_dims):
        super(MultimodalNet, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.fusion = FusionModel(feature_dims)

    def forward(self, text_inputs, topic_inputs, image_inputs, extra_features):
        text_feat = self.text_encoder(*text_inputs)[:, 0, :]          # (B, 768)
        topic_feat = topic_inputs                                     # (B, topic_dim)
        image_feat = self.image_encoder(image_inputs).mean(dim=1)    # (B, 768)
        fused_output = self.fusion([text_feat, topic_feat, image_feat, extra_features])
        return fused_output
