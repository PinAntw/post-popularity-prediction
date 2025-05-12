# # models/multimodal_net.py
# import torch.nn as nn
# from models.text_encoder import TextEncoder
# from models.image_encoder import ImageEncoder
# from models.fusion_model import FusionModel

# """
# MultimodalNet：整合多模態子模型（文字編碼器、圖像編碼器、融合層），
# 將不同來源的資訊融合並預測貼文的熱門程度（regression）。
# """

# class MultimodalNet(nn.Module):
#     def __init__(self, feature_dims):
#         super(MultimodalNet, self).__init__()
#         self.text_encoder = TextEncoder()
#         self.image_encoder = ImageEncoder()
#         self.fusion = FusionModel(feature_dims)
#         self.social_proj = nn.Linear(social_dim, 128)


#     def forward(self, text_inputs, topic_inputs, image_inputs, extra_features):
#         text_feat = self.text_encoder(*text_inputs)[:, 0, :]          # (B, 768)
#         topic_feat = topic_inputs                                     # (B, topic_dim)
#         image_feat = self.image_encoder(image_inputs).mean(dim=1)    # (B, 768)
#         fused_output = self.fusion([text_feat, topic_feat, image_feat, extra_features])
#         social_feat = self.social_proj(extra_features)
#         fused_output = self.fusion([text_feat, topic_feat, image_feat, social_feat])

#         return fused_output
# models/multimodal_net.py
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import FusionModel

"""
MultimodalNet：整合多模態子模型（文字編碼器、圖像編碼器、融合層），
加入 topic 和 social 特徵的壓縮層，減少維度與過擬合風險。
"""

class MultimodalNet(nn.Module):
    def __init__(self, feature_dims):
        """
        feature_dims: [text_dim, topic_dim, image_dim, social_dim]
        壓縮後將 topic 降到 64 維，social 降到 128 維
        """
        super(MultimodalNet, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        # 壓縮層
        self.topic_proj = nn.Linear(feature_dims[1], 128)
        self.social_proj = nn.Linear(feature_dims[3], 64)

        # 壓縮後維度：768 (text) + 64 (topic) + 768 (image) + 128 (social)
        self.fusion = FusionModel([768, 128, 768, 64])

    def forward(self, text_inputs, topic_inputs, image_inputs, extra_features):
        text_feat = self.text_encoder(*text_inputs)[:, 0, :]             # (B, 768)
        topic_feat = self.topic_proj(topic_inputs)                      # (B, 64)
        image_feat = self.image_encoder(image_inputs).mean(dim=1)       # (B, 768)
        social_feat = self.social_proj(extra_features)                  # (B, 128)

        fused_output = self.fusion([text_feat, topic_feat, image_feat, social_feat])
        return fused_output
