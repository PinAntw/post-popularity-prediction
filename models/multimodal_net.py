# models/multimodal_net.py
import torch.nn as nn
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import FusionModel
from attention import HashtagGuidedAttention

"""
MultimodalNet：整合多模態子模型（文字編碼器、圖像編碼器、融合層），
加入 topic+graph、social 特徵的壓縮層，並透過 Hashtag-Guided Attention 聚焦圖文關鍵資訊。
"""

class MultimodalNet(nn.Module):
    def __init__(self, feature_dims):
        """
        feature_dims: [text_dim, topic+graph_dim, image_dim, social_dim]
        topic_graph 降到 128 維，social 降到 6 維
        """
        super(MultimodalNet, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        self.topic_graph_proj = nn.Linear(feature_dims[1], 128)

        # Attention 模組（以 topic+graph 為 query）
        self.attn_text = HashtagGuidedAttention(query_dim=128, input_dim=768)
        self.attn_image = HashtagGuidedAttention(query_dim=128, input_dim=768)

        # 融合層輸入維度：768 (title) + 768 (image) + 128 (topic_graph) + 6 (social)
        self.fusion = FusionModel([768, 768, 128, 6])

    def forward(self, title_inputs, topic_graph_inputs, image_inputs, social_inputs):
        text_seq = self.text_encoder(*title_inputs)                # (B, L, 768)
        image_seq = self.image_encoder(image_inputs)              # (B, 49, 768)
        topic_graph_feat = self.topic_graph_proj(topic_graph_inputs)  # (B, 128)
        social_feat = self.social_proj(social_inputs)             # (B, 6)

        # Attention over sequence features
        text_feat = self.attn_text(topic_graph_feat, text_seq)    # (B, 768)
        image_feat = self.attn_image(topic_graph_feat, image_seq) # (B, 768)

        fused_output = self.fusion([text_feat, image_feat, topic_graph_feat, social_feat])
        return fused_output
