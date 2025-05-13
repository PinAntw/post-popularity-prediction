# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HashtagGuidedAttention(nn.Module):
    """
    Hashtag-Guided Attention 模組：使用 hashtag 向量作為 query，
    對 text 或 image encoder 輸出 (sequence) 做 weighted attention。
    """
    def __init__(self, query_dim, input_dim, hidden_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)     # 將 hashtag 向量映射為注意力 query
        self.key_proj = nn.Linear(input_dim, hidden_dim)        # 將 encoder output 映射為 key
        self.value_proj = nn.Linear(input_dim, hidden_dim)      # 將 encoder output 映射為 value
        self.output_proj = nn.Linear(hidden_dim, input_dim)     # 回到原來維度（可選）

    def forward(self, query, context):
        """
        query:   (B, D_q) - hashtag 向量
        context: (B, L, D_h) - text 或 image 特徵序列
        return:  (B, D_h) - weighted representation
        """
        Q = self.query_proj(query).unsqueeze(1)        # (B, 1, H)
        K = self.key_proj(context)                     # (B, L, H)
        V = self.value_proj(context)                   # (B, L, H)

        # attention score = dot(Q, K)
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # (B, 1, L)
        attn_weights = F.softmax(attn_weights, dim=-1)                       # (B, 1, L)

        attended = torch.bmm(attn_weights, V)         # (B, 1, H)
        attended = attended.squeeze(1)                # (B, H)
        output = self.output_proj(attended)           # (B, D_h)
        return output