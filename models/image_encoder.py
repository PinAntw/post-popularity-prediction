# models/image_encoder.py
import torch
import torch.nn as nn
import torchvision.models as models

"""
ImageEncoder 模組：使用 ResNet50 提取圖像特徵，保留 spatial feature map，
再將每個區域特徵映射到統一維度（如 768）以供後續融合使用。
"""

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # 保證固定輸出大小
        self.fc = nn.Linear(2048, output_dim)  # 將每個區域向量映射到 768 維

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)  # (B, 2048, H, W) → 通常為 (B, 2048, 7, 7)
            x = self.pool(x)      # 保險起見固定成 (7, 7)
        x = x.view(x.size(0), 2048, -1).permute(0, 2, 1)  # (B, 49, 2048)
        x = self.fc(x)  # (B, 49, output_dim)
        return x        # return 49 個 patch 向量，每個為 output_dim 維度