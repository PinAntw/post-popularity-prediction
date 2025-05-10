# models/image_encoder.py
import torch
import torch.nn as nn
import torchvision.models as models
"""
ImageEncoder 模組：使用 VGG19 提取圖像特徵，將圖像轉換為多個區域向量（patch features），
並透過線性層投影到統一維度以供融合使用。
"""

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(ImageEncoder, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), 49, 512)  # Flatten spatial dimensions
        x = self.fc(x)  # Project to output_dim
        return x  # (B, 49, output_dim)
