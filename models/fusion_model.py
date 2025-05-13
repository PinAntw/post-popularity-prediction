# models/fusion_model.py
import torch
import torch.nn as nn

"""
FusionModel：將所有模態特徵 concat 後，透過 12 層 Cascaded Fully Connected Network 預測 popularity 分數。
比照論文設計：層數固定為 12 層，每層單元數逐層減半，最後輸出為 1。
"""

class FusionModel(nn.Module):
    def __init__(self, input_dims):
        super(FusionModel, self).__init__()
        input_dim = sum(input_dims)  # 融合後輸入維度

        # 定義每層神經元數量（近似論文設計的逐層縮減）
        layer_dims = [input_dim]
        for _ in range(11):  # 生成 11 層（加上一層輸出共 12）
            next_dim = max(1, layer_dims[-1] // 2)
            layer_dims.append(next_dim)

        # 建立 12 層 FC 結構
        layers = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))

        self.fc = nn.Sequential(*layers)

    def forward(self, features):
        fused = torch.cat(features, dim=1)  # concat 所有模態特徵: (B, total_dim)
        out = self.fc(fused)               # 經過 12 層降維網路: (B, 1)
        return out.squeeze(-1)             # 最終回傳: (B,)