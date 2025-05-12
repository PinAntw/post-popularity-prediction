# models/fusion_model.py
import torch
import torch.nn as nn
"""
FusionModel 負責將多模態特徵（如文字、影像、社會特徵）
做 concat 後餵入 MLP 進行回歸預測（如貼文熱門度）。
"""
class FusionModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=512):
        super(FusionModel, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(sum(input_dims), hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, 1)  # Regression output
        # )
        self.fc = nn.Sequential(
            nn.Linear(sum(input_dims), 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    def forward(self, features):
        fused = torch.cat(features, dim=1)
        out = self.fc(fused)
        return out.squeeze(-1)