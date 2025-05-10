# test_train_small.py
"""
這是一個快速測試用的訓練腳本：
僅使用少量訓練與驗證資料（20 / 10 筆），確認模型能否學習、loss 是否下降。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from transformers import BertTokenizer
from tqdm import tqdm
import os

from data_scripts.dataset import MultimodalDataset
from data_scripts.transform import image_transforms
from models.multimodal_net import MultimodalNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4

# 載入 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 載入資料集
full_dataset = MultimodalDataset(
    csv_path='data/train.csv',
    tokenizer=tokenizer,
    image_root='data',
    transform=image_transforms
)

# 小樣本切分（前 20 筆 train, 10 筆 val）
train_subset = Subset(full_dataset, range(20))
val_subset = Subset(full_dataset, range(20, 30))
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

# 模型初始化
social_dim = full_dataset[0]['social'].shape[0]
model = MultimodalNet(feature_dims=[768, 768, 768, social_dim]).to(DEVICE)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("🚀 開始小規模訓練測試...")

# 訓練迴圈
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        title_ids = batch['title_input_ids'].to(DEVICE)
        title_mask = batch['title_mask'].to(DEVICE)
        tag_ids = batch['tag_input_ids'].to(DEVICE)
        tag_mask = batch['tag_mask'].to(DEVICE)
        images = batch['image'].to(DEVICE)
        social = batch['social'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model((title_ids, title_mask), (tag_ids, tag_mask), images, social)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # 驗證
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model((batch['title_input_ids'].to(DEVICE), batch['title_mask'].to(DEVICE)),
                             (batch['tag_input_ids'].to(DEVICE), batch['tag_mask'].to(DEVICE)),
                             batch['image'].to(DEVICE),
                             batch['social'].to(DEVICE))
            loss = criterion(outputs, batch['label'].to(DEVICE))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# 額外印出預測範例
print("\n🔍 預測示範：")
model.eval()
with torch.no_grad():
    for batch in val_loader:
        preds = model((batch['title_input_ids'].to(DEVICE), batch['title_mask'].to(DEVICE)),
                      (batch['tag_input_ids'].to(DEVICE), batch['tag_mask'].to(DEVICE)),
                      batch['image'].to(DEVICE),
                      batch['social'].to(DEVICE))
        print("Predicted:", preds[:5].cpu().numpy())
        print("Actual:   ", batch['label'][:5].numpy())
        break
