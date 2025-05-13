# train.py
"""
訓練腳本：融合多模態資訊進行迴歸任務
模態包括：Title、Hashtag（Topic+Graph）、Image、Social
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime

from data_scripts.dataset import MultimodalDataset
from data_scripts.transform import image_transforms
from models.multimodal_net import MultimodalNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
VAL_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 3

# 時間戳記
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f'./experiments/checkpoints/{timestamp}'
os.makedirs(save_dir, exist_ok=True)

# Tokenizer 載入
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 載入並切分訓練資料
dataset = MultimodalDataset(
    csv_path='data/train.csv',
    tokenizer=tokenizer,
    image_root='data',
    transform=image_transforms
)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型建立
topic_dim = dataset[0]['topic'].shape[0]
graph_dim = dataset[0]['graph'].shape[0]
social_dim = dataset[0]['social'].shape[0]
model = MultimodalNet(feature_dims=[768, topic_dim + graph_dim, 768, social_dim]).to(DEVICE)

# 損失與優化器
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# early stopping 變數
best_val_loss = float('inf')
all_train_losses = []
all_val_losses = []
patience_counter = 0

# 訓練迴圈
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in pbar:
        title_ids = batch['title_input_ids'].to(DEVICE)
        title_mask = batch['title_mask'].to(DEVICE)
        topic = batch['topic'].to(DEVICE)
        graph = batch['graph'].to(DEVICE)
        topic_graph = torch.cat([topic, graph], dim=1)  # 合併 hashtag 表徵
        images = batch['image'].to(DEVICE)
        social = batch['social'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model((title_ids, title_mask), topic_graph, images, social)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)
    all_train_losses.append(avg_train_loss)

    # 驗證階段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            title_ids = batch['title_input_ids'].to(DEVICE)
            title_mask = batch['title_mask'].to(DEVICE)
            topic = batch['topic'].to(DEVICE)
            graph = batch['graph'].to(DEVICE)
            topic_graph = torch.cat([topic, graph], dim=1)
            images = batch['image'].to(DEVICE)
            social = batch['social'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model((title_ids, title_mask), topic_graph, images, social)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    all_val_losses.append(avg_val_loss)
    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 儲存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved new best model to {model_path}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"⏹ Early stopping triggered after {epoch+1} epochs.")
            break

# 繪製 loss 曲線
plt.figure()
plt.plot(range(1, len(all_train_losses)+1), all_train_losses, label='Train Loss')
plt.plot(range(1, len(all_val_losses)+1), all_val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.title('Training & Validation Loss Curve')
plt.legend()
plt.grid(True)
fig_path = os.path.join(save_dir, 'loss_curve.png')
plt.savefig(fig_path)
print(f"📈 Loss curve saved to {fig_path}")
