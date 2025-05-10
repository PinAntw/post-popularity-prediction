# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np

from data_scripts.dataset import MultimodalDataset
from data_scripts.transform import image_transforms
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.fusion_model import FusionModel

# ====== 設定裝置與隨機種子 ======
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
VAL_SPLIT = 0.1
BATCH_SIZE = 16
MODEL_PATH = './experiments/checkpoints/best_model.pt'

# 舊版模型（不含 tag_encoder）
class OldMultimodalNet(nn.Module):
    def __init__(self, feature_dims):
        super(OldMultimodalNet, self).__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()
        self.fusion = FusionModel(feature_dims)

    def forward(self, text_inputs, image_inputs, extra_features):
        text_feat = self.text_encoder(*text_inputs)[:, 0, :]
        image_feat = self.image_encoder(image_inputs).mean(dim=1)
        fused_output = self.fusion([text_feat, image_feat, extra_features])
        return fused_output


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()

# ====== 載入 tokenizer ======
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ====== 載入完整訓練資料，從中切 validation subset ======
full_dataset = MultimodalDataset(
    csv_path='data/train.csv',
    tokenizer=tokenizer,
    image_root='data',
    transform=image_transforms
)

dataset_size = len(full_dataset)
indices = list(range(dataset_size))
split = int(np.floor(VAL_SPLIT * dataset_size))
np.random.shuffle(indices)
valid_indices = indices[:split]

valid_dataset = Subset(full_dataset, valid_indices)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== 初始化舊版模型與載入權重 ======
social_dim = full_dataset[0]['social'].shape[0]
model = OldMultimodalNet(feature_dims=[768, 768, social_dim]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ====== 評估流程 ======
criterion = nn.L1Loss()
total_loss = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(valid_loader, desc="Evaluating"):
        input_ids = batch['title_input_ids'].to(DEVICE)
        attention_mask = batch['title_mask'].to(DEVICE)
        images = batch['image'].to(DEVICE)
        social = batch['social'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model((input_ids, attention_mask), images, social)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * input_ids.size(0)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

avg_loss = total_loss / len(valid_dataset)
print(f"\n📊 Validation MAE (L1 Loss): {avg_loss:.4f}")

print("\n🔍 預測 vs 真實值（前 10 筆）:")
for i in range(min(10, len(all_preds))):
    print(f"Predicted: {all_preds[i]:.4f} \t Actual: {all_labels[i]:.4f}")
