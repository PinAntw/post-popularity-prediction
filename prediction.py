# prediction.py
"""
使用訓練好的 MultimodalNet（BERTopic 向量版）對 test.csv 進行預測，
輸出包含 Pid 與預測 label 的 CSV 檔。
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import os

from data_scripts.dataset import MultimodalDataset
from data_scripts.transform import image_transforms
from models.multimodal_net import MultimodalNet

# ====== 設定裝置與路徑 ======
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'experiments/checkpoints/20250511_115435/best_model.pt'  # ✔ 替換為實際訓練結果路徑
CSV_PATH = 'data/test.csv'
IMG_ROOT = 'data'
OUTPUT_PATH = 'results/predict_result.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ====== 載入 tokenizer ======
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ====== 載入資料集 ======
test_dataset = MultimodalDataset(
    csv_path=CSV_PATH,
    tokenizer=tokenizer,
    image_root=IMG_ROOT,
    transform=image_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ====== 模型初始化（feature dims: text, topic, image, social） ======
topic_dim = test_dataset[0]['topic'].shape[0]
social_dim = test_dataset[0]['social'].shape[0]
model = MultimodalNet(feature_dims=[768, topic_dim, 768, social_dim]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ====== 推論流程 ======
pids = []
preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        title_ids = batch['title_input_ids'].to(DEVICE)
        title_mask = batch['title_mask'].to(DEVICE)
        topic = batch['topic'].to(DEVICE)
        images = batch['image'].to(DEVICE)
        social = batch['social'].to(DEVICE)

        outputs = model((title_ids, title_mask), topic, images, social)
        preds.extend(outputs.cpu().numpy())

        # 取得對應的 Pid（若 csv 有包含）
        if 'Pid' in batch:
            pids.extend(batch['Pid'])
        else:
            pids.extend(test_dataset.df.iloc[len(pids):len(pids)+len(outputs)]["Pid"].tolist())

# ====== 輸出為 CSV ======
output_df = pd.DataFrame({
    "Pid": pids,
    "label": preds
})
output_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ 預測結果已儲存至：{OUTPUT_PATH}")
