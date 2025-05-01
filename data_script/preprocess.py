# data/preprocess.py

"""
任務：

處理 JSON → 產出csv檔案：
Encoded category, concept, subcategory → one-hot
"""

import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# 載入訓練與測試 JSON（格式為 JSON 陣列）
with open('./data/train_data.json') as f:
    train_data = json.load(f)

with open('./data/test_data.json') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# 確保必要欄位存在
cat_cols = ['Category', 'Concept', 'Subcategory']
for col in cat_cols:
    if col not in train_df.columns:
        train_df[col] = ""
    if col not in test_df.columns:
        test_df[col] = ""

# 擴充所有資料，方便 encoder 同時 fit
all_df = pd.concat([train_df, test_df], axis=0)

# One-hot encode 類別欄
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(all_df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# 合併編碼結果
all_df = pd.concat([all_df.reset_index(drop=True), encoded_df], axis=1).drop(columns=cat_cols)
print(f"✅ all-df維度：{all_df.iloc[1].shape}")

# 切回 train / test
train_df = all_df.iloc[:len(train_df)].copy()
label_df = pd.read_csv('./data/train_label.csv') 
train_df = train_df.merge(label_df, on='Pid', how='left')

test_df = all_df.iloc[len(train_df):].copy()

# 儲存 CSV
train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)

# 儲存 encoder 以供推論階段使用
os.makedirs('./experiments/checkpoints', exist_ok=True)
with open('./experiments/checkpoints/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)