# preprocess.py
"""
此腳本負責：
1. 載入訓練與測試 JSON 檔案。
2. 對分類欄位（Category, Concept, Subcategory）進行 One-hot 編碼。
3. 使用 BERTopic 對 Alltags 欄位做主題建模並加入 topic 向量。
4. 合併訓練資料與對應標籤。
5. 將處理後資料儲存成 CSV，並儲存 OneHotEncoder 物件以便日後使用。
"""

import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
import pickle
import os
from bertopic import BERTopic
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 載入訓練與測試 JSON（格式為 JSON 陣列）
with open('./data/train_data.json') as f:
    train_data = json.load(f)

with open('./data/test_data.json') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# 紀錄訓練資料長度
train_len = len(train_df)

# 確保必要欄位存在
cat_cols = ['Category', 'Concept', 'Subcategory']
for col in cat_cols:
    if col not in train_df.columns:
        train_df[col] = ""
    if col not in test_df.columns:
        test_df[col] = ""

# 合併以便進行 encoder.fit
all_df = pd.concat([train_df, test_df], axis=0)

# One-hot encode 類別欄
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(all_df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# === 使用 BERTopic 對 Alltags 做主題建模 ===
print("🚀 建立 BERTopic 模型...")
docs = all_df['Alltags'].astype(str).fillna("None").tolist()
topic_model = BERTopic(language="multilingual", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(docs)

topic_df = pd.DataFrame(probs, columns=[f"Topic_{i}" for i in range(probs.shape[1])])
print(f"✅ BERTopic 主題維度：{topic_df.shape[1]}")

# 合併編碼與主題結果
all_df = pd.concat([all_df.reset_index(drop=True), encoded_df, topic_df], axis=1).drop(columns=cat_cols)
print(f"✅ all-df維度（含 topic）：{all_df.shape}")

# 切回 train / test
train_df = all_df.iloc[:train_len].copy()
test_df = all_df.iloc[train_len:].copy()

# 合併 label
label_df = pd.read_csv('./data/train_label.csv')
train_df = train_df.merge(label_df, on='Pid', how='left')

# 儲存 CSV
train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)

# 儲存 encoder
os.makedirs('./experiments/checkpoints', exist_ok=True)
with open('./experiments/checkpoints/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# 儲存 topic model（可選）
topic_model.save("./experiments/checkpoints/bertopic_model")
