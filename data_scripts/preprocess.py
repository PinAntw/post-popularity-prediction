# preprocess.py
"""
此腳本負責：
1. 載入訓練與測試 JSON 檔案。
2. 對分類欄位（Category, Concept, Subcategory）進行 One-hot 編碼。
3. 使用 BERTopic 對 Alltags 欄位做主題建模並加入 topic 向量。
4. 對 One-hot 編碼結果做 PCA 降維（符合論文做法）。
5. 加入 GraphSAGE hashtag 嵌入（共現圖產出）
6. 合併訓練資料與對應標籤。
7. 將處理後資料儲存成 CSV，並儲存 Encoder 與 PCA 模型。
"""

import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import pickle
import os
from bertopic import BERTopic
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 載入訓練與測試 JSON（格式為 JSON 陣列）
with open('./data/train_data.json') as f:
    train_data = json.load(f)
with open('./data/test_data.json') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
train_len = len(train_df)

# 確保必要欄位存在
cat_cols = ['Category', 'Concept', 'Subcategory']
for col in cat_cols:
    if col not in train_df.columns:
        train_df[col] = ""
    if col not in test_df.columns:
        test_df[col] = ""

# 合併資料以便做 fit
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

# === 對 One-hot encoded social features 做 PCA 降維 ===
print("📉 執行 PCA 降維...")
social_cols = encoded_df.columns.tolist()
pca = PCA(n_components=6)  # 降到 6 維
social_pca = pca.fit_transform(encoded_df[social_cols])
social_df = pd.DataFrame(social_pca, columns=[f"social_pca_{i}" for i in range(social_pca.shape[1])])

# === 加入 GraphSAGE hashtag 嵌入（平均所有標籤節點） ===
print("🌐 載入 GraphSAGE hashtag 嵌入...")
with open('./experiments/checkpoints/graph_emb_dict.pkl', 'rb') as f:
    graph_emb_dict = pickle.load(f)
emb_dim = next(iter(graph_emb_dict.values())).shape[0]

def get_graph_emb_vector(tag_str):
    tags = tag_str.strip().split()
    vecs = [graph_emb_dict[t] for t in tags if t in graph_emb_dict]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(emb_dim)

graph_emb_arr = all_df['Alltags'].astype(str).apply(get_graph_emb_vector)
graph_emb_df = pd.DataFrame(graph_emb_arr.tolist(), columns=[f"graph_emb_{i}" for i in range(emb_dim)])

# 合併所有特徵
all_df = all_df.reset_index(drop=True)
all_df = pd.concat([all_df, topic_df, social_df, graph_emb_df], axis=1)
print(f"✅ all-df維度（含 topic+social_pca+graph_emb）：{all_df.shape}")

# 切回 train / test
train_df = all_df.iloc[:train_len].copy()
test_df = all_df.iloc[train_len:].copy()

# 合併 label
label_df = pd.read_csv('./data/train_label.csv')
train_df = train_df.merge(label_df, on='Pid', how='left')

# 儲存 CSV
train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)

# 儲存 Encoder 與 PCA
os.makedirs('./experiments/checkpoints', exist_ok=True)
with open('./experiments/checkpoints/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('./experiments/checkpoints/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

# 儲存 BERTopic 模型（可選）
topic_model.save("./experiments/checkpoints/bertopic_model")