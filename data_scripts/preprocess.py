import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from bertopic import BERTopic
from tqdm import tqdm
import pandas as pd
import json
import numpy as np
import pickle
import glob

# 參數
cat_cols = ['Category', 'Concept', 'Subcategory']
emb_path = './experiments/checkpoints/graph_emb_dict.pkl'
pca_dim = 6
image_root = './data/train'  # 圖片資料根目錄

# 載入 JSON
with open('./data/train_data.json') as f:
    train_data = json.load(f)
with open('./data/test_data.json') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
train_len = len(train_df)

# 補齊欄位
for col in cat_cols:
    train_df[col] = train_df.get(col, "")
    test_df[col] = test_df.get(col, "")

# 合併資料
all_df = pd.concat([train_df, test_df], axis=0)


# 步驟 1：先處理缺漏標記欄位
all_df['is_title_missing'] = all_df['Title'].fillna('').astype(str).str.strip().eq('').astype(int)
# 步驟 2：再把空字串改為 'None'
all_df['Title'] = all_df['Title'].fillna('').astype(str).apply(lambda x: 'None' if x.strip() == '' else x)

# 時間相關
all_df['post_dt'] = pd.to_datetime(all_df['Postdate'], unit='s')
# 幾點發文（0~23）
all_df['post_hour'] = all_df['post_dt'].dt.hour
# 禮拜幾（0=Mon, 6=Sun）
all_df['post_weekday'] = all_df['post_dt'].dt.weekday
# 是否夜間發文（0~6 → 早上， 18~23 → 晚上，其他為白天）
def classify_period(h):
    if h < 6:
        return 'midnight'
    elif h < 12:
        return 'morning'
    elif h < 18:
        return 'afternoon'
    else:
        return 'evening'
all_df['post_period'] = all_df['post_hour'].apply(classify_period)

# One-hot encode 分類欄  
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(all_df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=all_df.index)

# 加入其他 social 特徵
encoded_df['is_title_missing'] = all_df['is_title_missing'].values
# 整數類時間特徵
encoded_df['post_hour'] = all_df['post_hour']
encoded_df['post_weekday'] = all_df['post_weekday']
# 將 post_period 做 one-hot 編碼（morning / afternoon / evening / midnight）
period_dummies = pd.get_dummies(all_df['post_period'], prefix='period')
encoded_df = pd.concat([encoded_df, period_dummies], axis=1)

# 檢查 hashtag 缺漏數
empty_tag_count = all_df['Alltags'].astype(str).str.strip().eq('').sum()
print(f"🔍 沒有 hashtag 的貼文數量：{empty_tag_count}")

# BERTopic
print("🚀 建立 BERTopic 模型...")
docs = all_df['Alltags'].astype(str).fillna("None").tolist()
topic_model = BERTopic(language="multilingual", min_topic_size=10, calculate_probabilities=True)
topics, probs = topic_model.fit_transform(docs)
topic_df = pd.DataFrame(probs, columns=[f"Topic_{i}" for i in range(probs.shape[1])])
topic_df = topic_df.fillna(0)
print(f"✅ BERTopic 主題維度：{topic_df.shape[1]}")

# PCA 降維社群特徵
print("📉 執行 PCA 降維...")
pca = PCA(n_components=pca_dim)
social_pca = pca.fit_transform(encoded_df)
social_df = pd.DataFrame(social_pca, columns=[f"social_pca_{i}" for i in range(pca_dim)])

# GraphSAGE hashtag 嵌入
print("🌐 載入 GraphSAGE hashtag 嵌入...")
with open(emb_path, 'rb') as f:
    graph_emb_dict = pickle.load(f)
emb_dim = next(iter(graph_emb_dict.values())).shape[0]

def get_graph_emb_vector(tag_str):
    if not isinstance(tag_str, str):
        return np.zeros(emb_dim)
    tags = tag_str.strip().split()
    vecs = [graph_emb_dict[t] for t in tags if t in graph_emb_dict]
    return np.mean(vecs, axis=0) if vecs else np.zeros(emb_dim)

print("🔁 計算 hashtag 平均嵌入...")
tqdm.pandas()
graph_emb_arr = all_df['Alltags'].astype(str).progress_apply(get_graph_emb_vector)
graph_emb_df = pd.DataFrame(graph_emb_arr.tolist(), columns=[f"graph_emb_{i}" for i in range(emb_dim)])

no_embed_count = sum((not any(t in graph_emb_dict for t in str(tags).strip().split())) for tags in all_df['Alltags'])
print(f"⚠️ 沒有有效 hashtag 嵌入的樣本數：{no_embed_count}")

# 合併所有特徵
all_df = all_df.reset_index(drop=True)
all_df = pd.concat([all_df, topic_df, social_df, graph_emb_df], axis=1)
print(f"✅ 合併完成，all_df 維度：{all_df.shape}")

# 切回 train / test
train_df = all_df.iloc[:train_len].copy()
test_df = all_df.iloc[train_len:].copy()

# 合併 label
label_df = pd.read_csv('./data/train_label.csv')
train_df = train_df.merge(label_df, on='Pid', how='left')

# 儲存資料與模型
train_df.to_csv('./data/train.csv', index=False)
test_df.to_csv('./data/test.csv', index=False)

os.makedirs('./experiments/checkpoints', exist_ok=True)
with open('./experiments/checkpoints/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('./experiments/checkpoints/pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

topic_model.save("./experiments/checkpoints/bertopic_model")
print("✅ 資料預處理完成，已儲存至 CSV 與模型檔案")
