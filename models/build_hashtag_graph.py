# build_hashtag_graph.py (full-graph version with caching and data save)
"""
此腳本負責：
1. 從 train_data.json / test_data.json 建立 hashtag 共現圖（加速版）
2. 使用 GraphSAGE 在整圖上訓練
3. 儲存每個 hashtag 節點的向量到 graph_emb_dict.pkl
4. 圖物件 data 也快取儲存，加速後續使用
"""

import json
import networkx as nx
from collections import defaultdict, Counter
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import pickle
import os

# 參數
json_paths = ['./data/train_data.json', './data/test_data.json']
emb_dim = 64
output_path = './experiments/checkpoints/graph_emb_dict.pkl'
cache_path = './experiments/checkpoints/graph_cache.pkl'
data_cache_path = './experiments/checkpoints/graph_data.pkl'

# ====== 共現圖建立或載入 ======
if os.path.exists(cache_path):
    print("♻️ 偵測到快取，載入共現圖...")
    with open(cache_path, 'rb') as f:
        G, le, id2tag = pickle.load(f)
else:
    print("📥 載入資料並建立共現圖...")
    all_hashtag_lists = []
    for path in json_paths:
        with open(path) as f:
            records = json.load(f)
            for r in records:
                hashtags = r.get('Alltags', '')
                if hashtags:
                    tag_list = hashtags.strip().split()
                    all_hashtag_lists.append(tag_list)

    print("🔗 建立共現圖（加速版）...")
    print(f"  - 總 hashtag 筆數: {len(all_hashtag_lists)}")
    edge_counter = defaultdict(Counter)
    for tags in all_hashtag_lists:
        uniq_tags = list(set(tags))
        for i in range(len(uniq_tags)):
            for j in range(i + 1, len(uniq_tags)):
                a, b = sorted([uniq_tags[i], uniq_tags[j]])
                edge_counter[a][b] += 1

    edges = []
    for a, neighbors in edge_counter.items():
        for b, weight in neighbors.items():
            edges.append((a, b, weight))

    print(f"  - 共計邊數：{len(edges)}")
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    nodes = list(G.nodes)
    le = LabelEncoder()
    le.fit(nodes)
    id2tag = {i: tag for i, tag in enumerate(le.classes_)}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((G, le, id2tag), f)
    print("✅ 共現圖已快取儲存")

# ====== 建立圖資料物件或快取載入 ======
if os.path.exists(data_cache_path):
    print("📦 載入快取的圖資料物件 data...")
    with open(data_cache_path, 'rb') as f:
        data = pickle.load(f)
else:
    print("🧱 建立圖資料物件...")
    tag2id = {tag: idx for idx, tag in enumerate(le.classes_)}
    src = []
    dst = []
    for u, v in G.edges():
        src.append(tag2id[u])
        dst.append(tag2id[v])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # x = torch.eye(len(G.nodes))
    x = torch.randn(len(G.nodes), 64)  # 隨機初始化節點特徵（避免 one-hot 爆顯存）
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = x.size(0)

    with open(data_cache_path, 'wb') as f:
        pickle.dump(data, f)
    print("✅ 圖資料物件已儲存至快取")

# ====== 定義 GraphSAGE 模型 ======
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ====== 全圖訓練 ======
print("🧠 Full-graph 訓練 GraphSAGE...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(data.x.size(1), 128, emb_dim).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(20):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = out.norm(p=2, dim=1).mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}")

# ====== 推論嵌入並儲存 ======
print("📤 推論嵌入並儲存...")
model.eval()
with torch.no_grad():
    final_emb = model(data.x.to(device), data.edge_index.to(device)).cpu()

graph_emb_dict = {id2tag[i]: final_emb[i].numpy() for i in range(len(id2tag))}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(graph_emb_dict, f)

print(f"✅ Graph embeddings saved to {output_path}")
