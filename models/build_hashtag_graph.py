# build_hashtag_graph.py (with NeighborLoader)
"""
此腳本負責：
1. 從 train_data.json / test_data.json 建立 hashtag 共現圖
2. 使用 GraphSAGE + NeighborLoader 進行 mini-batch 訓練
3. 儲存每個 hashtag 節點的向量到 graph_emb_dict.pkl
"""

import json
import pandas as pd
import networkx as nx
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import pickle
import os

# 參數
json_paths = ['./data/train_data.json', './data/test_data.json']
emb_dim = 64
output_path = './experiments/checkpoints/graph_emb_dict.pkl'

# 1. 載入資料並抽出 hashtag 列表
print("📥 載入資料...")
all_hashtag_lists = []
for path in json_paths:
    with open(path) as f:
        records = json.load(f)
        for r in records:
            hashtags = r.get('Alltags', '')
            if hashtags:
                tag_list = hashtags.strip().split()
                all_hashtag_lists.append(tag_list)

# 2. 建立 hashtag 共現圖
print("🔗 建立共現圖...")
print(f"  - 總 hashtag 數量: {len(all_hashtag_lists)}")
G = nx.Graph()
for tags in all_hashtag_lists:
    for a, b in combinations(set(tags), 2):
        if G.has_edge(a, b):
            G[a][b]['weight'] += 1
        else:
            G.add_edge(a, b, weight=1)

nodes = list(G.nodes)
le = LabelEncoder()
le.fit(nodes)
id2tag = {i: tag for i, tag in enumerate(le.classes_)}

src = []
dst = []
for u, v in G.edges():
    src.append(le.transform([u])[0])
    dst.append(le.transform([v])[0])

edge_index = torch.tensor([src, dst], dtype=torch.long)
x = torch.eye(len(nodes))
data = Data(x=x, edge_index=edge_index)
data.num_nodes = x.size(0)  # ✅ VERY IMPORTANT

# 3. 定義 GraphSAGE 模型
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

# 4. 使用 NeighborLoader 訓練
print("🧠 Mini-batch 訓練 GraphSAGE...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(x.size(1), 128, emb_dim).to(device)
data = data.to(device)

train_loader = NeighborLoader(
    data,
    num_neighbors=[10, 10],
    batch_size=512,
    shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = (out @ out.T).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}, Loss: {total_loss:.4f}")

# 5. 全圖推論（產生完整節點嵌入）
print("📤 推論嵌入並儲存...")
model.eval()
with torch.no_grad():
    final_emb = model(data.x.to(device), data.edge_index.to(device)).cpu()

graph_emb_dict = {id2tag[i]: final_emb[i].numpy() for i in range(len(id2tag))}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'wb') as f:
    pickle.dump(graph_emb_dict, f)

print(f"✅ Graph embeddings saved to {output_path}")