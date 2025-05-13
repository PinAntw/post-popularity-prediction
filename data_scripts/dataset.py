# data_scripts/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df['Title'] = self.df['Title'].fillna('None')
        self.df['Alltags'] = self.df['Alltags'].fillna('None')

        self.tokenizer = tokenizer
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 文字編碼
        title_enc = self.tokenizer(row['Title'], truncation=True, padding='max_length', max_length=30, return_tensors='pt')

        # 圖片處理
        img_path = os.path.join(self.image_root, row['img_filepath'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # PCA 降維後的社會特徵
        social_cols = [col for col in self.df.columns if col.startswith('social_pca_')]
        social_tensor = torch.tensor(row[social_cols].values.astype(float), dtype=torch.float) if social_cols else torch.zeros(64)

        # BERTopic 主題特徵
        topic_cols = [col for col in self.df.columns if col.startswith("Topic_")]
        topic_tensor = torch.tensor(row[topic_cols].astype(float).values, dtype=torch.float) if topic_cols else torch.zeros(1)

        # GraphSAGE hashtag 圖嵌入特徵
        graph_cols = [col for col in self.df.columns if col.startswith("graph_emb_")]
        graph_tensor = torch.tensor(row[graph_cols].astype(float).values, dtype=torch.float) if graph_cols else torch.zeros(1)

        # 標籤處理
        label_value = row['label'] if 'label' in self.df.columns and not pd.isna(row['label']) else 0.0
        label = torch.tensor(label_value, dtype=torch.float)

        return {
            'title_input_ids': title_enc['input_ids'].squeeze(0),
            'title_mask': title_enc['attention_mask'].squeeze(0),
            'image': image,
            'social': social_tensor,
            'topic': topic_tensor,
            'graph': graph_tensor,
            'label': label
        }