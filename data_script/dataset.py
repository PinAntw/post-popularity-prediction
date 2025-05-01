# data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 文字編碼
        title_enc = self.tokenizer(row['Title'], truncation=True, padding='max_length', max_length=15, return_tensors='pt')
        tag_enc = self.tokenizer(row['Alltags'], truncation=True, padding='max_length', max_length=15, return_tensors='pt')

        # 圖片處理
        img_path = os.path.join(self.image_root, row['img_filepath'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # One-hot 類別特徵
        social_cols = [col for col in self.df.columns if col.startswith(('Category_', 'Concept_', 'Subcategory_'))]
        social_tensor = torch.tensor(row[social_cols].values, dtype=torch.float)

        # 標籤處理（若有）
        label_value = row['label'] if 'label' in row and not pd.isna(row['label']) else 0.0
        label = torch.tensor(label_value, dtype=torch.float)

        return {
            'title_input_ids': title_enc['input_ids'].squeeze(0),
            'title_mask': title_enc['attention_mask'].squeeze(0),
            'tag_input_ids': tag_enc['input_ids'].squeeze(0),
            'tag_mask': tag_enc['attention_mask'].squeeze(0),
            'image': image,
            'social': social_tensor,
            'label': label
        }
