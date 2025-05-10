# models/text_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

"""
TextEncoder：使用預訓練 BERT 產生詞向量，再經由 LSTM 建模序列依賴，
最終輸出每個 token 的 contextualized 表徵序列。
"""


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', hidden_size=768):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state  # (B, L, H)
        lstm_out, _ = self.lstm(cls_embeddings)
        return lstm_out
