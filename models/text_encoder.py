# models/text_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel

"""
TextEncoder：使用預訓練 BERT 產生 token-level 表徵，再用 LSTM 建模順序性。
對齊論文作法：保留所有 token 表徵（非僅取 [CLS]），輸出 LSTM hidden state 序列。
"""

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', hidden_size=768):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False  # 論文未用 BiLSTM
        )

    def forward(self, input_ids, attention_mask):
        # BERT 輸出所有 token 的 contextual embedding（B, L, 768）
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_output.last_hidden_state

        # 經過 LSTM，輸出序列 hidden states（B, L, 768）
        lstm_output, _ = self.lstm(token_embeddings)

        return lstm_output  # 每個 token 的表示（可用於 attention）
