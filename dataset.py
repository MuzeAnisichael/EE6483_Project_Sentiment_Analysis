# dataset.py
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, is_test=False):
        self.texts = texts
        self.labels = labels
        self.is_test = is_test
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sent = str(self.texts[idx])

        # 转为张量
        enc = self.tokenizer(
            sent,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

        if not self.is_test:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item
