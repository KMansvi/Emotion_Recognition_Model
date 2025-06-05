import json
import pandas as pd
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from src.utils import preprocess_text

# Add single_label flag in Dataset init

# Add single_label flag in Dataset init

class EmojiDataset(Dataset):
    def __init__(self, csv_path, label2idx, max_length=128, single_label=False):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.label2idx = label2idx
        self.max_length = max_length
        self.single_label = single_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = preprocess_text(row["text"])
        labels = [lbl.strip() for lbl in row["labels"].split(",")]

        encoding = self.tokenizer(text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")

        if self.single_label:
            # Take only first label for single-label classification
            label_str = labels[0]
            if label_str not in self.label2idx:
                raise ValueError(f"Label '{label_str}' not found in label2idx mapping.")
            label_id = self.label2idx[label_str]

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label_id, dtype=torch.long)
            }
        else:
            # Multi-label (if needed later)
            label_ids = [self.label2idx[label] for label in labels if label in self.label2idx]
            multi_hot = torch.zeros(len(self.label2idx))
            multi_hot[label_ids] = 1.0
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": multi_hot
            }
