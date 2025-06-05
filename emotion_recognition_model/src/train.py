import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import EmojiDataset
from src.model import EmojiEmotionModel
from src.utils import get_label2idx


def train(train_csv, val_csv, model_save_path, epochs=8, batch_size=16, lr=3e-5, max_length=128):
    label2idx = get_label2idx(train_csv)
    with open('label2idx.json', 'w') as f:
        json.dump(label2idx, f, indent=4)

    train_dataset = EmojiDataset(train_csv, label2idx, max_length=max_length, single_label=True)
    val_dataset = EmojiDataset(val_csv, label2idx, max_length=max_length, single_label=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmojiEmotionModel(num_labels=len(label2idx)).to(device)

    # 🧠 Compute class weights from train labels
    df = pd.read_csv(train_csv)
    all_labels = df['labels'].tolist()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print("Class Weights:", class_weights)

    # 🎯 Use class weights in the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    return model, label2idx


if __name__ == "__main__":
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    model_save_path = "models/emoji_roberta.pth"
    train(train_csv, val_csv, model_save_path)
