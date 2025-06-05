import torch
import json
import sys
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch.nn.functional as F

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import EmojiDataset
from src.model import EmojiEmotionModel


def evaluate(test_csv, model_path, max_length=64, batch_size=16):
    # Load label2idx and create idx2label
    with open('label2idx.json') as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test dataset and dataloader
    test_dataset = EmojiDataset(test_csv, label2idx=label2idx, max_length=max_length, single_label=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Load model with correct number of labels
    num_labels = len(label2idx)
    model = EmojiEmotionModel(num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Ensure only present labels are reported
    present_labels = sorted(set(all_labels) | set(all_preds))
    target_names_filtered = [idx2label[i] for i in present_labels]

    # Print classification results
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds,
                                labels=present_labels,
                                target_names=target_names_filtered,
                                zero_division=0))
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy: {acc:.4f}")


if __name__ == "__main__":
    test_csv = "data/test.csv"
    model_path = "models/emoji_roberta.pth"
    evaluate(test_csv, model_path)
