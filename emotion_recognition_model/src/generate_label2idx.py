# create_label2idx.py
import pandas as pd
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import preprocess_text

def get_label2idx_from_multiple(files):
    all_labels = set()
    for file in files:
        df = pd.read_csv(file)
        for lbls in df["emotion"]:
            for lbl in lbls.split(","):
                all_labels.add(lbl.strip())
    label2idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    return label2idx

# Update with your actual paths
files = [
    "data/train.csv",
    "data/val.csv",
    "data/test.csv"
]

label2idx = get_label2idx_from_multiple(files)

with open("label2idx.json", "w") as f:
    json.dump(label2idx, f, indent=4)

print("✅ label2idx.json saved with labels:", label2idx.keys())
