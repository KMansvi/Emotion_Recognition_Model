import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def predict(text, model, tokenizer, label2idx, max_length=64):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_length)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    idx2label = {v: k for k, v in label2idx.items()}
    label = idx2label[pred_idx]

    return label
