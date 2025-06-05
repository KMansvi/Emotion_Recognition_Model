import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train
from src.evaluate import evaluate
from src.predict import predict  # your predict function must also return a single label emoji emotion
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"
    test_csv = "data/test.csv"
    model_save_path = "best_model.pt"

    # Train the model
    model, label2idx = train(train_csv, val_csv, model_save_path)

    # Evaluate on the test set
    evaluate(test_csv, model_save_path)

    # Interactive single-label prediction
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    print("\n🔮 Ready for single-label emoji emotion prediction. Type 'exit' to quit.")
    while True:
        txt = input("\nEnter text: ")
        if txt.lower() == 'exit':
            break
        prediction = predict(txt, model, tokenizer, label2idx)
        print("Predicted Emotion:", prediction)
