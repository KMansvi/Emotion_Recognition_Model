# Emotion_Recognition_Model

This repository contains the code, model, and results of a multi-label emotion recognition system that jointly processes textual data and emojis using a transformer-based architecture (RoBERTa). The model is trained and evaluated on a modified version of the GoEmotions dataset enriched with emoji-based annotations. Emotions play a crucial role in human communication, especially on digital platforms where emojis often complement or substitute textual cues. Traditional emotion detection systems often overlook emojis or treat them separately. This project proposes a unified approach to jointly analyze both text and emojis as a single input sequence, enabling a more accurate and context-aware emotion detection. Our model predicts multiple emotions from a given sentence that contains both text and emojis, capturing the nuanced emotional expressions frequently used in real-world social media and chat messages.

🧠 Model Architecture
Base Model: RoBERTa-base
Task Type: Multi-label Emotion Classification
Input: Concatenated sequence of text and emojis
Output: Multiple emotion labels (up to 3 per sample)
Loss Function: Binary Cross Entropy (BCEWithLogitsLoss)
Optimizer: AdamW with weight decay
Scheduler: Linear scheduler with warm-up
Evaluation Metrics: Accuracy, Micro/Macro F1-Score
Class Weights: Automatically computed to handle imbalance

⚙️ How to Run
- Install the dependencies: pip install -r requirements.txt
- Train the model: python train.py
- Evaluate the model: python evaluate.py
- View metrics: Results will include F1-score, accuracy, and a confusion matrix on the test set.

📈 Results & Performance
The model achieved high multi-label classification performance, accurately identifying subtle emotional signals from both text and emoji. Confusion matrices and per-class metrics demonstrate that the joint processing of emojis and text improves recognition compared to traditional text-only systems. The architecture is platform-agnostic, as it treats emoji as Unicode characters without relying on glyph-based rendering.

✅ Features
- Joint text + emoji processing
- Multi-label prediction
- Automatic class imbalance handling
- Preprocessing pipeline
- Easy evaluation and confusion matrix generation
- Fine-tunable RoBERTa transformer
