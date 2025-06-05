import emoji
import re
import pandas as pd

def preprocess_text(text):
    # simple cleaning (customize as needed)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

emoji_emotion_map = {
    # Joy / Happiness
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "😆": "happy", "🙂": "happy",
    "🙃": "happy", "😊": "happy", "☺": "happy", "🥲": "happy", "🤭": "happy", "😺": "happy", "😸": "happy",
    "😅": "joy", "😂": "joy", "🤣": "joy", "😹": "joy",

    # Peaceful
    "😇": "peaceful", "😌": "peaceful",

    # Playful
    "😉": "playful", "😛": "playful", "😝": "playful", "😜": "playful", "🤪": "playful", "😼": "playful",

    # Cool
    "🤓": "cool", "🥶": "cool", "😶‍🌫️": "cool",

    # Hot
    "🥵": "hot",

    # Excited
    "🤩": "excited", "🥳": "excited",

    # Disappointed
    "😒": "disappointed", "😞": "disappointed", "😔": "disappointed",

    # Nervous
    "😟": "nervous",

    # Sadness
    "😢": "sadness", "😭": "sadness", "😿": "sadness",

    # Anger
    "😠": "anger", "😡": "anger", "🤬": "anger", "😤": "anger", "😾": "anger",

    # Love
    "😍": "love", "🥰": "love", "😘": "love", "😗": "love", "😙": "love", "😚": "love",
    "😻": "love", "😽": "love", "🩷": "love", "💓": "love", "💕": "love", "🫶🏻": "love",

    # Yummy
    "😋": "yummy", "🤤": "yummy",

    # Suspicious
    "🤨": "suspicious", "🧐": "suspicious", "🥸": "suspicious", "🤔": "suspicious",

    # Fear / Surprise
    "😱": "fear", "😨": "fear", "😰": "fear", "😧": "fear", "🙀": "fear",
    "😲": "surprise", "😳": "surprise",

    # Disgust
    "😖": "disgust", "🤮": "disgust", "🤢": "disgust", "🥴": "disgust", "😵‍💫": "disgust",

    # Confusion
    "😕": "confusion", "🙁": "confusion", "☹": "confusion", "😫": "confusion",
    "😩": "confusion", "😥": "confusion", "😓": "confusion", "😣": "confusion", "🥺": "confusion",

    # Pride / Gratitude / Other
    "😎": "pride", "😏": "pride", "🙏": "gratitude", "💐": "gratitude", "🥹": "gratitude",
    "😐": "neutral", "🙄": "annoyance"
}

import emoji
import re
import pandas as pd

def preprocess_text(text):
    # simple cleaning (customize as needed)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

emoji_emotion_map = {
    # Joy / Happiness
    "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "😆": "happy", "🙂": "happy",
    "🙃": "happy", "😊": "happy", "☺": "happy", "🥲": "happy", "🤭": "happy", "😺": "happy", "😸": "happy",
    "😅": "joy", "😂": "joy", "🤣": "joy", "😹": "joy",

    # Peaceful
    "😇": "peaceful", "😌": "peaceful",

    # Playful
    "😉": "playful", "😛": "playful", "😝": "playful", "😜": "playful", "🤪": "playful", "😼": "playful",

    # Cool
    "🤓": "cool", "🥶": "cool", "😶‍🌫️": "cool",

    # Excited
    "🤩": "excited", "🥳": "excited",

    # Disappointed
    "😒": "disappointed", "😞": "disappointed", "😔": "disappointed",

    # Nervous
    "😟": "nervous",

    # Sadness
    "😢": "sadness", "😭": "sadness", "😿": "sadness",

    # Anger
    "😠": "anger", "😡": "anger", "🤬": "anger", "😤": "anger", "😾": "anger",

    # Love
    "😍": "love", "🥰": "love", "😘": "love", "😗": "love", "😙": "love", "😚": "love",
    "😻": "love", "😽": "love", "🩷": "love", "💓": "love", "💕": "love", "🫶🏻": "love",

    # Yummy
    "😋": "yummy", "🤤": "yummy",

    # Suspicious
    "🤨": "suspicious", "🧐": "suspicious", "🥸": "suspicious", "🤔": "suspicious",

    # Fear / Surprise
    "😱": "fear", "😨": "fear", "😰": "fear", "😧": "fear", "🙀": "fear",
    "😲": "surprise", "😳": "surprise",

    # Disgust
    "😖": "disgust", "🤮": "disgust", "🤢": "disgust", "🥴": "disgust", "😵‍💫": "disgust",

    # Confusion
    "😕": "confusion", "🙁": "confusion", "☹": "confusion", "😫": "confusion",
    "😩": "confusion", "😥": "confusion", "😓": "confusion", "😣": "confusion", "🥺": "confusion",

    # Pride / Gratitude / Other
    "😎": "pride", "😏": "pride", "🙏": "gratitude", "💐": "gratitude", "🥹": "gratitude",
    "😐": "neutral", "🙄": "annoyance"
}

def get_label2idx(csv_path):
    df = pd.read_csv(csv_path)
    labels = sorted(df["labels"].unique())
    label2idx = {label.strip(): idx for idx, label in enumerate(labels)}
    return label2idx