"""Text preprocessing utilities shared by training and inference."""
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Normalize text by removing punctuation, special characters and stopwords."""
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"[^a-z\\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    return ' '.join(tokens)
