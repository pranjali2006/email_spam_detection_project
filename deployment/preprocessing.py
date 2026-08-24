import string
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already available
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))


def preprocess_message(message: str) -> str:
    message = message.lower()

    message = " ".join(
        word for word in message.split()
        if word not in STOP_WORDS
    )

    return message.translate(
        str.maketrans("", "", string.punctuation)
    )