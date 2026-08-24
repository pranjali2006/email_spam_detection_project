import string

from nltk.corpus import stopwords


STOP_WORDS = set(stopwords.words("english"))


def preprocess_message(message: str) -> str:
    message = message.lower()
    message = " ".join(
        word for word in message.split() if word not in STOP_WORDS
    )
    return message.translate(str.maketrans("", "", string.punctuation))