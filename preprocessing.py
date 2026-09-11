import string
from nltk.corpus import stopwords


# English stopword list
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text):
    """
    Preprocessing pipeline:

    1. Lowercase
    2. Strip punctuation
    3. Tokenize
    4. Remove stopwords

    """

    # --------------------------------------------------
    # 1. LOWERCASE
    # --------------------------------------------------
    text = text.lower()

    # --------------------------------------------------
    # 2. STRIP PUNCTUATION
    # --------------------------------------------------
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # --------------------------------------------------
    # 3. TOKENIZE
    # --------------------------------------------------
    tokens = text.split()

    # --------------------------------------------------
    # 4. STOPWORD REMOVAL
    # --------------------------------------------------
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return tokens