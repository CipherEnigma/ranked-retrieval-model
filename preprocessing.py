import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Deployed environments (e.g. Streamlit Cloud) start with no NLTK corpora
# downloaded, so fetch the stopwords list on demand if it's missing.
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


# Stopword policy: NLTK's standard English stopword list (~179 words e.g.
# "the", "is", "and", "of") is used as-is, with no additions or removals.
# These words carry no discriminative power for a clothing-product corpus
# (they don't distinguish one item from another), so dropping them shrinks
# the vocabulary/index without losing retrieval-relevant signal. The same
# list is applied identically to documents and queries via this shared
# preprocess() function.
STOP_WORDS = set(stopwords.words("english"))

# Single shared stemmer instance, reused across all preprocess() calls.
_stemmer = PorterStemmer()


def preprocess(text):
    """
    Preprocessing pipeline:

    1. Lowercase
    2. Strip punctuation
    3. Tokenize
    4. Remove stopwords
    5. Stem (Porter)

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

    # --------------------------------------------------
    # 5. STEM (PORTER)
    # --------------------------------------------------
    tokens = [_stemmer.stem(token) for token in tokens]

    return tokens