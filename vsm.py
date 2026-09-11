import math
from collections import Counter

from corpus_parser import parse_corpus
from preprocessing import preprocess
from inverted_index import build_inverted_index


# ============================================================
# DOCUMENT VECTOR: lnc
# ============================================================

def calculate_document_vector(document):
    """
    Calculate the lnc weighted vector for a document.

    lnc:
        l = logarithmic TF
        n = no IDF
        c = cosine normalization

    Weight:
        1 + log10(tf)

    No IDF is applied to document weights.
    """

    # Get term frequencies
    term_frequency = Counter(document.tokens)

    # Calculate logarithmic TF weights
    weights = {}

    for term, tf in term_frequency.items():
        weights[term] = 1 + math.log10(tf)

    # Cosine normalization
    magnitude = math.sqrt(
        sum(weight ** 2 for weight in weights.values())
    )

    if magnitude == 0:
        return {}

    normalized_weights = {
        term: weight / magnitude
        for term, weight in weights.items()
    }

    return normalized_weights


# ============================================================
# QUERY VECTOR: ltc
# ============================================================

def calculate_query_vector(query, inverted_index, N):
    """
    Calculate the ltc weighted vector for a query.

    ltc:
        l = logarithmic TF
        t = IDF
        c = cosine normalization

    Weight:

        (1 + log10(tf)) * log10(N / df)
    """

    # Preprocess query using the same pipeline as documents
    query_tokens = preprocess(query)

    # Get query term frequencies
    term_frequency = Counter(query_tokens)

    weights = {}

    for term, tf in term_frequency.items():

        # Ignore terms that do not occur in the corpus
        if term not in inverted_index:
            continue

        # Document frequency
        df = len(inverted_index[term])

        # Logarithmic TF
        tf_weight = 1 + math.log10(tf)

        # IDF
        idf = math.log10(N / df)

        # TF-IDF
        weights[term] = tf_weight * idf

    # Cosine normalization
    magnitude = math.sqrt(
        sum(weight ** 2 for weight in weights.values())
    )

    if magnitude == 0:
        return {}

    normalized_weights = {
        term: weight / magnitude
        for term, weight in weights.items()
    }

    return normalized_weights


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(query_vector, document_vector):
    """
    Calculate cosine similarity between query and document.

    Since both vectors are already cosine-normalized:

        cosine similarity = dot product
    """

    common_terms = set(query_vector) & set(document_vector)

    score = sum(
        query_vector[term] * document_vector[term]
        for term in common_terms
    )

    return score


# ============================================================
# RANK DOCUMENTS
# ============================================================

def rank_documents(query_vector, document_vectors):
    """
    Calculate similarity scores for all documents
    and return them ranked.

    Tie-break:
        Higher score first.
        If scores are equal, smaller DOCID first.
    """

    scores = []

    for docid, document_vector in document_vectors.items():

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        scores.append((docid, score))

    # Sort:
    # 1. Score descending
    # 2. DOCID ascending for ties
    scores.sort(
        key=lambda x: (-x[1], x[0])
    )

    return scores


# ============================================================
# TOP-K RETRIEVAL
# ============================================================

def retrieve(query, documents, inverted_index, k=10):
    """
    Retrieve the top-k documents for a query.
    """

    N = len(documents)

    # Create document vectors
    document_vectors = {}

    for document in documents:
        document_vectors[document.docid] = (
            calculate_document_vector(document)
        )

    # Create query vector
    query_vector = calculate_query_vector(
        query,
        inverted_index,
        N
    )

    # If no query terms occur in corpus
    if not query_vector:
        return []

    # Rank documents
    ranked_documents = rank_documents(
        query_vector,
        document_vectors
    )

    # Return top k
    return ranked_documents[:k]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. LOAD CORPUS
    # --------------------------------------------------------

    documents = parse_corpus("data/corpus_100.txt")

    print("Number of documents:", len(documents))


    # --------------------------------------------------------
    # 2. PREPROCESS DOCUMENTS
    # --------------------------------------------------------

    for document in documents:
        document.tokens = preprocess(document.text)


    # --------------------------------------------------------
    # 3. BUILD INVERTED INDEX
    # --------------------------------------------------------

    inverted_index = build_inverted_index(documents)

    print("Number of unique terms:", len(inverted_index))


    # --------------------------------------------------------
    # 4. CREATE DOCUMENT VECTORS
    # --------------------------------------------------------

    document_vectors = {}

    for document in documents:

        document_vectors[document.docid] = (
            calculate_document_vector(document)
        )


    # --------------------------------------------------------
    # 5. TEST QUERY
    # --------------------------------------------------------

    query = "black cotton shirt"

    query_vector = calculate_query_vector(
        query,
        inverted_index,
        len(documents)
    )


    print("\nQuery:", query)

    print("\nQuery vector:")
    for term, weight in query_vector.items():
        print(f"{term}: {weight:.6f}")


    # --------------------------------------------------------
    # 6. RANK DOCUMENTS
    # --------------------------------------------------------

    ranked_documents = rank_documents(
        query_vector,
        document_vectors
    )


    # --------------------------------------------------------
    # 7. DISPLAY TOP 10
    # --------------------------------------------------------

    print("\nTop 10 documents:")

    for rank, (docid, score) in enumerate(
        ranked_documents[:10],
        start=1
    ):
        print(
            f"{rank}. {docid} -> {score:.6f}"
        )