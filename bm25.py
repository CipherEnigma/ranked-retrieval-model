import math
from collections import Counter

from preprocessing import preprocess


def calculate_bm25_score(document, query, inverted_index, avg_doc_length, N,
                        k1=1.5, b=0.75):
    """
    Compute the BM25 score for one document against a query.

    Uses the standard BM25 ranking function:

        score = sum_{t in q} IDF(t) * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl))))

    where:
        - tf is the term frequency in the document
        - doc_len is the number of tokens in the document
        - avgdl is the average document length in the corpus
        - IDF is computed with the Robertson-Sparck Jones form
    """
    if not document.tokens:
        document.tokens = preprocess(document.text)

    query_terms = preprocess(query)
    if not query_terms:
        return 0.0

    query_term_counts = Counter(query_terms)
    document_term_counts = Counter(document.tokens)
    doc_len = len(document.tokens)

    score = 0.0

    for term, query_tf in query_term_counts.items():
        if term not in inverted_index:
            continue

        df = len(inverted_index[term])
        tf = document_term_counts.get(term, 0)

        if tf == 0:
            continue

        # Standard BM25 idf with smoothing.
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
        score += idf * ((tf * (k1 + 1)) / denominator)

    return score


def retrieve_bm25(query, documents, inverted_index, k=10, k1=1.5, b=0.75):
    """
    Retrieve the top-k documents using BM25.

    Returns a list of (doc_id, score) pairs sorted by descending score and
    then by increasing doc_id as a tie-breaker.
    """
    if not documents:
        return []

    N = len(documents)
    avg_doc_length = sum(len(document.tokens) for document in documents) / N

    scores = []

    for document in documents:
        if not document.tokens:
            document.tokens = preprocess(document.text)

        score = calculate_bm25_score(
            document=document,
            query=query,
            inverted_index=inverted_index,
            avg_doc_length=avg_doc_length,
            N=N,
            k1=k1,
            b=b,
        )

        if score > 0.0:
            scores.append((document.docid, score))

    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:k]


def bm25_model(engine, query, top_k=10, k1=1.5, b=0.75):
    """
    Uniform model signature compatible with the project contract:

        model(engine, query, top_k) -> [(doc_id, score)]

    The engine is expected to expose `.documents` and `.inverted_index`.
    """
    documents = engine.documents
    inverted_index = engine.inverted_index
    return retrieve_bm25(query, documents, inverted_index, k=top_k, k1=k1, b=b)


if __name__ == "__main__":
    from corpus_parser import parse_corpus
    from inverted_index import build_inverted_index

    documents = parse_corpus("data/corpus_100.txt")
    for document in documents:
        document.tokens = preprocess(document.text)

    inverted_index = build_inverted_index(documents)
    query = "black cotton shirt"

    results = retrieve_bm25(query, documents, inverted_index, k=10)
    print(f"Query: {query}")
    for doc_id, score in results:
        print(f"{doc_id} -> {score:.6f}")
