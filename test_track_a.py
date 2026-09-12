"""
Part E test battery - Track A (free-text ranked retrieval).

Required by CP1:
    - >= 10 free-text queries
    - at least one query containing an out-of-vocabulary term
    - report the top-10 ranked documents for every query

Run: python test_track_a.py
"""

from corpus_parser import parse_corpus
from preprocessing import preprocess
from inverted_index import build_inverted_index
from vsm import retrieve


QUERIES = [
    "men's black cotton t-shirt",
    "women's casual blue shirt",
    "men's grey slim fit jeans",
    "women's printed kurta",
    "daily wear cotton saree",
    "green midi dress",
    "black fleece hoodie",
    "beige winter jacket",
    "olive green stretch leggings",
    "black cotton shirt zyzzyva",
]


def load_index():
    """Load the corpus and build the same inverted index used by retrieval."""
    documents = parse_corpus("data/corpus_100.txt")
    inverted_index = build_inverted_index(documents)
    return documents, inverted_index


def show_results(number, query, results, query_terms, vocabulary):
    """Print query terms and the ranked top-10 result list."""
    out_of_vocabulary = sorted(
        term for term in query_terms if term not in vocabulary
    )

    print(f"{number}. Query: {query}")
    print(f"   Processed terms: {query_terms}")
    if out_of_vocabulary:
        print(f"   OOV terms ignored: {out_of_vocabulary}")

    if not results:
        print("   No matching documents")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"   {rank:2}. {doc_id} -> {score:.6f}")


def main():
    documents, inverted_index = load_index()
    vocabulary = set(inverted_index)

    print("=" * 70)
    print("TRACK A: FREE-TEXT RANKED RETRIEVAL (TOP-10)")
    print("=" * 70)
    print(f"Corpus documents: {len(documents)}")
    print(f"Vocabulary terms: {len(vocabulary)}")
    print(f"Queries tested: {len(QUERIES)}")
    print()

    for number, query in enumerate(QUERIES, start=1):
        query_terms = preprocess(query)
        results = retrieve(query, documents, inverted_index, k=10)
        show_results(number, query, results, query_terms, vocabulary)
        print()


if __name__ == "__main__":
    main()