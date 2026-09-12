"""
Part E test battery - Track B (positional search).

Required by CP1:
    - >= 5 exact phrase queries
    - >= 3 proximity queries, different k
    - 2 analyses where positional info changes the result set/order

Run: python test_track_b.py
"""

from corpus_parser import parse_corpus
from preprocessing import preprocess
from positional_index import build_positional_index
from phrase_search import phrase_search, proximity_search


def load_index():
    documents = parse_corpus("data/corpus_100.txt")
    for document in documents:
        document.tokens = preprocess(document.text)
    return documents, build_positional_index(documents)


def show(label, result):
    if len(result) > 10:
        print(f"{label}: {len(result)} docs -> {result[:10]} ...")
    else:
        print(f"{label}: {result}")


def main():
    documents, index = load_index()

    print("=" * 70)
    print("EXACT PHRASE QUERIES (>= 5 required)")
    print("=" * 70)
    show('1. "cotton crew neck"', phrase_search("cotton crew neck", index))
    show('2. "regular fit"', phrase_search("regular fit", index))
    show('3. "everyday indian wear"', phrase_search("everyday indian wear", index))
    show('4. "durable stitching"', phrase_search("durable stitching", index))
    show('5. "wardrobe essentials"', phrase_search("wardrobe essentials", index))
    show('6. "zip fly"', phrase_search("zip fly", index))

    print()
    print("=" * 70)
    print("PROXIMITY QUERIES, WITHIN/k (>= 3 required, different k)")
    print("=" * 70)
    show("1. cotton WITHIN/3 black", proximity_search("cotton", "black", 3, index))
    show("2. cotton WITHIN/10 black", proximity_search("cotton", "black", 10, index))
    show("3. denim WITHIN/2 durable", proximity_search("denim", "durable", 2, index))
    show("4. fabric WITHIN/1 breathable", proximity_search("fabric", "breathable", 1, index))

    print()
    print("=" * 70)
    print("ANALYSIS 1 - term order changes the result set (phrase vs reorder)")
    print("=" * 70)
    forward = phrase_search("cotton crew neck", index)
    reordered = phrase_search("neck cotton crew", index)
    print(f'"cotton crew neck"  (in order)   -> {forward}')
    print(f'"neck cotton crew"  (reordered)  -> {reordered}')
    print(
        "Explanation: co-occurrence of the same 3 terms is identical in both "
        "cases (same candidate docs pass the set-intersection stage), but the "
        "positional index enforces consecutive, in-order positions, so only "
        "the correctly-ordered phrase matches. A pure bag-of-words / "
        "inverted-index model (no positions) would return the SAME result "
        "for both queries, since it only sees term co-occurrence."
    )

    print()
    print("=" * 70)
    print("ANALYSIS 2 - proximity k changes the result set/size")
    print("=" * 70)
    k0 = proximity_search("cotton", "black", 0, index)
    k3 = proximity_search("cotton", "black", 3, index)
    k20 = proximity_search("cotton", "black", 20, index)
    print(f"cotton WITHIN/0  black -> {len(k0)} docs: {k0}")
    print(f"cotton WITHIN/3  black -> {len(k3)} docs: {k3}")
    print(f"cotton WITHIN/20 black -> {len(k20)} docs: {k20}")
    print(
        "Explanation: as k grows, the result set is monotonically "
        "non-decreasing (every doc matching a smaller k also matches a "
        "larger k), because a wider position window only admits more "
        "documents, never fewer. This shows positional distance - not just "
        "presence of both terms - is what the WITHIN/k operator is scoring."
    )


if __name__ == "__main__":
    main()
