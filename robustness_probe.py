import random

STOPWORDS_TO_INJECT = ["the", "a", "for", "please", "with", "of"]


# ============================================================
# QUERY PERTURBATIONS
# ============================================================

def inject_typo(query, rng):
    """
    Swap two adjacent characters inside one word (length >= 4) to simulate
    a mistyped query. Picks the first eligible word deterministically
    given the rng seed, so results are reproducible.
    """

    words = query.split()
    eligible_indices = [i for i, w in enumerate(words) if len(w) >= 4]

    if not eligible_indices:
        return query

    index = rng.choice(eligible_indices)
    word = list(words[index])

    position = rng.randint(0, len(word) - 2)
    word[position], word[position + 1] = word[position + 1], word[position]

    words[index] = "".join(word)
    return " ".join(words)


def shuffle_words(query, rng):
    """Reorder the query's words. Bag-of-words models should be unaffected."""

    words = query.split()

    if len(words) < 2:
        return query

    shuffled = words[:]
    while shuffled == words:
        rng.shuffle(shuffled)

    return " ".join(shuffled)


def inject_stopwords(query, rng, count=2):
    """Insert `count` filler stopwords at random positions in the query."""

    words = query.split()

    for _ in range(count):
        filler = rng.choice(STOPWORDS_TO_INJECT)
        position = rng.randint(0, len(words))
        words.insert(position, filler)

    return " ".join(words)


# ============================================================
# STABILITY METRICS
# ============================================================

def overlap_at_k(list1, list2, k=10):
    """Fraction of the top-k doc_ids shared between two ranked lists."""

    top1 = set(list1[:k])
    top2 = set(list2[:k])

    if not top1 and not top2:
        return 1.0

    return len(top1 & top2) / k


def kendall_tau(list1, list2):
    """
    Kendall's tau over the documents common to both ranked lists, i.e. how
    much their RELATIVE ORDER agrees (not just whether they're present).

    Returns a value in [-1, 1] (1 = identical order, -1 = fully reversed),
    or None if fewer than 2 documents are shared (tau undefined).
    """

    common = list(set(list1) & set(list2))

    if len(common) < 2:
        return None

    rank1 = {doc_id: i for i, doc_id in enumerate(list1)}
    rank2 = {doc_id: i for i, doc_id in enumerate(list2)}

    concordant = 0
    discordant = 0

    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]

            direction1 = rank1[a] - rank1[b]
            direction2 = rank2[a] - rank2[b]

            if direction1 * direction2 > 0:
                concordant += 1
            elif direction1 * direction2 < 0:
                discordant += 1

    total_pairs = concordant + discordant

    if total_pairs == 0:
        return None

    return (concordant - discordant) / total_pairs


# ============================================================
# PROBE
# ============================================================

PERTURBATIONS = {
    "typo": inject_typo,
    "shuffle": shuffle_words,
    "stopword_inject": inject_stopwords,
}


def probe_query(query, model_fn, top_k=10, seed=42):
    """
    Run one query and its 3 perturbed variants through model_fn, and
    compare each variant's ranking against the clean query's ranking.

    model_fn(query, top_k) -> [(doc_id, score), ...]

    Returns a dict: {perturbation_name: {"variant_query", "overlap", "tau"}}
    """

    rng = random.Random(seed)

    clean_result = model_fn(query, top_k)
    clean_docs = [doc_id for doc_id, _ in clean_result]

    report = {}

    for name, perturb_fn in PERTURBATIONS.items():
        variant_query = perturb_fn(query, rng)
        variant_result = model_fn(variant_query, top_k)
        variant_docs = [doc_id for doc_id, _ in variant_result]

        report[name] = {
            "variant_query": variant_query,
            "overlap": overlap_at_k(clean_docs, variant_docs, k=top_k),
            "tau": kendall_tau(clean_docs, variant_docs),
        }

    return report


def run_probe_battery(queries, model_fn, top_k=10, seed=42):
    """Run probe_query over a list of base queries and print a summary table."""

    print(f"{'query':<30} {'perturbation':<16} {'overlap@10':<12} {'tau'}")
    print("-" * 75)

    averages = {name: {"overlap": [], "tau": []} for name in PERTURBATIONS}

    for query in queries:
        report = probe_query(query, model_fn, top_k=top_k, seed=seed)

        for name, result in report.items():
            tau_display = f"{result['tau']:.3f}" if result["tau"] is not None else "n/a"
            print(f"{query:<30} {name:<16} {result['overlap']:<12.2f} {tau_display}")

            averages[name]["overlap"].append(result["overlap"])
            if result["tau"] is not None:
                averages[name]["tau"].append(result["tau"])

    print()
    print("AVERAGES:")
    for name, values in averages.items():
        avg_overlap = sum(values["overlap"]) / len(values["overlap"])
        avg_tau = (
            sum(values["tau"]) / len(values["tau"])
            if values["tau"] else None
        )
        tau_display = f"{avg_tau:.3f}" if avg_tau is not None else "n/a"
        print(f"  {name:<16} overlap@10={avg_overlap:.3f}  tau={tau_display}")


if __name__ == "__main__":
    from corpus_parser import parse_corpus
    from preprocessing import preprocess
    from inverted_index import build_inverted_index
    from vsm import retrieve

    documents = parse_corpus("data/corpus_100.txt")
    for document in documents:
        document.tokens = preprocess(document.text)

    inverted_index = build_inverted_index(documents)

    def vsm_model(query, top_k):
        return retrieve(query, documents, inverted_index, k=top_k)

    test_queries = [
        "black cotton shirt",
        "regular fit denim jeans",
        "women's festive saree",
        "grey hoodie for casual wear",
        "linen blend kurta pink",
    ]

    run_probe_battery(test_queries, vsm_model, top_k=10)
