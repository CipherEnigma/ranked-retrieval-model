from preprocessing import preprocess
from vsm import calculate_document_vector, calculate_query_vector
from hybrid_model import proximity_boost, DEFAULT_LAMBDA


def explain_score(query, document, inverted_index, positional_index, N,
                   lambda_weight=DEFAULT_LAMBDA):
    """
    Break down a document's hybrid score into per-term contributions, so
    a ranking result can be explained rather than treated as a black box.

    Returns:
        {
            "doc_id": ...,
            "term_contributions": [
                {"term": ..., "query_weight": ..., "doc_weight": ...,
                 "contribution": query_weight * doc_weight},
                ...
            ],
            "base_score": sum of all term contributions (= VSM cosine score),
            "proximity_boost": ...,
            "lambda_weight": ...,
            "final_score": base_score + lambda_weight * proximity_boost,
        }

    term_contributions is sorted by contribution, descending, so the
    biggest drivers of the score appear first.
    """

    query_vector = calculate_query_vector(query, inverted_index, N)
    document_vector = calculate_document_vector(document)

    term_contributions = []

    for term, query_weight in query_vector.items():
        doc_weight = document_vector.get(term, 0.0)

        if doc_weight == 0.0:
            continue

        term_contributions.append({
            "term": term,
            "query_weight": query_weight,
            "doc_weight": doc_weight,
            "contribution": query_weight * doc_weight,
        })

    term_contributions.sort(key=lambda t: t["contribution"], reverse=True)

    base_score = sum(t["contribution"] for t in term_contributions)

    query_terms = preprocess(query)
    boost = proximity_boost(query_terms, positional_index, document.docid)

    return {
        "doc_id": document.docid,
        "term_contributions": term_contributions,
        "base_score": base_score,
        "proximity_boost": boost,
        "lambda_weight": lambda_weight,
        "final_score": base_score + lambda_weight * boost,
    }


def explain_why_not(query, winner_document, loser_document, inverted_index,
                     positional_index, N, lambda_weight=DEFAULT_LAMBDA):
    """
    Counterfactual explanation: why did `winner_document` outrank
    `loser_document` for this query?

    Returns a dict comparing the two explanations, plus a list of query
    terms the loser is missing or scoring lower on than the winner.
    """

    winner_explanation = explain_score(
        query, winner_document, inverted_index, positional_index, N, lambda_weight
    )
    loser_explanation = explain_score(
        query, loser_document, inverted_index, positional_index, N, lambda_weight
    )

    winner_terms = {t["term"]: t["contribution"] for t in winner_explanation["term_contributions"]}
    loser_terms = {t["term"]: t["contribution"] for t in loser_explanation["term_contributions"]}

    reasons = []

    for term, winner_contribution in winner_terms.items():
        loser_contribution = loser_terms.get(term, 0.0)

        if loser_contribution < winner_contribution:
            if term not in loser_terms:
                reasons.append(f'"{term}" does not occur in {loser_document.docid} at all')
            else:
                reasons.append(
                    f'"{term}" contributes less in {loser_document.docid} '
                    f"({loser_contribution:.4f} vs {winner_contribution:.4f})"
                )

    if winner_explanation["proximity_boost"] > loser_explanation["proximity_boost"]:
        reasons.append(
            f"query terms are more tightly clustered in {winner_document.docid} "
            f"(proximity boost {winner_explanation['proximity_boost']:.4f} "
            f"vs {loser_explanation['proximity_boost']:.4f})"
        )

    return {
        "winner": winner_explanation,
        "loser": loser_explanation,
        "reasons": reasons,
    }


def positional_heatmap(query, document, positional_index):
    """
    Build a simple positional heatmap: for each token position in the
    document, note which query term (if any) occurs there.

    Returns a list of (position, token_or_None) tuples the length of the
    document's token stream, where token_or_None is the matched query term
    at that position, or None if no query term occurs there.
    """

    query_terms = set(preprocess(query))

    heatmap = [None] * len(document.tokens)

    for term in query_terms:
        if term not in positional_index or document.docid not in positional_index[term]:
            continue

        for position in positional_index[term][document.docid]:
            heatmap[position] = term

    return list(enumerate(heatmap))


def render_heatmap(heatmap, width=100):
    """
    Render a positional heatmap as a compact text strip: '.' for no match,
    the first letter of the term (uppercased) where a query term occurs.
    Useful for a quick terminal/CLI visualization.
    """

    symbols = []

    for _, term in heatmap:
        symbols.append(term[0].upper() if term else ".")

    line = "".join(symbols)

    rows = [line[i:i + width] for i in range(0, len(line), width)]
    return "\n".join(rows)
