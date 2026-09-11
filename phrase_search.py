from preprocessing import preprocess


def phrase_search(query, positional_index):
    """
    Exact phrase search using the positional index.

    The query is preprocessed with the same preprocess() function used for
    documents, so terms line up (stopwords dropped, stemmed) before matching.

    Returns a sorted list of doc_ids (increasing) where the query terms
    appear consecutively, in order, exactly as given.
    """

    terms = preprocess(query)

    if not terms:
        return []

    # Any term missing from the vocabulary entirely -> no document can match.
    for term in terms:
        if term not in positional_index:
            return []

    # Candidate docs: intersection of doc sets for every term.
    candidate_docs = set(positional_index[terms[0]].keys())
    for term in terms[1:]:
        candidate_docs &= set(positional_index[term].keys())

    matches = []

    for doc_id in candidate_docs:
        # Positions of the first term anchor the phrase; for each such
        # position p, check terms[1] is at p+1, terms[2] at p+2, etc.
        first_term_positions = positional_index[terms[0]][doc_id]

        for start_position in first_term_positions:
            if _phrase_matches_at(terms, positional_index, doc_id, start_position):
                matches.append(doc_id)
                break

    return sorted(matches)


def _phrase_matches_at(terms, positional_index, doc_id, start_position):
    """Check if terms[1:] occupy consecutive positions after start_position."""

    for offset, term in enumerate(terms[1:], start=1):
        expected_position = start_position + offset
        if expected_position not in positional_index[term][doc_id]:
            return False

    return True


def proximity_search(term1, term2, k, positional_index):
    """
    WITHIN/k proximity search between two terms using the positional index.

    Both terms are preprocessed individually (each should be a single word
    already, but running them through preprocess() keeps stemming/case
    consistent with the index).

    A document matches if some occurrence of term1 and some occurrence of
    term2 are at most k positions apart (unordered - proximity, not phrase).

    Returns a sorted list of matching doc_ids (increasing).
    """

    term1_processed = preprocess(term1)
    term2_processed = preprocess(term2)

    if not term1_processed or not term2_processed:
        return []

    term1 = term1_processed[0]
    term2 = term2_processed[0]

    if term1 not in positional_index or term2 not in positional_index:
        return []

    candidate_docs = set(positional_index[term1].keys()) & set(positional_index[term2].keys())

    matches = []

    for doc_id in candidate_docs:
        positions1 = positional_index[term1][doc_id]
        positions2 = positional_index[term2][doc_id]

        if _within_k(positions1, positions2, k):
            matches.append(doc_id)

    return sorted(matches)


def _within_k(positions1, positions2, k):
    """True if any position in positions1 is within k of any in positions2."""

    for p1 in positions1:
        for p2 in positions2:
            if abs(p1 - p2) <= k:
                return True

    return False
