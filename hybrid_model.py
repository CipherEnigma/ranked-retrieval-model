import heapq

from preprocessing import preprocess
from vsm import calculate_document_vector, calculate_query_vector, cosine_similarity


DEFAULT_LAMBDA = 0.2


def smallest_span(position_lists):
    """
    Given a list of sorted position lists (one per distinct query term found
    in a document), find the smallest window [lo, hi] that contains at
    least one position from every list.

    Classic "smallest range covering elements from k sorted lists" problem,
    solved with a min-heap: keep one pointer per list, always advance the
    list holding the current minimum, and track the tightest window seen.

    Returns the span (hi - lo) as an int, or None if any list is empty.
    """

    if any(len(positions) == 0 for positions in position_lists):
        return None

    pointers = [0] * len(position_lists)

    heap = [
        (position_lists[i][0], i)
        for i in range(len(position_lists))
    ]
    heapq.heapify(heap)

    current_max = max(positions[0] for positions in position_lists)
    best_span = None

    while True:
        current_min, list_index = heapq.heappop(heap)

        span = current_max - current_min
        if best_span is None or span < best_span:
            best_span = span

        pointers[list_index] += 1

        # One list is exhausted -> no wider window can beat what we have.
        if pointers[list_index] == len(position_lists[list_index]):
            break

        next_value = position_lists[list_index][pointers[list_index]]
        heapq.heappush(heap, (next_value, list_index))
        current_max = max(current_max, next_value)

    return best_span


def proximity_boost(query_terms, positional_index, doc_id):
    """
    Proximity boost in [0, 1] for one document given the query's terms.

    Looks up the positions of each distinct query term that occurs in this
    document, finds the smallest span covering all of them, and converts
    that span to a boost: tighter clustering -> higher boost.

    Returns 0.0 unless EVERY distinct query term occurs in this document.
    This is deliberate: proximity should only refine the ranking among
    documents that already fully match the query, never let a partial
    match (missing a term) leapfrog a full match just because the terms
    it does have are close together.
    """

    distinct_terms = set(query_terms)

    if len(distinct_terms) < 2:
        return 0.0

    position_lists = []

    for term in distinct_terms:
        if term in positional_index and doc_id in positional_index[term]:
            position_lists.append(sorted(positional_index[term][doc_id]))
        else:
            return 0.0

    span = smallest_span(position_lists)

    if span is None:
        return 0.0

    # span == 0 means the terms sit right next to each other (best case).
    # Larger span -> boost decays towards 0 but never reaches it exactly.
    return 1.0 / (1.0 + span)


def hybrid_search(engine, query, top_k=10, lambda_weight=DEFAULT_LAMBDA):
    """
    Positional-aware hybrid model: VSM (lnc.ltc) base score + a proximity
    boost from the positional index.

    engine must expose: .documents, .inverted_index, .positional_index

    Uniform model signature: model(engine, query, top_k) -> [(doc_id, score)]
    sorted by decreasing score, ties broken by increasing doc_id.
    """

    documents = engine.documents
    inverted_index = engine.inverted_index
    positional_index = engine.positional_index

    N = len(documents)

    query_terms = preprocess(query)
    query_vector = calculate_query_vector(query, inverted_index, N)

    if not query_vector:
        return []

    scores = []

    for document in documents:
        document_vector = calculate_document_vector(document)
        base_score = cosine_similarity(query_vector, document_vector)

        boost = proximity_boost(query_terms, positional_index, document.docid)

        combined_score = base_score + lambda_weight * boost

        scores.append((document.docid, combined_score))

    scores.sort(key=lambda x: (-x[1], x[0]))

    return scores[:top_k]
