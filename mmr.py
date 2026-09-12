"""Maximal Marginal Relevance (MMR) re-ranking for diverse retrieval."""

from vsm import (
    calculate_document_vector,
    calculate_query_vector,
    cosine_similarity,
)


DEFAULT_LAMBDA = 0.7


def rerank_mmr(
    query_vector,
    document_vectors,
    ranked_documents,
    top_k=10,
    lambda_weight=DEFAULT_LAMBDA,
):
    """Re-rank scored documents using Maximal Marginal Relevance.

    MMR selects the document that maximizes:

        lambda * relevance - (1 - lambda) * redundancy

    ``ranked_documents`` must contain ``(doc_id, relevance_score)`` pairs,
    usually from :func:`vsm.rank_documents`. Document vectors should be
    cosine-normalized, as are the vectors produced by the VSM module.

    The returned score is the MMR score. Ties are broken by the document ID
    to match the repository's ranking contract.
    """
    if top_k <= 0 or not ranked_documents:
        return []
    if not 0 <= lambda_weight <= 1:
        raise ValueError("lambda_weight must be between 0 and 1")

    candidates = list(ranked_documents)
    selected = []
    selected_ids = set()

    while candidates and len(selected) < top_k:
        best_index = None
        best_mmr_score = None

        for index, (doc_id, relevance_score) in enumerate(candidates):
            if selected_ids:
                redundancy = max(
                    cosine_similarity(
                        document_vectors.get(doc_id, {}),
                        document_vectors.get(selected_id, {}),
                    )
                    for selected_id in selected_ids
                )
            else:
                redundancy = 0.0

            mmr_score = (
                lambda_weight * relevance_score
                - (1 - lambda_weight) * redundancy
            )

            # Candidates arrive in the base rank order, which already applies
            # the repository's ascending-docID tie-break.
            if best_mmr_score is None or mmr_score > best_mmr_score:
                best_index = index
                best_mmr_score = mmr_score

        doc_id, _ = candidates.pop(best_index)
        selected_ids.add(doc_id)
        selected.append((doc_id, best_mmr_score))

    return selected


def retrieve_mmr(
    query,
    documents,
    inverted_index,
    k=10,
    lambda_weight=DEFAULT_LAMBDA,
):
    """Retrieve VSM results and apply MMR diversity re-ranking."""
    document_vectors = {
        document.docid: calculate_document_vector(document)
        for document in documents
    }
    query_vector = calculate_query_vector(
        query,
        inverted_index,
        len(documents),
    )

    if not query_vector:
        return []

    ranked_documents = sorted(
        (
            (doc_id, cosine_similarity(query_vector, document_vector))
            for doc_id, document_vector in document_vectors.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )

    return rerank_mmr(
        query_vector,
        document_vectors,
        ranked_documents,
        top_k=k,
        lambda_weight=lambda_weight,
    )