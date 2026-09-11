def build_positional_index(documents):
    """
    Build the positional index from a list of Document objects whose
    `.tokens` field has already been populated by preprocess().

    Returns:
        positional_index[term] = {doc_id: [positions]}

    Positions are indices into the *preprocessed* token list (post
    stopword-removal, post-stemming), so they line up with how a
    preprocessed query is tokenized.
    """

    positional_index = {}

    for document in documents:
        for position, term in enumerate(document.tokens):

            if term not in positional_index:
                positional_index[term] = {}

            if document.docid not in positional_index[term]:
                positional_index[term][document.docid] = []

            positional_index[term][document.docid].append(position)

    return positional_index


def dump_positional_index(positional_index, file_path):
    """
    Write the positional index to disk in a human-readable format:

        term -> df
            doc_id: pos1,pos2,...
            doc_id: pos1,pos2,...

    Sorted alphabetically by term, and by doc_id within each term, so the
    output is deterministic across runs.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        for term in sorted(positional_index.keys()):
            postings = positional_index[term]
            df = len(postings)

            file.write(f"{term} -> {df}\n")

            for doc_id in sorted(postings.keys()):
                positions = ",".join(str(p) for p in postings[doc_id])
                file.write(f"\t{doc_id}: {positions}\n")
