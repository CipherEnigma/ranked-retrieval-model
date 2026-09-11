from corpus_parser import parse_corpus
from preprocessing import preprocess


def build_inverted_index(documents):
    """
    Build an inverted index from a list of Document objects.

    Structure:

        term -> set of document IDs

    Example:

        {
            "cotton": {"D001", "D002", "D005"},
            "shirt": {"D002", "D012"}
        }
    """

    inverted_index = {}

    for document in documents:

        # Preprocess the document text
        tokens = preprocess(document.text)

        # Store the processed tokens in the Document object
        document.tokens = tokens

        # Add each term to the inverted index
        for token in tokens:

            if token not in inverted_index:
                inverted_index[token] = set()

            inverted_index[token].add(document.docid)

    return inverted_index


def write_dictionary(inverted_index, output_file):
    """
    Write the inverted index to dictionary.txt.

    Each line contains:

        term -> document IDs
    """

    with open(output_file, "w", encoding="utf-8") as file:

        # Sort terms alphabetically
        for term in sorted(inverted_index):

            doc_ids = sorted(inverted_index[term])

            file.write(
                f"{term} -> {', '.join(doc_ids)}\n"
            )


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. LOAD CORPUS
    # --------------------------------------------------

    documents = parse_corpus("data/corpus_100.txt")

    print("Number of documents:", len(documents))


    # --------------------------------------------------
    # 2. BUILD INVERTED INDEX
    # --------------------------------------------------

    inverted_index = build_inverted_index(documents)

    print("Number of unique terms:", len(inverted_index))


    # --------------------------------------------------
    # 3. WRITE DICTIONARY
    # --------------------------------------------------

    write_dictionary(
        inverted_index,
        "dictionary.txt"
    )

    print("dictionary.txt created successfully.")

