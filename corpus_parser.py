import re
from document import Document

def parse_corpus(file_path):
    """
    Parse the clothing corpus and return a list of Document objects.

    Corpus format:
        <DOC>
        <DOCID>...</DOCID>
        <CATEGORY>...</CATEGORY>
        <TITLE>...</TITLE>
        <TEXT>...</TEXT>
        </DOC>
    """

    documents = []

    with open(file_path, "r", encoding="utf-8") as file:
        corpus = file.read()

    # Extract each <DOC>...</DOC> block
    doc_blocks = re.findall(
        r"<DOC>(.*?)</DOC>",
        corpus,
        flags=re.DOTALL
    )

    for block in doc_blocks:

        docid_match = re.search(
            r"<DOCID>(.*?)</DOCID>",
            block,
            flags=re.DOTALL
        )

        category_match = re.search(
            r"<CATEGORY>(.*?)</CATEGORY>",
            block,
            flags=re.DOTALL
        )

        title_match = re.search(
            r"<TITLE>(.*?)</TITLE>",
            block,
            flags=re.DOTALL
        )

        text_match = re.search(
            r"<TEXT>(.*?)</TEXT>",
            block,
            flags=re.DOTALL
        )

        # Skip malformed documents
        if not all([docid_match, category_match, title_match, text_match]):
            continue

        docid = docid_match.group(1).strip()
        category = category_match.group(1).strip()
        title = title_match.group(1).strip()
        text = text_match.group(1).strip()

        document = Document(
            docid=docid,
            category=category,
            title=title,
            text=text
        )

        documents.append(document)

    return documents
