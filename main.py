from corpus_parser import parse_corpus
from preprocessing import preprocess


# --------------------------------------------------
# 1. LOAD THE CORPUS
# --------------------------------------------------

file_path = "data/corpus_100.txt"

documents = parse_corpus(file_path)


# --------------------------------------------------
# 2. CHECK NUMBER OF DOCUMENTS
# --------------------------------------------------

print("Number of documents:", len(documents))


# --------------------------------------------------
# 3. PREPROCESS EVERY DOCUMENT
# --------------------------------------------------

for document in documents:
    document.tokens = preprocess(document.text)


# --------------------------------------------------
# 4. DISPLAY FIRST 5 DOCUMENTS
# --------------------------------------------------

for document in documents[:5]:

    print("\n" + "=" * 80)

    print("DOCID:")
    print(document.docid)

    print("\nCATEGORY:")
    print(document.category)

    print("\nTITLE:")
    print(document.title)

    print("\nORIGINAL TEXT:")
    print(document.text)

    print("\nPROCESSED TOKENS:")
    print(document.tokens)