from pathlib import Path
from types import SimpleNamespace

import streamlit as st

from bm25 import retrieve_bm25
from corpus_parser import parse_corpus
from hybrid_model import hybrid_search
from inverted_index import build_inverted_index
from mmr import retrieve_mmr
from phrase_search import phrase_search, proximity_search
from positional_index import build_positional_index
from preprocessing import preprocess
from vsm import calculate_document_vector, cosine_similarity, calculate_query_vector, retrieve


CORPUS_PATH = Path(__file__).parent / "data" / "corpus_100.txt"


@st.cache_resource
def load_search_data():
    """Parse the corpus and build the indexes once per app process."""
    documents = parse_corpus(str(CORPUS_PATH))
    for document in documents:
        document.tokens = preprocess(document.text)

    inverted_index = build_inverted_index(documents)
    positional_index = build_positional_index(documents)
    document_by_id = {document.docid: document for document in documents}
    document_vectors = {
        document.docid: calculate_document_vector(document)
        for document in documents
    }
    return documents, inverted_index, positional_index, document_by_id, document_vectors


def build_engine(documents, inverted_index, positional_index):
    """Adapt the project indexes to the shared model interface."""
    return SimpleNamespace(
        documents=documents,
        inverted_index=inverted_index,
        positional_index=positional_index,
    )


def run_model(model_name, query, engine, top_k=10):
    model_functions = {
        "VSM cosine": lambda: retrieve(query, engine.documents, engine.inverted_index, k=top_k),
        "BM25": lambda: retrieve_bm25(query, engine.documents, engine.inverted_index, k=top_k),
        "MMR diversity": lambda: retrieve_mmr(
            query, engine.documents, engine.inverted_index, k=top_k
        ),
        "Hybrid positional": lambda: hybrid_search(engine, query, top_k=top_k),
    }
    return model_functions[model_name]()


def cosine_scores(query, doc_ids, inverted_index, documents, document_vectors):
    """Score a supplied set of documents with the same VSM cosine function."""
    query_vector = calculate_query_vector(query, inverted_index, len(documents))
    scored = [
        (doc_id, cosine_similarity(query_vector, document_vectors[doc_id]))
        for doc_id in doc_ids
        if doc_id in document_vectors
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:10]


def result_rows(results, document_by_id, score_label="Score"):
    return [
        {
            "Rank": rank,
            "DocID": doc_id,
            "Title": document_by_id[doc_id].title,
            "Category": document_by_id[doc_id].category,
            score_label: round(score, 6),
        }
        for rank, (doc_id, score) in enumerate(results, start=1)
    ]


def positions_for_query(query, doc_id, positional_index):
    """Return the indexed positions for each processed query term in a doc."""
    terms = preprocess(query)
    return {
        term: positional_index[term][doc_id]
        for term in dict.fromkeys(terms)
        if term in positional_index and doc_id in positional_index[term]
    }


def main():
    st.set_page_config(page_title="Clothing Search Engine", page_icon="🔎", layout="wide")
    st.title("Clothing Search Engine")
    st.caption("Explore cosine-ranked retrieval and positional phrase search over the clothing corpus.")

    documents, inverted_index, positional_index, document_by_id, document_vectors = load_search_data()
    engine = build_engine(documents, inverted_index, positional_index)

    mode = st.radio(
        "Search mode",
        ["Free-text ranked search", "Phrase / proximity search"],
        horizontal=True,
    )
    query = st.text_input(
        "Search query",
        value="black cotton shirt" if mode == "Free-text ranked search" else "cotton crew neck",
        placeholder="Try: black cotton shirt",
    )

    if mode == "Free-text ranked search":
        model_names = ["VSM cosine", "BM25", "MMR diversity", "Hybrid positional"]
        selected_model = st.selectbox("Model for the ranked results", model_names)
        compare_models = st.checkbox("Show model comparison", value=True)
        st.write("Compare the implemented ranking models on the same clothing query.")
        if st.button("Search", type="primary") and query.strip():
            models_to_run = model_names if compare_models else [selected_model]
            model_results = {
                model_name: run_model(model_name, query, engine)
                for model_name in models_to_run
            }
            results = model_results[selected_model]
            if results:
                st.subheader("Top 10 results")
                st.dataframe(
                    result_rows(results, document_by_id),
                    hide_index=True,
                    use_container_width=True,
                )

                if compare_models:
                    baseline_ids = {
                        doc_id for doc_id, _ in model_results[selected_model]
                    }
                    comparison = []
                    for model_name, model_result in model_results.items():
                        result_ids = [doc_id for doc_id, _ in model_result]
                        comparison.append({
                            "Model": model_name,
                            "Top result": result_ids[0] if result_ids else "-",
                            "Top score": round(model_result[0][1], 6) if model_result else 0.0,
                            "Results returned": len(result_ids),
                            f"Overlap with {selected_model}": len(baseline_ids & set(result_ids)),
                        })
                    st.subheader("Model comparison")
                    st.dataframe(comparison, hide_index=True, use_container_width=True)
            else:
                st.info("No documents match the processed query terms.")
    else:
        search_type = st.radio("Positional operation", ["Exact phrase", "WITHIN/k proximity"], horizontal=True)
        if search_type == "WITHIN/k proximity":
            left, right = st.columns([3, 1])
            with left:
                first_term = st.text_input("First term", value="cotton")
            with right:
                distance = st.number_input("k", min_value=0, max_value=100, value=3, step=1)
            second_term = st.text_input("Second term", value="black")
        else:
            first_term = query
            second_term = ""
            distance = 0

        st.write("Matches are selected from the positional index, then ordered by cosine score.")
        if st.button("Search", type="primary"):
            if search_type == "Exact phrase":
                matched_ids = phrase_search(query, positional_index) if query.strip() else []
                evidence_query = query
            else:
                matched_ids = proximity_search(first_term, second_term, distance, positional_index)
                evidence_query = f"{first_term} {second_term}"

            results = cosine_scores(
                evidence_query,
                matched_ids,
                inverted_index,
                documents,
                document_vectors,
            )
            if results:
                st.subheader("Top positional matches")
                st.dataframe(result_rows(results, document_by_id), hide_index=True, use_container_width=True)

                evidence_doc_id = results[0][0]
                st.subheader(f"Positional evidence: {evidence_doc_id}")
                st.caption("Positions are zero-based offsets in the preprocessed token sequence.")
                st.json(positions_for_query(evidence_query, evidence_doc_id, positional_index))
            else:
                st.info("No positional matches found.")

    st.sidebar.metric("Documents", len(documents))
    st.sidebar.metric("Indexed terms", len(inverted_index))


if __name__ == "__main__":
    main()