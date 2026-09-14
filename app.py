from pathlib import Path
from types import SimpleNamespace

import plotly.graph_objects as go
import streamlit as st

from bm25 import retrieve_bm25
from corpus_parser import parse_corpus
from explainability import explain_score, positional_heatmap
from hybrid_model import DEFAULT_LAMBDA
from inverted_index import build_inverted_index
from mmr import retrieve_mmr
from phrase_search import phrase_search, proximity_search
from positional_index import build_positional_index
from preprocessing import preprocess
from vsm import calculate_document_vector, cosine_similarity, calculate_query_vector, retrieve
from hybrid_model import hybrid_search


CORPUS_PATH = Path(__file__).parent / "data" / "corpus_100.txt"

# Palette (validated for adjacent-pair CVD/contrast on both light and dark
# chart surfaces -- see the project's dataviz reference).
MODEL_COLORS = {
    "VSM cosine": "#2a78d6",       # categorical slot 1: blue
    "BM25": "#eb6834",             # slot 2: orange
    "MMR diversity": "#1baf7a",    # slot 3: aqua
    "Hybrid positional": "#eda100",  # slot 4: yellow
}
SEQUENTIAL_BLUE = "#2a78d6"
MUTED_INK = "#898781"      # axis/label ink, stable across light & dark
GRIDLINE = "rgba(137, 135, 129, 0.25)"
NO_MATCH_COLOR = "#e1e0d9"
TRANSPARENT = "rgba(0,0,0,0)"


def _base_layout(fig, height=360):
    """Apply shared chrome so every chart matches the app's look and stays
    legible on both light and dark Streamlit themes."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font=dict(color=MUTED_INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE)
    return fig


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


def render_score_chart(results, document_by_id, score_label="Score"):
    """Horizontal bar chart of the top results, ranked descending (top result
    on top). One sequential hue -- these bars encode a single magnitude."""
    ranked = list(reversed(results))
    labels = [f"{doc_id} · {document_by_id[doc_id].title[:30]}" for doc_id, _ in ranked]
    scores = [score for _, score in ranked]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=SEQUENTIAL_BLUE),
            hovertemplate=f"%{{y}}<br>{score_label}: %{{x:.4f}}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(automargin=True)
    return _base_layout(fig, height=max(220, 36 * len(labels)))


def render_model_comparison_chart(model_results, top_n=5):
    """Grouped bar chart comparing each model's top-N scores, one fixed
    categorical color per model so a re-run never repaints a model's color."""
    fig = go.Figure()
    for model_name, results in model_results.items():
        top = results[:top_n]
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=[doc_id for doc_id, _ in top],
                y=[score for _, score in top],
                marker=dict(color=MODEL_COLORS.get(model_name, MUTED_INK)),
                hovertemplate=f"{model_name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", legend_title_text="")
    fig.update_yaxes(title_text="Score")
    return _base_layout(fig, height=360)


def render_term_contribution_chart(explanation):
    """Horizontal bar chart of per-term score contributions for one document,
    biggest driver on top. Magnitude only, so a single sequential hue."""
    contributions = explanation["term_contributions"]
    if not contributions:
        return None

    ordered = list(reversed(contributions))
    fig = go.Figure(
        go.Bar(
            x=[c["contribution"] for c in ordered],
            y=[c["term"] for c in ordered],
            orientation="h",
            marker=dict(color=SEQUENTIAL_BLUE),
            hovertemplate="%{y}<br>contribution: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Contribution to score")
    fig.update_yaxes(automargin=True)
    return _base_layout(fig, height=max(180, 34 * len(ordered)))


def render_positional_heatmap(heatmap, width=25):
    """Grid heatmap of a document's token stream, colored by which query term
    (if any) matches at each position. Query terms are identity, so each gets
    a fixed categorical color, in first-seen order; no-match cells stay gray."""
    terms_in_order = []
    for _, term in heatmap:
        if term and term not in terms_in_order:
            terms_in_order.append(term)

    palette = list(MODEL_COLORS.values())
    term_color = {term: palette[i % len(palette)] for i, term in enumerate(terms_in_order)}

    n = len(heatmap)
    rows = -(-n // width)  # ceil
    grid_terms = [[None] * width for _ in range(rows)]
    grid_text = [[""] * width for _ in range(rows)]
    for position, term in heatmap:
        r, c = divmod(position, width)
        grid_terms[r][c] = term
        grid_text[r][c] = f"pos {position}: {term}" if term else f"pos {position}"

    # Map terms to numeric z-values so plotly can color discretely.
    term_to_z = {term: i + 1 for i, term in enumerate(terms_in_order)}
    z = [[term_to_z.get(t, 0) for t in row] for row in grid_terms]

    colorscale = [[0.0, NO_MATCH_COLOR]]
    n_terms = max(len(terms_in_order), 1)
    for i, term in enumerate(terms_in_order):
        lo = i / n_terms
        hi = (i + 1) / n_terms
        colorscale.append([lo if lo > 0 else 1e-6, term_color[term]])
        colorscale.append([hi, term_color[term]])
    if len(terms_in_order) == 0:
        colorscale = [[0.0, NO_MATCH_COLOR], [1.0, NO_MATCH_COLOR]]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=grid_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=0,
            zmax=n_terms,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    fig.update_yaxes(autorange="reversed", visible=False)
    fig.update_xaxes(visible=False)
    fig = _base_layout(fig, height=max(120, 28 * rows))

    # Legend as colored annotations, since a discretized heatmap has no
    # built-in per-term legend.
    legend_text = "  ".join(
        f"<span style='color:{term_color[t]}'>■</span> {t}" for t in terms_in_order
    )
    return fig, legend_text


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
                left, right = st.columns([3, 2])
                with left:
                    st.dataframe(
                        result_rows(results, document_by_id),
                        hide_index=True,
                        use_container_width=True,
                    )
                with right:
                    st.plotly_chart(
                        render_score_chart(results, document_by_id),
                        use_container_width=True,
                        config={"displayModeBar": False},
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
                    st.caption("Top-5 score per model, so magnitudes across ranking functions stay comparable at a glance.")
                    st.plotly_chart(
                        render_model_comparison_chart(model_results),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                st.subheader("Why did the top result rank first?")
                top_doc_id = results[0][0]
                explanation = explain_score(
                    query, document_by_id[top_doc_id], inverted_index, positional_index,
                    len(documents), lambda_weight=DEFAULT_LAMBDA,
                )
                exp_left, exp_right = st.columns([3, 2])
                with exp_left:
                    st.caption(f"Per-term score contribution for {top_doc_id} · {document_by_id[top_doc_id].title}")
                    term_chart = render_term_contribution_chart(explanation)
                    if term_chart:
                        st.plotly_chart(term_chart, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("No overlapping query terms to explain.")
                with exp_right:
                    st.caption("Where query terms land in the document's token stream")
                    heatmap = positional_heatmap(query, document_by_id[top_doc_id], positional_index)
                    fig, legend_text = render_positional_heatmap(heatmap)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    if legend_text:
                        st.markdown(legend_text, unsafe_allow_html=True)
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
                left, right = st.columns([3, 2])
                with left:
                    st.dataframe(result_rows(results, document_by_id), hide_index=True, use_container_width=True)
                with right:
                    st.plotly_chart(
                        render_score_chart(results, document_by_id),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                evidence_doc_id = results[0][0]
                st.subheader(f"Positional evidence: {evidence_doc_id}")
                st.caption("Each cell is one token in the document; colored cells are where a query term matches.")
                heatmap = positional_heatmap(evidence_query, document_by_id[evidence_doc_id], positional_index)
                fig, legend_text = render_positional_heatmap(heatmap)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                if legend_text:
                    st.markdown(legend_text, unsafe_allow_html=True)
                with st.expander("Raw positions (preprocessed token offsets)"):
                    st.json(positions_for_query(evidence_query, evidence_doc_id, positional_index))
            else:
                st.info("No positional matches found.")

    st.sidebar.metric("Documents", len(documents))
    st.sidebar.metric("Indexed terms", len(inverted_index))


if __name__ == "__main__":
    main()