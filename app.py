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


def render_search_tab(documents, inverted_index, positional_index, document_by_id, document_vectors, engine):
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


def render_how_it_works():
    st.markdown(
        """
Every model here answers the same question — *"how well does this document
match the query?"* — but each one defines "well" a little differently. This
tab walks through each one in plain language, with the actual formula
underneath for anyone who wants the detail.
"""
    )

    st.divider()

    st.header("Step 0 — Preprocessing (the shared groundwork)")
    st.markdown(
        """
Before any scoring happens, every document **and** every query goes through
the exact same cleanup pipeline, so they end up speaking the same language:

1. **Lowercase** everything
2. **Strip punctuation**
3. **Split into words** (tokens)
4. **Remove stopwords** — throw away words like "the", "is", "and" that don't
   help tell products apart
5. **Stem** each word with the Porter stemmer — chop words down to a root
   form, e.g. `"running"` -> `"run"`, `"shirts"` -> `"shirt"`

**Why this matters for the demo:** if someone searches `"running shoes"` and
a product is described as `"runs great, ideal for jogging"`, stemming is why
`"running"` and `"runs"` can still connect. Everything downstream — every
index, every model — works on these cleaned-up word stems, never the raw
text.
"""
    )

    st.divider()

    st.header("Step 1 — The two indexes (built once, used by everything)")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inverted Index")
        st.markdown(
            """
            A lookup table: **word -> which documents contain it.**

            ```
            "cotton" -> {D001, D002, D005, ...}
            "shirt"  -> {D002, D012, D042, ...}
            ```

            This is what lets us instantly find *candidate* documents for a
            query instead of scanning all 100 documents word by word every
            time.
            """
        )
    with col2:
        st.subheader("Positional Index")
        st.markdown(
            """
            Same idea, but also remembers **where** each word sits inside
            the document:

            ```
            "cotton" -> { D002: [3, 17], D012: [0] }
            ```

            This is what powers phrase search, proximity search, and the
            "how close together are the query words" bonus in the hybrid
            model.
            """
        )

    st.divider()

    st.header("Model 1 — Vector Space Model (VSM Cosine)")
    st.markdown(
        """
**Plain language:** turn the document and the query into two lists of
numbers (a "vector"), one number per word, then measure how much they point
in the same direction. The more important words they share, the more
similar the direction.

- A word that appears **more often** in a document → weighted higher, but
  with diminishing returns (a word appearing 10 times isn't 10x as
  important as it appearing once — we take a log so it levels off).
- A word that is **rare across the whole corpus** (appears in few
  documents) → weighted higher in the *query*, because rare words are more
  distinguishing. Common words like "shirt" barely move the needle.
"""
    )
    st.markdown("**Document weight** (per word, no rarity adjustment):")
    st.latex(r"\text{weight} = 1 + \log_{10}(tf)")
    st.markdown("**Query weight** (per word, rarity-adjusted):")
    st.latex(r"\text{weight} = \big(1 + \log_{10}(tf)\big) \times \log_{10}\!\left(\frac{N}{df}\right)")
    st.markdown(
        """
Where `tf` = how many times the word appears, `N` = total number of
documents, `df` = how many documents contain that word.

Both vectors are then scaled down to length 1 ("normalized"), so the final
comparison is just a dot product:
"""
    )
    st.latex(r"\text{score} = \sum_{\text{shared words}} (\text{query weight}) \times (\text{doc weight})")
    st.info(
        "This scheme has a name in IR textbooks: **lnc.ltc** — documents use "
        "*l*og-tf, *n*o rarity adjustment, *c*osine-normalized; queries use "
        "*l*og-tf, *t*f-idf rarity adjustment, *c*osine-normalized."
    )

    st.divider()

    st.header("Model 2 — BM25")
    st.markdown(
        """
**Plain language:** BM25 is VSM's more sophisticated cousin. It fixes two
things VSM does poorly:

1. **Diminishing returns cap harder.** In VSM, a word repeated many times
   keeps adding (a little) score forever. BM25 makes that bonus *saturate* —
   after a point, repeating the word more just stops mattering.
2. **Length fairness.** A long product description will naturally contain
   more words, which VSM can accidentally reward. BM25 normalizes against
   the *average* document length in the corpus, so long documents don't win
   just by being long.
"""
    )
    st.latex(
        r"\text{score} = \sum_{t \in q} \text{IDF}(t) \cdot "
        r"\frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot \left(1 - b + b \cdot \dfrac{|d|}{\text{avgdl}}\right)}"
    )
    st.markdown(
        """
- `tf` = term frequency in the document, `|d|` = document length,
  `avgdl` = average document length across the corpus
- `k1 = 1.5` controls how quickly repeated terms saturate
- `b = 0.75` controls how strongly document length is penalized
- IDF here uses a slightly different (smoothed) formula than VSM's, but
  same idea: rare words count more.

**Demo talking point:** BM25 tends to give more "sensible" rankings on
real-world text because of the saturation and length-normalization — it's
the model most production search engines (before neural search) actually
used.
"""
    )

    st.divider()

    st.header("Model 3 — MMR (Maximal Marginal Relevance)")
    st.markdown(
        """
**Plain language:** MMR isn't a new way of *scoring* a document — it's a way
of **re-ordering** an already-ranked list to avoid showing 10 near-identical
results.

It builds the final list one slot at a time. For each remaining candidate,
it asks: *"how relevant is this, minus how similar is it to what I've
already picked?"* — then picks whichever document scores best on that
trade-off, and repeats.
"""
    )
    st.latex(
        r"\text{MMR score} = \lambda \cdot \text{relevance}(d) - "
        r"(1-\lambda) \cdot \max_{d_j \in \text{already picked}} \text{similarity}(d, d_j)"
    )
    st.markdown(
        """
- `relevance(d)` = the document's plain VSM cosine score against the query
- `similarity(d, dj)` = cosine similarity between two documents (how alike
  they are to each other, not to the query)
- `λ` (lambda) controls the trade-off: higher λ → prioritize relevance,
  lower λ → prioritize variety. This app uses `λ = 0.7`.

**Demo talking point:** this is the fix for a "5 identical shirts in 5
colors" problem — when several color variants of the same product tie for
the same score, MMR notices they're near-duplicates of each other and
starts swapping some of those slots for something more varied, even if it
scores slightly lower on relevance alone.
"""
    )

    st.divider()

    st.header("Model 4 — Hybrid Positional")
    st.markdown(
        """
**Plain language:** start from the normal VSM cosine score, then add a small
bonus if the query words appear **close together** in the document — not
scattered across unrelated sentences.

The bonus only kicks in if **every** query word appears somewhere in the
document. Then we find the tightest window of text that contains at least
one occurrence of each word (like finding the shortest sentence that
mentions "black", "cotton", *and* "shirt" all at once), and the tighter that
window, the bigger the bonus.
"""
    )
    st.latex(r"\text{proximity boost} = \frac{1}{1 + \text{span}}")
    st.latex(r"\text{final score} = \text{VSM cosine score} + \lambda \cdot \text{proximity boost}")
    st.markdown(
        """
- `span` = the width (in word positions) of the tightest window covering
  every distinct query word. Smaller span → bigger boost. If the words sit
  right next to each other, span is close to 0 and the boost is close to 1.
- `λ = 0.2` here, so this is a *tie-breaking nudge*, not a dominant factor —
  it mainly matters when VSM scores are close.

**Demo talking point:** this is where the *positional heatmap* visualization
earns its keep. Two documents can have identical VSM scores (same words,
same frequency) but very different heatmaps — one has "black", "cotton",
"shirt" sitting side by side describing one product, the other has them
scattered across separate sentences. Hybrid ranks the tightly-clustered one
higher, and the heatmap is the visual proof of *why*.
"""
    )

    st.divider()

    st.header("Bonus — Exact Phrase & Proximity Search")
    st.markdown(
        """
These aren't scoring models — they're **filters** that run before scoring.

- **Exact phrase**: only keeps documents where the query words appear
  *consecutively, in that exact order* — checked using the positional index
  (word at position `p`, next word at `p+1`, and so on).
- **`WITHIN/k` proximity**: looser — keeps documents where two words appear
  *anywhere within k positions of each other*, in any order.

Whatever survives the filter then gets ranked by plain VSM cosine, so you
still get a meaningful order rather than an unsorted list of matches.
"""
    )

    st.divider()

    st.header("🎯 One-line cheat sheet for the demo")
    st.table(
        {
            "Model": ["VSM Cosine", "BM25", "MMR Diversity", "Hybrid Positional"],
            "In one line": [
                "Classic word-overlap scoring, rewards rare shared words",
                "VSM's smarter cousin — caps repeated-word bonus, corrects for document length",
                "Re-orders VSM's list to avoid near-duplicate results",
                "VSM plus a bonus when query words sit close together in the text",
            ],
            "Best query to demo": [
                "Any free-text query",
                "Same query as VSM, to contrast the ranking",
                "A broad query with near-duplicate products (e.g. \"cotton shirt\")",
                "A 3+ word query where word order/closeness matters",
            ],
        }
    )
    st.caption(
        "Tip: run the same query across all four models with 'Show model comparison' turned on — "
        "the grouped bar chart on the Search tab makes these differences visible at a glance."
    )


def main():
    st.set_page_config(page_title="Clothing Search Engine", page_icon="🔎", layout="wide")
    st.title("Clothing Search Engine")
    st.caption("Explore cosine-ranked retrieval and positional phrase search over the clothing corpus.")

    documents, inverted_index, positional_index, document_by_id, document_vectors = load_search_data()
    engine = build_engine(documents, inverted_index, positional_index)

    search_tab, how_it_works_tab = st.tabs(["🔎 Search", "📖 How It Works"])

    with search_tab:
        render_search_tab(documents, inverted_index, positional_index, document_by_id, document_vectors, engine)

    with how_it_works_tab:
        render_how_it_works()

    st.sidebar.metric("Documents", len(documents))
    st.sidebar.metric("Indexed terms", len(inverted_index))


if __name__ == "__main__":
    main()