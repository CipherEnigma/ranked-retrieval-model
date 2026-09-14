# Ranked Retrieval Model — Clothing Search Engine

A ranked-retrieval search engine built from scratch over a 100-document clothing
corpus, implementing classic IR data structures (inverted index, positional
index) and four ranking models (VSM cosine, BM25, MMR diversity, positional
hybrid), served through a Streamlit UI.

Built for CSD358 (Information Retrieval) Assignment 1.

## Live demo

Deployed on Streamlit Community Cloud: _add your app URL here after deploying_.

## Features

- **Corpus parsing** — tagged `<DOC><DOCID><CATEGORY><TITLE><TEXT>` format ([corpus_parser.py](corpus_parser.py))
- **Preprocessing** — lowercase → strip punctuation → tokenize → NLTK stopword removal → Porter stemming, identical for documents and queries ([preprocessing.py](preprocessing.py))
- **Inverted index** — `term -> {doc_ids}`, dumped to `data/dictionary.txt` ([inverted_index.py](inverted_index.py))
- **Positional index** — `term -> {doc_id: [positions]}`, dumped to `data/positional_index.txt` ([positional_index.py](positional_index.py))
- **Ranking models**
  - **VSM cosine** — `lnc.ltc` weighting (log-TF, cosine-normalized documents; log-TF × IDF, cosine-normalized queries) ([vsm.py](vsm.py))
  - **BM25** — Robertson-Sparck Jones formula, `k1=1.5, b=0.75` ([bm25.py](bm25.py))
  - **MMR diversity** — re-ranks VSM results to balance relevance against redundancy ([mmr.py](mmr.py))
  - **Hybrid positional** — VSM cosine score plus a proximity boost from the positional index (tighter query-term clustering scores higher) ([hybrid_model.py](hybrid_model.py))
- **Phrase & proximity search** — exact phrase matching and `WITHIN/k` proximity queries via the positional index ([phrase_search.py](phrase_search.py))
- **Explainability** — per-term score breakdown, counterfactual "why did A outrank B" comparisons, and a positional heatmap ([explainability.py](explainability.py))
- **Robustness probing** — measures ranking stability under typos, word shuffling, and stopword injection ([robustness_probe.py](robustness_probe.py))
- **Streamlit UI** — free-text search with a model picker and side-by-side model comparison, plus a phrase/proximity mode with positional evidence ([app.py](app.py))

## Project structure

```
├── app.py                  # Streamlit UI
├── corpus_parser.py        # Parses the tagged corpus into Document objects
├── document.py              # Document data class
├── preprocessing.py        # Shared tokenization/stemming pipeline
├── inverted_index.py       # Inverted index + dictionary.txt dump
├── positional_index.py     # Positional index + positional_index.txt dump
├── vsm.py                  # Vector space model (lnc.ltc cosine ranking)
├── bm25.py                 # BM25 ranking
├── mmr.py                  # MMR diversity re-ranking
├── hybrid_model.py         # Positional-aware hybrid ranking
├── phrase_search.py        # Exact phrase + WITHIN/k proximity search
├── explainability.py       # Score explanations, counterfactuals, heatmaps
├── facet_parser.py         # Query facet parsing (color, fabric, category, ...)
├── robustness_probe.py     # Ranking stability under query perturbation
├── data/
│   ├── corpus_100.txt      # Source corpus (100 clothing documents)
│   ├── dictionary.txt      # Inverted index dump
│   └── positional_index.txt# Positional index dump
├── test_track_a.py         # Tests: indexing/ranking track
├── test_track_b.py         # Tests: positional/query-side track
└── requirements.txt
```

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

The first run downloads the NLTK `stopwords` corpus automatically if it isn't
already present.

## Running

**Streamlit app:**

```bash
streamlit run app.py
```

**Command-line demos** (each module has a runnable example under `if __name__ == "__main__":`):

```bash
python vsm.py      # VSM ranking demo
python bm25.py      # BM25 ranking demo
python inverted_index.py   # Rebuild dictionary.txt
```

**Tests:**

```bash
python -m pytest test_track_a.py test_track_b.py -v
```

## Deploying on Streamlit Community Cloud

1. Push `requirements.txt` and the `data/` folder (already tracked in this repo) to `main`.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at this repo, branch `main`, main file `app.py`.
3. Streamlit installs `requirements.txt` (`streamlit`, `nltk`) and launches; `preprocessing.py` downloads the NLTK stopwords corpus on first import if needed.

## Design notes

- **Stop-word policy**: NLTK's standard ~179-word English list, used as-is with no additions or removals — sufficient for a clothing-product corpus where these words carry no discriminative signal.
- **Weighting scheme**: documents use `lnc` (log-TF, no IDF, cosine-normalized); queries use `ltc` (log-TF, IDF, cosine-normalized). Since both are pre-normalized, cosine similarity reduces to a dot product.
- **Tie-breaking**: all models sort by descending score, then ascending `doc_id`, for deterministic output.
- **Model contract**: every ranking function returns `[(doc_id, score), ...]` sorted per the tie-break rule above, so models are interchangeable in the UI and evaluation harness.
