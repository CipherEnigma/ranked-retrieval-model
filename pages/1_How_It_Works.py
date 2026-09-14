import streamlit as st

st.set_page_config(page_title="How It Works", page_icon="📖", layout="wide")

st.title("📖 How It Works")
st.caption("A plain-language walkthrough of every model in this search engine — built for explaining the demo.")

st.markdown(
    """
Every model here answers the same question — *"how well does this document
match the query?"* — but they each define "well" a little differently. This
page walks through each one in plain language, with the actual formula
underneath for anyone who wants the detail.
"""
)

st.divider()

# ------------------------------------------------------------------
# STEP 0 — Preprocessing
# ------------------------------------------------------------------
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
   form, e.g. `"running"` → `"run"`, `"shirts"` → `"shirt"`

**Why this matters for the demo:** if someone searches `"running shoes"` and
a product is described as `"runs great, ideal for jogging"`, stemming is why
`"running"` and `"runs"` can still connect. Everything downstream — every
index, every model — works on these cleaned-up word stems, never the raw
text.
"""
)

st.divider()

# ------------------------------------------------------------------
# STEP 1 — Indexes
# ------------------------------------------------------------------
st.header("Step 1 — The two indexes (built once, used by everything)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Inverted Index")
    st.markdown(
        """
        A lookup table: **word → which documents contain it.**

        ```
        "cotton" -> {D001, D002, D005, ...}
        "shirt"  -> {D002, D012, D042, ...}
        ```

        This is what lets us instantly find *candidate* documents for a query
        instead of scanning all 100 documents word by word every time.
        """
    )
with col2:
    st.subheader("Positional Index")
    st.markdown(
        """
        Same idea, but also remembers **where** each word sits inside the
        document:

        ```
        "cotton" -> { D002: [3, 17], D012: [0] }
        ```

        This is what powers phrase search, proximity search, and the
        "how close together are the query words" bonus in the hybrid model.
        """
    )

st.divider()

# ------------------------------------------------------------------
# STEP 2 — VSM
# ------------------------------------------------------------------
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

st.info("This scheme has a name in IR textbooks: **lnc.ltc** — documents use "
        "*l*og-tf, *n*o rarity adjustment, *c*osine-normalized; queries use "
        "*l*og-tf, *t*f-idf rarity adjustment, *c*osine-normalized.")

st.divider()

# ------------------------------------------------------------------
# STEP 3 — BM25
# ------------------------------------------------------------------
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
real-world text because of the saturation and length-normalization —
it's the model most production search engines (before neural search)
actually used.
"""
)

st.divider()

# ------------------------------------------------------------------
# STEP 4 — MMR
# ------------------------------------------------------------------
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

**Demo talking point:** this is the fix for the "5 identical shirts in 5
colors" problem — see the *Top 10 results* screenshot where checked cotton
shirts in White/Blue/Black/Maroon/Olive all tied for the same score. MMR
notices they're near-duplicates of each other and starts swapping some of
those slots for something more varied, even if it scores slightly lower on
relevance alone.
"""
)

st.divider()

# ------------------------------------------------------------------
# STEP 5 — Hybrid
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# STEP 6 — Phrase / proximity search
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Cheat sheet
# ------------------------------------------------------------------
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

st.caption("Tip: run the same query across all four models with 'Show model comparison' turned on — "
           "the grouped bar chart on the Search page makes these differences visible at a glance.")
