# PLAN.md — Clothing Search Engine (CSD358 IR Assignment-1)

**Team of two.** Built from scratch. Scope: full base (Parts A–E) **plus** novelty
extensions. **Deadline: Sept 12, 11:59 pm** — ~1 day of work, so the base is
locked at a hard checkpoint first, then extensions are stacked on top.

> **Non-negotiable line:** hit **Checkpoint CP1 (base A–E)** before touching any
> extension. Everything after CP1 is additive and independently droppable, so
> partial progress never breaks the submission.

---

## Roles
- **Person A — Indexing core + ranking + evaluation** (the "scoring" half)
- **Person B — Positional + query-side + interface** (the "matching & UX" half)

_Assign names:_ Person A = `__________`  ·  Person B = `__________`

---

## The idea that prevents wasted time
The only thing that blocks one person on the other is **shared data structures**.
So we freeze **one interface contract** in Phase 0. After that, both people code
against the contract in parallel and never wait — if the real version of
something isn't ready, you develop against a stub/mock and swap it in at
integration.

### Frozen contract (write as stubs first; both import it)
- `Document` fields: `doc_id, category, title, text, tokens, tf, num_tokens, length`
- `preprocess(text) -> [tokens]` — **identical** for documents and queries
- Engine attributes both sides rely on:
  - `inverted_index[term] = {doc_id: tf}`
  - `positional_index[term] = {doc_id: [positions]}`
  - `df[term]`, `N`, `doc_by_id`
  - per-doc `length` (cosine norm) and `num_tokens`
- **Uniform model signature:** `model(engine, query, top_k) -> [(doc_id, score)]`
  sorted by decreasing score, ties broken by **increasing doc_id**

---

## Phase 0 — Joint foundation + contract  *(do together, ~2 hrs, BLOCKING)*
The only truly blocking work. Finish before splitting off.

| Sub-task | Owner |
|---|---|
| Corpus parser (`DOCID/CATEGORY/TITLE/TEXT` → Document objects) | A |
| Preprocessing pipeline: lowercase → strip punctuation → tokenize → stopword removal | A |
| Porter stemmer + documented stop-word policy | B |
| **Agree & freeze the contract above** (stubs committed) | both |

**Exit condition:** the contract stubs exist and both can import them. From here,
zero blocking.

---

## Track A — Person A  *(independent after Phase 0)*
| Task | Part | Depends on | Est. |
|---|---|---|---|
| Inverted index + `dictionary.txt` dump | **A** | contract | 1.5h |
| VSM `lnc.ltc`: doc weight `1+log10(tf)` (no idf), query weight `(1+log10(tf))·log10(N/df)`, cosine, top-10, tie-break | **B** | inverted index | 2h |
| BM25 + tf-idf (`ltc.ltc`) models | ext | model signature | 1.5h |
| Evaluation harness: P@k, R@k, MAP, nDCG@k, MRR + model comparison table + chart | ext | ≥1 model | 2.5h |
| MMR diversity re-ranking (λ tunable) | ext | VSM vectors | 1.5h |
| Rocchio pseudo-relevance feedback (α, β, top-k) | ext | VSM vectors | 1.5h |

## Track B — Person B  *(independent after Phase 0)*
| Task | Part | Depends on | Est. |
|---|---|---|---|
| Positional index + `positional_index.txt` dump | **C** | contract | 1.5h |
| Exact phrase search + ordered `WITHIN/k` proximity | **C** | positional index | 2h |
| Positional-aware hybrid model (proximity boost on VSM base) | ext | positional index + model signature | 1.5h |
| Query understanding / facet parser (color, fabric, gender, category, occasion, size) | ext | preprocess + vocab | 1.5h |
| Spelling correction + wildcard via k-gram index | ext | surface vocab (own) | 2h |
| Explainability + counterfactual "why not?" + positional heatmap | ext | scoring interface | 1.5h |
| Robustness probing: typo / word-shuffle / stopword-inject → overlap@10, Kendall-τ | ext | model signature | 1.5h |

---

## Dependency rules (how nobody sits idle)
1. **Contract-first, stub-then-fill.** B's positional-hybrid and robustness need
   "a base VSM score" — B codes against a stub scorer and swaps in A's real
   `lnc.ltc` at integration. B never waits for A.
2. **A's evaluation harness** develops against A's own three models first
   (`lnc.ltc`, `bm25`, `tfidf`); B's positional model plugs in later as one
   clean call.
3. **Part A/B (Person A) and Part C (Person B) share nothing but the contract** —
   true parallel from hour 2.
4. **Finished early? Pull the next checkpoint forward**, don't idle.

---

## Integration checkpoints (short syncs, both present)

### CP1 — BASE LOCK *(mandatory; hit this first)*
- Inverted index + positional index both built
- `lnc.ltc` VSM working; exact phrase + `WITHIN/k` proximity working
- Both index dumps written (`dictionary.txt`, `positional_index.txt`)
- **Part E test battery** — split the queries:
  - **A:** ≥10 free-text queries + ≥1 query with an out-of-vocabulary term
  - **B:** ≥5 exact phrase + ≥3 proximity (different k) + the **2 required
    analyses** where positional info changes the result set/order
- ✅ **After CP1 a complete, submittable assignment exists.**

### CP2 — Models + evaluation
- Plug all four models into the harness → comparison table + chart

### CP3 — Query-side + UX
- Wire facets, spell/wildcard, explanation, heatmap into the app

### CP4 — Interface + report + ZIP
- Web UI (B leads shell, A wires model/eval endpoints)
- Screenshots (Part D + representative query results)
- README (design decisions, stop-word policy justification, weighting scheme)
- Short report (methodology, model comparison, MMR before/after, robustness)
- **Single ZIP** with all required files

---

## One-day timeline
| Window | Work |
|---|---|
| Hr 0–2 | Phase 0 (joint) → contract frozen |
| Hr 2–7 | Tracks A & B in parallel → **CP1 base lock by hr 7** |
| Hr 7–11 | Extensions in parallel → CP2 + CP3 |
| Final block | CP4 — UI, screenshots, report, ZIP |

---

## Deliverables checklist (from the assignment)
- [ ] Source code with comments/documentation
- [ ] Dictionary / inverted-index output
- [ ] Positional-index output
- [ ] Screenshots of the application + representative query results
- [ ] Single ZIP containing all required files
- [ ] (Extensions) evaluation table + chart, MMR demo, robustness table, report

## Mandatory tests (Part E — do not hard-code expected doc IDs)
- [ ] ≥10 free-text queries
- [ ] ≥5 exact phrase queries
- [ ] ≥3 proximity queries with different k
- [ ] ≥1 query containing a term not in the corpus
- [ ] Report top-10 and explain ≥2 cases where positional info changes result set/order
