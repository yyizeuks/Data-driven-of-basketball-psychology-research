# Knowledge Structure and Trend Dynamics of Basketball Psychology — analysis pipeline and derived data

Companion repository for the systematic mapping review *"Knowledge Structure and Trend Dynamics of Basketball Psychology: A Systematic Mapping Review Using Multi-Database Topic Modeling"* (Jea Woog Lee & Doug Hyun Han, Frontiers in Psychology, under review).

It contains everything needed to re-execute the analysis: the complete, version-pinned code; the construct-pattern, basketball-term and stop-word lists that define the corpus; the derived document–term matrix and document-level metadata for all 2,358 included articles; all model outputs, validation results, trend statistics and sensitivity analyses; and the figure code.

## What can and cannot be shared
| Class | Content | Status |
|---|---|---|
| Raw records | Web of Science Core Collection export | Licensed — **not redistributed**; place your own export in the repository root (see `README_raw.txt`) |
| Raw records | PubMed/MEDLINE and Crossref records | Openly retrievable; retrieval specification in the article's Supplementary S1 |
| Derived data | `document_term_matrix.mtx` + `vocabulary.txt` (bag-of-noun counts, 2,358 × 1,342) | Shared — every model in the article can be re-estimated from these without the licensed abstracts |
| Derived data | `document_metadata.csv` (title, year, source stratum, WoS research areas, dominant topic, topic proportions θ, construct triggers) | Shared |
| Lists | `construct_patterns.txt`, `basketball_terms.txt`, `stopwords_domain.txt` | Shared |
| Outputs | `topic_terms_top20.csv`, `topic_word_phi.csv`, `sweep.json`, `val.json`, `trend_stats.csv`, `construct_hits.csv`, `nmf_topics.csv`, `sens_results.json`, `sens2_results.json`, `openalex_counts.json`, `prisma_counts_clean.json` | Shared |
| Code | numbered scripts `00a`–`11` and `nlp_defs.py` (flat layout, repository root) | MIT license |

## Environment
Python 3.11/3.12 with the pinned versions in `requirements.txt` (`pip install -r requirements.txt`). NLTK resources: `python -c "import nltk; [nltk.download(p) for p in ('stopwords','wordnet','averaged_perceptron_tagger_eng','punkt')]"`.
The reference model was estimated under Python 3.12.3 (NumPy 2.4.4, SciPy 1.17.1, NLTK 3.10.0, Gensim 4.4.0) and re-executed under Python 3.11.15 with identical library versions; all reported statistics reproduced exactly.

## Re-executing the full pipeline (requires the raw exports in the repository root)
Run from the repository root, in order:
```
python 00a_wos_stratum_screen.py        # original WoS single-database screen  -> corpus_filtered.pkl
python 00b_wos_stratum_nlp.py           # original WoS NLP step               -> corpus_final.pkl (WoS stratum, n = 1,678)
python 01_build_corpus.py               # pooled construct-anchored screening, dedup, preprocessing -> 2,358 docs, V = 1,342
python 02_fit_reference_model.py        # k = 8, seed 42, passes 25, iterations 600 -> C_v = 0.3819
python 03_model_selection_sweep.py      # k = 4..18 dual-metric sweep (Table 1)
python 04_validation_battery.py         # seeds, split-half, trend replication, criterion validity (Figure 3)
python 05_trend_statistics.py           # OLS + 95% CI, Newey–West, WLS, diagnostics, Mann–Kendall/Sen, BH-FDR (Table 3, Figure 9)
python 06_sensitivity_k.py              # k = 6, 7, 9, 10 under final settings, Hungarian-matched
python 07a_openalex_screen.py           # (optional) screen an OpenAlex export
python 07_sensitivity_nmf_strict_openalex.py   # NMF, strict-construct, OpenAlex expansion
python 10_sensitivity_single_pass_screening.py # single-pass screening sensitivity (2,362 docs)
python 08_figures_3_9.py; python 09_figure_1_prisma.py
python 11_gap_search_pubmed.py                # reproducible gap search (Supplementary S0)
```
Scripts that fit many models are resumable (state files) and accept an optional time budget in seconds as the first argument.

## Re-estimating models without the raw abstracts
Load `data/document_term_matrix.mtx` (Matrix Market; rows = `document_metadata.csv` order, columns = `data/vocabulary.txt`), convert rows to Gensim bag-of-words `(term_id, count)` lists, and run `LdaModel(num_topics=8, random_state=42, passes=25, iterations=600, alpha='auto', eta='auto', chunksize=200)`. Note that Gensim's `Dictionary` object is needed only for `id2word`; build it from `vocabulary.txt`.

## Key reproduced quantities
Corpus 2,358 documents (WoS 1,178; PubMed/Crossref 1,181), vocabulary 1,342, final C_v 0.3819; PRISMA flow 8,928 → 5,290 → 2,914 → 2,359 → 2,358; seed stability Jaccard@20 M = .18 (SD = .03); criterion validity χ²(56) = 190.1, Cramér's V = .152; robust trends after BH-FDR: T1 +1.56 pp/decade (q = .022), T8 −1.04 pp/decade (q = .022).

## Search specification
All database searches used the query *basketball AND psychology* (February–March 2026): Web of Science records exported from the database interface; PubMed, Crossref and OpenAlex records retrieved with a custom Python script. Screening rules and counts: Supplementary S1 of the article and `prisma_counts_clean.json`.

## Citation
See `CITATION.cff`. Archived releases receive a Zenodo DOI. Repository: https://github.com/yyizeuks/Data-driven-of-basketball-psychology-research
