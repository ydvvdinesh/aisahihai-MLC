# 📚 Intern Onboarding & Codebase Architecture Guide

```text
Raw Data ➔ [1. Preprocessing] ➔ [2. Hybrid Blocking] ➔ [3. Features] ➔ [4. Modeling] ➔ [5. Post-Processing] ➔ Final TSVs

Phase	File	Primary Responsibility & Method
1. Normalization	

normalize.py
Domain NLP Engine: Script-agnostic transliteration across 9 Indic scripts via Unicode properties; phonetic consonant skeleton extraction (skeleton); legal suffix stripping; address canonicalization (normalize_address).


preprocess.py
Batch Parallel Runner: Multiprocess parser converting raw TSVs into normalized binary Parquet tables (<split>_s1.parquet and <split>_q.parquet).
2. Blocking	

blocking.py
Sparse Inverted Index: Runs reverse top-$K$ candidate retrieval (since queries map to $\le 1$ entity) using compound skeleton keys, acronyms, and IDF-weighted sparse matrix dot products.


export_native.py
Native Script Filter: Extracts query records containing Brahmic/Indic scripts ([ऀ-ൿ]) for GPU embedding.


emb_knn.py
Dense Semantic Retrieval: Encodes native names using frozen LaBSE (sentence-transformers/LaBSE) on GPU and retrieves top-100 nearest Source-1 candidates.


emb_candidates.py
Address-Aware Re-ranking: Re-ranks LaBSE top-100 candidates with address overlap ($\cos + 0.5 \times Jaccard(nums) + 0.5 \times Jaccard(words)$), appending up to 2 extra candidates to sparse blocking.
3. Features	

features.py
Pairwise Feature Engineering: 30+ RapidFuzz string metrics (Levenshtein, token sort, token set, Jaro-Winkler), house-number edit types, and candidate competition context features (blk_margin, blk_gap12).
4. Modeling	

common.py
Core Utilities & Metrics: Ground-truth index mapping and exact competition Macro $F_{0.5}$ calculation (

fbeta_from_assign
).


pipeline.py
Stage-1 GBDT Baseline: 2-fold cross-validation grouped strictly by source1_entity_id; LightGBM GBDT; 1-to-1 disjoint constraint assignment (

assign_from_prob
).


maximumoutput.py
Champion Pipeline: Stage-1 + LaBSE + Robust Stage-2 variant with self-support features stripped to prevent sibling distractor collapse.


cross_encoder.py
Deep Transformer Reranker: Fine-tunes xlm-roberta-base on hard/uncertain pairs ($p \in [0.02, 0.98]$) for joint token-level cross-attention.


export_ce.py
 / 

export_ce_test.py
Uncertain Pair Samplers: Extracts uncertain pairs from out-of-fold LightGBM outputs for cross-encoder training/inference.
5. Post-Processing	

restore_exact.py
Transductive Safeguard: Identifies low-coverage countries (France) and restores high-confidence exact name + house number pairs dropped by the adapted model (98.6% precision on train).


sibling_prior.py
Prior Shift Correction: Corrects candidate log-odds for house-number mismatch sub-populations via Bayesian label-shift ratios.


stage2.py
Group Consistency: Support-counting group features and expected-$F_{0.5}$ decision boundary optimizer (

decide_expected_f
).


hybrid_coverage.py
Vocabulary Routing: Fallback mechanism routing between models based on language coverage.
Runners & Tools	

validate_submission.py
Official Validator: Asserts header formats, ID prefixes, subset constraints (matched $\subseteq$ candidate), and disjointness.


kaggle_submission_runner.ipynb
Cloud Runner: Notebook setup to clone, install, and execute the full pipeline on Kaggle GPU instances.


Welcome to the **Business Entity Resolution** project! This document is designed as a complete, self-contained engineering guide for onboarding interns and new team members. It details what every file does, how the data flows through the system, the algorithms involved, and why each design choice was made.

---

## 🧭 Table of Contents

1. [The Big Picture: Mental Model & Data Flow](#1-the-big-picture-mental-model--data-flow)
2. [Dataset & Formats](#2-dataset--formats)
3. [File-by-File Breakdown: What Every File Does & How](#3-file-by-file-breakdown-what-every-file-does--how)
   - [Phase 1: Normalization & Preprocessing](#phase-1-normalization--preprocessing)
   - [Phase 2: Candidate Blocking (Sparse & Dense)](#phase-2-candidate-blocking-sparse--dense)
   - [Phase 3: Feature Engineering](#phase-3-feature-engineering)
   - [Phase 4: Classification & Modeling](#phase-4-classification--modeling)
   - [Phase 5: Post-Processing & Domain Adaptation](#phase-5-post-processing--domain-adaptation)
   - [Utilities, Runners & Diagnostics](#utilities-runners--diagnostics)
4. [Step-by-Step Execution Walkthrough](#4-step-by-step-execution-walkthrough)
5. [Core Concepts & Pitfalls Cheat Sheet](#5-core-concepts--pitfalls-cheat-sheet)

---

## 1. The Big Picture: Mental Model & Data Flow

### The Problem
We have canonical, clean target businesses in **Source 1** (e.g. `Christ Chapel | 2100 Cameron Drive, Dundalk, MD`).
We receive millions of noisy query listings in **Source 2** and **Source 3** (e.g. scraped directories, tax filings, web crawls).
Our task: for every query record, decide whether it matches a canonical entity in Source 1, and if so, which one.

### The 5-Stage Funnel
Comparing all Source-2/3 records against all Source-1 entities requires $O(N \times M) \approx 10^{13}$ comparisons, which is computationally impossible. We solve this with a multi-stage funnel:

```
+-----------------------------------------------------------------------------------+
| 1. RAW DATA (Millions of records in TSV)                                          |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| 2. NORMALIZATION (normalize.py, preprocess.py)                                     |
|    Indic script transliteration, consonant skeletons, address canonicalization     |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| 3. HYBRID BLOCKING (blocking.py, emb_candidates.py, emb_knn.py)                   |
|    - Sparse: Compound inverted index (reduces 10^7 pairs to ~15-20 candidates/S1) |
|    - Dense: LaBSE GPU embeddings for Indic native scripts (adds top 2 neighbors)   |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| 4. PAIRWISE FEATURE EXTRACTION (features.py)                                      |
|    RapidFuzz string similarities, house number edit types, competition context    |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| 5. MATCHING & DECISION (maximumoutput.py, cross_encoder.py, restore_exact.py)      |
|    - LightGBM GBDT + XLM-RoBERTa cross-encoder                                    |
|    - 1-to-1 disjoint constraint assignment + Macro F0.5 threshold optimization    |
|    - France transductive vocabulary adaptation + exact identity restore            |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
| 6. FINAL SUBMISSION (matching_results.tsv, candidate_pairs.tsv)                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dataset & Formats

The raw data lives in `dataset/train/` and `dataset/test/`:

| File | Columns | Description |
|---|---|---|
| `train_source1.tsv` / `test_source1.tsv` | `entity_id`, `business_name`, `business_address`, `country` | Clean canonical target entities ($S_1$). |
| `train_source2.tsv` / `test_source2.tsv` | `entity_id`, `business_name`, `business_address`, `country` | Noisy query listings ($S_2$). |
| `train_source3.tsv` / `test_source3.tsv` | `entity_id`, `business_name`, `business_address`, `country` | Additional noisy query listings ($S_3$). |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` | Comma-separated ground-truth matches. |

Countries present:
- **Train:** `US`, `India`
- **Test:** `US`, `India`, and `France` (an **unseen** country distribution!).

---

## 3. File-by-File Breakdown: What Every File Does & How

### Phase 1: Normalization & Preprocessing

#### 📄 [`normalize.py`](business_entity_resolution/src/normalize.py)
- **Role:** Pure text normalization engine. Country-agnostic by design.
- **Key Functions & Logic:**
  1. `translit_indic(s)`: A custom rule-based transliterator covering 9 Brahmic scripts: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, and Malayalam. Instead of heavy NLP packages, it uses Unicode character properties (`unicodedata.name`) to map consonants, matras (vowel signs), viramas, anusvaras, and schwa endings to phonetic Latin characters.
  2. `skeleton_word(w)` / `skeleton(s)`: Consonant skeletonizer. Converts words to typo-resilient phonetic representations by mapping `ph` $\to$ `f`, `w` $\to$ `v`, `q/c` $\to$ `k`, `z/g` $\to$ `j`, dropping all vowels (`aeiouy`), and collapsing duplicate consonants (`"holloway"` $\to$ `"hlv"`).
  3. `normalize_name(raw)`: Strips accents (`NFKD`), parses aliases (`d/b/a`, `f/k/a`, `trading as`, website URLs), strips legal entity suffixes (`inc`, `llc`, `pvt ltd`, `sarl`, `gmbh`), and returns multiple views:
     - `n_full`: Cleaned full name.
     - `n_core`: Name with legal and generic filler tokens removed.
     - `n_alias`: Pipe-separated alternate alias views.
     - `n_nospace`: Core name with whitespace removed (matches domain names).
  4. `normalize_address(raw)`: Standardizes street types using `ADDR_CANON` (`street` $\to$ `st`, `boulevard` $\to$ `bd`, `allee` $\to$ `all`), converts words to numeric ordinals (`first` $\to$ `1st`), normalizes US states and French administrative regions, extracts house numbers (`a_nums`), and isolates address words (`a_words`).

#### 📄 [`preprocess.py`](business_entity_resolution/src/preprocess.py)
- **Role:** Parallel dataset normalization runner.
- **Inputs:** `dataset/<split>/<split>_source*.tsv`.
- **How it works:**
  - Reads TSVs with Pandas.
  - Spawns a Python `multiprocessing.Pool` applying `normalize_name` and `normalize_address` across all rows in parallel chunks.
  - Combines Source 2 and Source 3 into a single unified query DataFrame $q$.
- **Outputs:** Saves cached binary Parquet files in `$WORK/`:
  - `<split>_s1.parquet`: Normalized Source 1 table.
  - `<split>_q.parquet`: Normalized combined query table ($S_2 \cup S_3$).

---

### Phase 2: Candidate Blocking (Sparse & Dense)

#### 📄 [`blocking.py`](business_entity_resolution/src/blocking.py)
- **Role:** Reverse Top-$K$ candidate blocking via an inverted index.
- **Why "Reverse"?**
  Every query record belongs to **at most one** Source-1 entity. Therefore, we index Source-1 and for each query record, we retrieve its Top-$K$ candidate Source-1 entities (within the same country).
- **Compound Blocking Keys:**
  Simple token keys are too frequent and cause index explosion. We generate compound keys:
  - `N:<skeleton>`: Sorted name skeleton tokens.
  - `b:<skel_i>_<skel_j>`: Pairs of name skeleton tokens.
  - `i:<initials>`: Acronyms (`"tci"` for `"Talava Certified Ishares"`).
  - `p:<prefix>`: First 5 letters of space-less name.
  - `h:<number>_<word>`: House number paired with street word (e.g. `h:105_elm`).
  - `x:<name_skel>_<address_word>`: Name skeleton $\times$ address word.
- **Inverted Index Math:**
  Keys with document frequency above `df_cap` (default 2000) are pruned as stop-words. Candidate scores are calculated using an **IDF-weighted sparse matrix multiplication**:
  $$\text{Score}(q, s) = \sum_{k \in \text{keys}(q) \cap \text{keys}(s)} \text{IDF}(k)^2$$
- **Outputs:** `<split>_cand_K<K>_cap<cap>.parquet` with columns `[q_idx, s1_idx, score, rank]`.

#### 📄 [`export_native.py`](business_entity_resolution/src/export_native.py)
- **Role:** Isolates records written in native Indic scripts for GPU processing.
- **How it works:** Inspects query records for characters in the Unicode Indic range `[ऀ-ൿ]`. Exports these native-script name/address strings into `$EMB/<split>_native.parquet`.

#### 📄 [`emb_knn.py`](business_entity_resolution/src/emb_knn.py)
- **Role:** GPU-accelerated multilingual dense embedding retrieval.
- **Model:** `sentence-transformers/LaBSE` (Language-agnostic BERT Sentence Embeddings, 471M params, Apache-2.0).
- **How it works:**
  - Encodes Source-1 entity names and native-script query names into 768-dimensional normalized vectors.
  - Computes cosine similarity matrices on GPU via PyTorch FP16 matrix multiplication (`torch.matmul`).
  - Finds the top-100 nearest Source-1 candidates for each native-script query.
- **Outputs:** Saves `$EMB/<split>_labse_knn.parquet`.

#### 📄 [`emb_candidates.py`](business_entity_resolution/src/emb_candidates.py)
- **Role:** Merges dense embedding neighbors into the sparse candidate pool.
- **How it works:**
  - Names alone are ambiguous (many Indian businesses share similar titles).
  - Re-ranks LaBSE top-100 candidates using address context:
    $$\text{Rerank Score} = \text{Cosine} + 0.5 \times \text{Jaccard}(A_{\text{nums}}) + 0.5 \times \text{Jaccard}(A_{\text{words}})$$
  - Takes up to `extra` (default 2) top re-ranked candidates that were missed by sparse blocking and appends them to the candidate list.
  - Appends dense feature columns: `emb_cos` (LaBSE cosine similarity) and `is_emb` (flag indicating an embedding candidate).

---

### Phase 3: Feature Engineering

#### 📄 [`features.py`](business_entity_resolution/src/features.py)
- **Role:** Computes 30+ pairwise distance, overlap, and candidate-competition features for each candidate pair.
- **Key Feature Categories:**
  1. **String Distances (via RapidFuzz C++ engine):**
     - Full name & core name: `fuzz.ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio`.
     - Alias & domain matching: best token set ratio across aliases, Jaro-Winkler similarity on space-less names.
     - Skeleton similarity: `fuzz.ratio` on consonant skeletons.
  2. **Address & House Number Matching:**
     - Address string ratio, address word token set ratio, address word Jaccard.
     - First house number exact match (`num_first_eq`), suffix match (`101` vs `101A`), number set Jaccard, count of numbers.
  3. **Candidate Competition Context (`add_context`):**
     - `blk_score`: Raw blocking score.
     - `blk_rank`: Rank of this candidate among all retrieved candidates for this query record.
     - `blk_margin`: Difference between this candidate's score and the query record's top candidate score ($\text{score} - \text{top\_score}$).
     - `blk_gap12`: Margin between the record's 1st and 2nd best candidate scores (measures retrieval ambiguity).
     - `s1_ncand`: Total candidates competing for this Source-1 entity.
- **Outputs:** Feature matrix Parquet files with columns matching `FEATS`.

---

### Phase 4: Classification & Modeling

#### 📄 [`common.py`](business_entity_resolution/src/common.py)
- **Role:** Shared utilities, data loaders, and metric implementations.
- **Key Functions:**
  - `load_split(work, split)`: Loads normalized $S_1$ and $q$ Parquet files.
  - `gt_pairs(work, s1, q)`: Resolves ground-truth TSVs to dense integer arrays mapping each query index `q_idx` to its true `s1_idx` (-1 if unassigned).
  - `fbeta_from_assign(assign, truth, n_s1, keep=None, beta=0.5)`: **Exact evaluation metric**. Computes the competition Macro $F_{0.5}$ score across all Source-1 entities (including singletons with 0 matches).

#### 📄 [`pipeline.py`](business_entity_resolution/src/pipeline.py)
- **Role:** Baseline Stage-1 LightGBM trainer and evaluator.
- **How it works:**
  - Trains a GBDT binary classifier (`num_leaves=127`, `learning_rate=0.08`, `rounds=400`).
  - **Grouped 2-Fold CV:** Folds are split strictly by `source1_entity_id` (`cand.s1_idx % 2`), preventing target leakage.
  - `assign_from_prob(cand, prob, n_q, thr)`: Assigns each query record to its highest probability candidate if $P > \text{thr}$, strictly enforcing the 1-to-1 assignment constraint.
  - Sweeps probability thresholds to maximize validation Macro $F_{0.5}$.

#### 📄 [`maximumoutput.py`](business_entity_resolution/src/maximumoutput.py)
- **Role:** The **Champion Model Pipeline** (Stage-1 + LaBSE + Robust Stage-2).
- **The "Sibling Distractor" Discovery:**
  - Standard Stage-2 group features count how many other query records support this candidate's house number.
  - In the test set, "sibling businesses" (businesses on the same street with adjacent numbers like 101 vs 102 whose canonical S1 entity is absent) formed clusters that vouched for themselves, causing a drop on the public leaderboard.
  - `maximumoutput.py` drops self-support features (`SELF_SUPPORT = ["g_num_me_w", "g_num_me_n", ...]`), ensuring sibling clusters cannot artificially inflate their confidence.
- **Outputs:** Generates `output_v5a/` and `output_v5b/` containing `matching_results.tsv` and `candidate_pairs.tsv`.

#### 📄 [`cross_encoder.py`](business_entity_resolution/src/cross_encoder.py)
- **Role:** Deep transformer pair classifier fine-tuned on hard/uncertain pairs.
- **Model:** `xlm-roberta-base` (278M parameters, MIT license).
- **How it works:**
  - Instead of encoding records separately (bi-encoder), it feeds both records into the transformer simultaneously:
    `<s> <name_q> | <address_q> </s></s> <name_s1> | <address_s1> </s>`
  - Allows full cross-attention over subtle token substitutions and digit transpositions.
  - Trains exclusively on uncertain candidate pairs ($P_{\text{LightGBM}} \in [0.02, 0.98]$).
  - Achieves **0.985 AUC** on hard pairs, acting as a powerful stacking feature.

#### 📄 [`export_ce.py`](business_entity_resolution/src/export_ce.py) & [`export_ce_test.py`](business_entity_resolution/src/export_ce_test.py)
- **Role:** Data extraction scripts that identify uncertain pairs from LightGBM out-of-fold predictions and export tokenized text batches for `cross_encoder.py`.

---

### Phase 5: Post-Processing & Domain Adaptation

#### 📄 [`restore_exact.py`](business_entity_resolution/src/restore_exact.py)
- **Role:** Transductive safeguard for unseen countries (France).
- **The Problem:** The training set only contained US and India. Word-frequency odds learned from train only covered 51% of French vocabulary. Complex models became over-cautious and dropped obvious true matches.
- **The Solution:** For countries with vocabulary coverage below 80%:
  - Identifies candidate pairs where the core name (`n_core`) is **identical** AND the first house number is **identical**.
  - On the training set, this pattern is a true match **98.6% of the time**.
  - Restores any such pairs that were rejected by the adapted model, as long as the query record was not assigned to another entity.

#### 📄 [`sibling_prior.py`](business_entity_resolution/src/sibling_prior.py)
- **Role:** Bayesian prior-shift (label-shift) calibration for house-number mismatches.
- **Math:** Adjusts candidate log-odds using the ratio of test-to-train prior distribution of sibling pairs:
  $$w = \frac{\text{Odds}(\pi_{\text{test}})}{\text{Odds}(\pi_{\text{train}})}$$

#### 📄 [`stage2.py`](business_entity_resolution/src/stage2.py)
- **Role:** Group-consistency feature builder and `decide_expected_f` per-entity dynamic threshold optimizer.

#### 📄 [`hybrid_coverage.py`](business_entity_resolution/src/hybrid_coverage.py)
- **Role:** Ensembles predictions by routing: applies high-capacity models (v9/v10) to well-covered countries and falls back to conservative models (v7) for low-coverage countries.

---

### Utilities, Runners & Diagnostics

#### 📄 [`utils/validate_submission.py`](utils/validate_submission.py)
- **Role:** Official submission checker. Verifies headers, IDs, subset consistency (`matched_ids` $\subseteq$ `candidate_ids`), and ensures no duplicate assignments.

#### 📄 [`kaggle_submission_runner.ipynb`](kaggle_submission_runner.ipynb)
- **Role:** Kaggle GPU notebook designed to clone this repo, install dependencies, run the pipeline on Kaggle GPUs, and package outputs.

#### 📁 [`dev/`](business_entity_resolution/src/dev/)
- **Role:** Research diagnostics (e.g. `loss_v7.py` for error decomposition, blocking recall experiments). Not required for test inference.

---

## 4. Step-by-Step Execution Walkthrough

Here is the exact command sequence an intern should follow to reproduce the entire pipeline from scratch:

```bash
# Navigate to the source directory
cd business_entity_resolution/src

# Define paths
DATA=../../dataset
WORK=../work
EMB=../emb
mkdir -p $WORK $EMB

# -------------------------------------------------------------
# STEP 1: Text Preprocessing & Parquet Caching
# -------------------------------------------------------------
# Normalizes names, transliterates Indic text, and extracts numbers
python preprocess.py --data $DATA --work $WORK --split train
python preprocess.py --data $DATA --work $WORK --split test

# -------------------------------------------------------------
# STEP 2: Candidate Generation (Sparse Blocking)
# -------------------------------------------------------------
# Builds compound inverted index and retrieves top-3 candidates
python pipeline.py train   --work $WORK --K 5 --Kuse 3 --df_cap 2000 --rounds 400
python pipeline.py predict --work $WORK --out ../output_base

# -------------------------------------------------------------
# STEP 3: Multilingual Embedding Retrieval (GPU)
# -------------------------------------------------------------
# Encodes Indic records with LaBSE and finds top-100 nearest neighbours
python export_native.py --work $WORK --emb_dir $EMB
python emb_knn.py $EMB train test

# -------------------------------------------------------------
# STEP 4: Train Champion Model (LightGBM + Robust Stage-2)
# -------------------------------------------------------------
# Re-ranks LaBSE neighbours, computes pairwise features, and trains GBDT
python maximumoutput.py --work $WORK --knn_dir $EMB --out_root .. --extra 2

# -------------------------------------------------------------
# STEP 5: Exact Identity Restore for France (Unseen Country)
# -------------------------------------------------------------
# Recovers high-precision exact-match pairs for low-coverage countries
python restore_exact.py $WORK ../output_v5a ../output_base ../output_final France

# -------------------------------------------------------------
# STEP 6: Validate Final Submission Files
# -------------------------------------------------------------
python ../../utils/validate_submission.py \
    --matching ../output_final/matching_results.tsv \
    --candidate ../output_final/candidate_pairs.tsv \
    --test-dir $DATA/test
```

---

## 5. Core Concepts & Pitfalls Cheat Sheet

### 1. Why Macro $F_{0.5}$ and not $F_1$?
In $F_{\beta}$, $\beta = 0.5$ places **twice as much weight on precision than recall**:
$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times P \times R}{0.25 P + R}$$
- **Lesson for Interns:** False merges (assigning a query record to the wrong business) hurt your score far more than missing a difficult match. When in doubt, prefer precision over recall.

### 2. The 1-to-1 Disjointness Constraint
Each Source-2/3 query record can belong to **at most one** Source-1 entity:
- If a query record has multiple candidate entities scoring above threshold, **assign it only to the candidate with the highest probability**.
- `pipeline.py` enforces this with `assign_from_prob` using lexicographical sorting.

### 3. Sibling Business Distractors
- **What they are:** Two distinct businesses located on the same street with adjacent house numbers (e.g. `Pizza Hut` at `101 Main St` vs `103 Main St`).
- **The trap:** In training, most candidates with matching names on the same street were true matches. In test, many queries are sibling businesses whose true Source-1 entity was omitted from the database.
- **The fix:** Do not use features that count how many other query records support the record's *own* house number. Rely on difference-detection features (house-number edit taxonomy).

### 4. Grouped Cross-Validation is Mandatory
- **Never split train/validation randomly by row!**
- If records belonging to the same `source1_entity_id` appear in both train and validation folds, the model will overfit on specific entity names, giving inflated validation scores that collapse on the test set.
- We group folds strictly by `source1_entity_id` (`cand.s1_idx % 2`).
