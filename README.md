# 🏥 Medicare Provider Benchmarking Engine  
A reproducible, end-to-end portfolio project that builds an **expected-cost benchmarking engine** on Medicare provider service lines, then layers on **anomaly surfacing**, **provider tiering (clustering)**, and a **transparent explainer model**, all wrapped in a **Python Shiny** product-style app.

> ⚠️ Important framing: This is **not fraud detection**.  
> This system surfaces **model-based cost variance** and **statistical outliers** that warrant **follow-up review**.

---

## ✨ What this repo contains (high-level)
### ✅ Core products
1. **Expected-cost benchmark engine (final scoring: `DG_V3`)**  
   Produces expected cost and residual signals at a row-grain used downstream.
2. **Anomaly surfacing (four “products”)**
   - **ANOM.2.a.1**: Row-level, **robust + size-aware** (primary)
   - **ANOM.2.b.1**: Row-level, **magnitude-aware + size-aware** (contrast)
   - **ANOM.3.a.1**: Provider-level, **repeat offenders** (robust)
   - **ANOM.3.b.1**: Provider-level, **shock/severity** (magnitude)
3. **Provider tiering (clustering on eligible providers)**  
   A strict “minimum evidence” gate, then KMeans-based segmentation of providers into:
   - `cluster_0`: typical behavior  
   - `cluster_1`: elevated anomaly burden
4. **Explainer classification model (logistic regression, no-state)**  
   A deliberately simple, auditable model used to explain what predicts `cluster_1` **without leaking clustering features**.
5. **Python Shiny app (product layer)**  
   A narrative-first UI that lets someone explore the benchmark logic, anomalies, tiering, and explainability.


---

## 🚀 Live demo (Python Shiny)
Explore the results interactively: https://ozkangelincikshinyapp.shinyapps.io/medicare-provider-benchmarking-engine/

---

## 🎯 Goal
Build a system that answers:

- **Row level:** “Which provider-service rows are most over-expected relative to comparable peers?”  
- **Provider level:** “Which providers repeatedly show up as extreme over-expected?”  
- **Segmentation:** “Can we tier providers into interpretable groups based on their anomaly footprint?”  
- **Explainability:** “What provider characteristics are associated with being in the elevated anomaly cluster, beyond the clustering features?”

---

## 🔬 Data grain and key definitions
### Row grain (the fundamental scoring unit)
**Row-grain used throughout the product layer:**
- `(Rndrng_NPI, HCPCS_Cd, Place_Of_Srvc, Year)`

This is the unit where the benchmark engine outputs:
- `observed_cost`
- `expected_cost`
- `residual = observed_cost - expected_cost`
- `oe_ratio = observed_cost / expected_cost`
- `log_oe = log(oe_ratio)` (when expected_cost > 0)

### Provider grain
- `(Rndrng_NPI, provider_type, state)`

Provider-level summaries roll up row-level behavior (counts, rates, tail metrics, volume).

---

## 🧠 Benchmarking engine (Expected cost, final: `DG_V3`)
### What it outputs
The frozen evaluation universe includes the scored row-grain dataset:

- `artifacts/eval_universe/eval_scored_DG_V3.parquet`

This file is the **single source of truth** used by:
- Benchmark Explorer plots
- Overview distributions
- Slice summaries
- Provider drill-downs
- Anomaly workflows (via frozen worklists)

### Why we moved to HCPCS-level benchmarking
The engine evolved to emphasize **HCPCS-level** benchmarking because it is:
- more granular and interpretable for operational review,
- better aligned with how anomalies are actioned (code-level review),
- consistent with downstream “slice” logic (HCPCS × Year, optionally stratified by other context).

---

## 🛡️ Guardrails (robustness layer)
The pipeline includes guardrail logic to reduce failure modes where expected cost can become unrealistically low or unstable (especially in cold-start or sparse situations).

Two concrete guardrail patterns that appear in the frozen artifacts and anomaly worklists:

- **Cold-start floor for 2023-only “cold” rows**  
  A “global floor bounded by observed” pattern for very low predicted expected cost in a specific cold-start segment.

- **Residual cap bounded by observed for specific clinical families**  
  Example guardrail name seen in anomaly outputs:  
  `C5v2_radiation_planning_family_residual_cap_bounded_by_observed`

> ✅ Principle: guardrails are *post-processing safety policies* applied to scoring outputs to reduce brittle, high-impact underestimation and improve downstream anomaly stability.

---

## 🚨 Anomaly surfacing (four products)

This repo deliberately ships **two row-level worklists** and **two provider-level watchlists**, each **frozen from artifacts** for reproducibility.

Across all products, the workflow is intentionally “audit-friendly”:
- **Start from row-grain predictions** at `(Rndrng_NPI, HCPCS_Cd, Place_Of_Srvc, Year)` with `observed_cost`, `expected_cost`, `residual`, `oe_ratio`, `log_oe`.
- **Enforce minimum peer support** per comparison slice using `slice_n` (the peer group size).
- **Prefer high-support rows** (e.g., strong expected-cost support tiers) when defining “high confidence”.
- **Rank** by an explicit anomaly score (robust tail vs magnitude tail).
- **Freeze top-N outputs** (e.g., top 200) into reproducible worklists/watchlists.

---

### Row-level anomalies (worklists)

#### 1) **Primary: ANOM.2.a.1 (robust + size-aware)**
**What it considers an “anomaly”**
- A row is anomalous if it is **extreme within its peer slice** on a **robust tail definition**, with enough slice support to trust the percentile.

**How the product is built (step-by-step)**
- **Grain:** row-level `(NPI, HCPCS, POS, Year)`
- **Compute core signals:**  
  - `residual = observed_cost - expected_cost`  
  - `oe_ratio = observed_cost / expected_cost`  
  - `log_oe = log(oe_ratio)`  
- **Define peer slice:** comparison group within the same **HCPCS × Year** (and your fixed peer constraints).
- **Apply eligibility filters (row-level):**
  - Require **positive residual** (over-expected behavior).
  - Require **minimum slice support** `slice_n >= MIN_SLICE_N` (size-aware trust gate).
  - Require **high confidence** (e.g., strong support tier and sufficient volume thresholds, per frozen params).
- **Score (robust tail):**
  - Mark as a **robust tail event** if `log_oe` is in the **top 1% of its slice** (or your configured robust percentile rule).
  - `anom_score_robust` reflects this robust tail exceedance (per frozen params).
- **Rank + freeze:**
  - Sort rows by `(anom_score_robust desc, log_oe desc, residual desc)` (or your configured ordering).
  - **Freeze top 200 rows** as the primary row-level worklist.

**What it surfaces**
- Rows that are **consistently extreme relative to peers** under a percentile-based robust definition, not just large-dollar outliers.

---

#### 2) **Alternative: ANOM.2.b.1 (magnitude-aware + size-aware)**
**What it considers an “anomaly”**
- A row is anomalous if it exhibits **severe magnitude** of over-expected behavior (a “shock-style” view), while still respecting peer slice support.

**How the product is built (step-by-step)**
- **Grain:** row-level `(NPI, HCPCS, POS, Year)`
- **Compute core signals:** same as above (`residual`, `oe_ratio`, `log_oe`).
- **Define peer slice:** same slice definition used for robust worklist.
- **Apply eligibility filters (row-level):**
  - Require **positive residual**.
  - Require **minimum slice support** `slice_n >= MIN_SLICE_N`.
  - Require **high confidence** (same strict gate used in primary, per frozen params).
- **Score (magnitude tail):**
  - Emphasize **severity**, typically via `log_oe` and/or `residual` magnitude under your frozen magnitude-tail rule.
  - `anom_score_mag` captures the “shock” intensity (per frozen params).
- **Rank + freeze:**
  - Sort rows by `(anom_score_mag desc, log_oe desc, residual desc)` (or your configured ordering).
  - **Freeze top 200 rows** as the alternative magnitude-focused worklist.

**What it surfaces**
- Rows that look most like **rare, high-severity spikes**, useful as a contrast view to the robust percentile product.

---

### Provider-level anomalies (watchlists)

#### 3) **ANOM.3.a.1 (repeat offenders)**
**What it considers an “anomalous provider”**
- A provider is anomalous if they **repeatedly** show up as **robust row-level tail events** across **many codes and years** (breadth + recurrence).

**How the product is built (step-by-step)**
- **Start from:** the frozen **row-level robust events** (ANOM.2.a.1 candidate logic applied across the eligible universe).
- **Aggregate to provider grain:** `(Rndrng_NPI, provider_type, state)`
- **Compute provider signals:**
  - `n_anom_rows_robust`: count of robust tail rows for that provider.
  - `anom_rate_pct_robust`: `100 * n_anom_rows_robust / n_rows`
  - `n_unique_codes`, `n_unique_years`: breadth of where the provider shows up.
  - Optional stability context: `median_log_oe`, `median_residual`, `total_services`, `total_benes`.
- **Apply provider eligibility filters:**
  - Require minimum evidence depth (e.g., minimum `n_rows`, minimum services) so rates are meaningful.
- **Score (repeat offender):**
  - `provider_anom_score_robust` increases with **frequency** and **breadth** of robust events (per frozen params).
- **Rank + freeze:**
  - Rank providers by `provider_anom_score_robust` (and tie-break with recurrence/breadth).
  - **Freeze top-N providers** as the repeat-offender watchlist.

**What it surfaces**
- Providers who “**consistently pop**” across the dataset, not providers with just one extreme claim.

---

#### 4) **ANOM.3.b.1 (shock / severity providers)**
**What it considers an “anomalous provider”**
- A provider is anomalous if they have **rare but very severe** magnitude events, even if they do not have many robust tail events.

**How the product is built (step-by-step)**
- **Start from:** the frozen **row-level magnitude events** (ANOM.2.b.1 logic).
- **Aggregate to provider grain:** `(Rndrng_NPI, provider_type, state)`
- **Compute provider signals:**
  - `n_anom_rows_mag`: count of magnitude-tail rows for that provider.
  - `anom_rate_pct_mag`: `100 * n_anom_rows_mag / n_rows`
  - `max_log_oe_mag`, `p95_log_oe_mag`: severity summary of the provider’s worst tail behavior.
  - Context: `median_log_oe`, `total_services`, `total_benes`.
- **Apply provider eligibility filters:**
  - Require minimum evidence depth, so severity isn’t dominated by tiny denominators.
- **Score (shock/severity):**
  - `provider_anom_score_mag` emphasizes **how extreme the provider’s worst events are** (per frozen params).
- **Rank + freeze:**
  - Rank providers by `provider_anom_score_mag` (severity-first).
  - **Freeze top-N providers** as the shock/severity watchlist.

**What it surfaces**
- Providers who exhibit **“spike” behavior**, useful for triage when you care most about the highest-severity events.

---

✅ **Why four products (and not one)**
- **Robust products** are best for “repeatability” and defensible peer-relative outliers.
- **Magnitude products** are best for “severity-first” triage and shock-style events.
- Having both at **row** and **provider** levels lets you move from **worklists** (what happened) to **watchlists** (who repeatedly or severely shows it).

---

## 🧩 Tiering (eligibility gate + clustering)
### Eligibility gate (strict by design)
Clustering is performed only on providers who pass a **minimum evidence** gate (volume and row depth).  
This keeps the tiering stable and prevents “thin data” providers from being over-interpreted.

### Clustering output
Eligible providers receive:
- `cluster_label_v1` (`cluster_0` or `cluster_1`)
- `cluster_definition_v1` (human-readable meaning)
- `dist_to_centroid_v1` (how atypical within their assigned cluster)

Example counts from a recent frozen run (as seen in the app UI):
- Providers (total): **23,366**
- Eligible (clustered): **7,488**
- `cluster_0`: **5,662**
- `cluster_1`: **1,826** (≈ **24.39%** of eligible)

---

## 🧾 Explainer model (classification, explanation-first)
### Why classification exists in this project
We did **not** build classification to “beat clustering.”  
We built it to answer a defensible question:

> “Among eligible providers, what characteristics are associated with being in `cluster_1` beyond the clustering features, and without leakage?”

### Final locked explainer model
✅ **Logistic regression (no-state, explicit baselines)**  
- chosen for transparency, auditability, and stable directionality  
- trained on **eligible providers only**
- excludes `state` for transferability and to avoid learning geography as a shortcut

Key performance (eligible-only, cluster_1 vs cluster_0):
- **ROC AUC:** ~0.729  
- **PR AUC:** ~0.443  
(These numbers are from the CLF.2.c.1 output in the modeling workflow.)

### Stability (bootstrap)
The logistic model includes a bootstrap stability check that reports:
- how often each feature lands in the **top-k by |coef|** across resamples
- odds ratio distribution across resamples
- (and optionally sign stability)

Headline stable drivers observed in the bootstrap output include:
- `log_total_services_base` (strong, consistently positive)
- provider type contrasts (e.g., Surgical Oncology and Gynecological Oncology strongly negative vs baseline)
- geography context via RUCA buckets (Rural/Suburban vs Urban baseline)

---

## 🖥️ Python Shiny app (product layer)
The Shiny app is designed as a **portfolio showcase**, not an internal analyst tool. It prioritizes:
- narrative flow over dense controls,
- reproducibility via frozen artifacts,
- transparent rules and parameter displays.

### Top-level navigation
- **Overview**: KPIs, distribution views, slice summary, cluster label card
- **Benchmark Explorer**: trust-builder plots  
  - Slice mode: HCPCS × Year, with optional POS/State/NPI refinement  
  - Provider mode: NPI explorer
- **Anomaly Surfacing**: row-level and provider-level worklists (stacked tables) + rules
- **Tiering**: eligibility and clustering diagnostics + cluster profiling
- **Explainer Model**: artifact provenance and what-if provider view
- **Provider Profile**: stitched narrative for a single NPI (tiering + anomalies + explainer contributions)
- **Documentation**: run provenance + loaded params JSON

### Artifact-driven reproducibility
The app auto-discovers the latest frozen runs under:
- `artifacts/eval_universe/`
- `artifacts/anomaly_surfaces/`
- `artifacts/provider_tiering/`
- `artifacts/provider_classification/`

---

## 📁 Repo structure (what matters)
- `app.py`  
  Python Shiny app (portfolio showcase UI)
- `artifacts/`  
  Frozen outputs used by the app (parquets, params, manifests)
- `preprocessing.py`  
  Shared utilities used by notebooks and/or app workflows
- `*.ipynb` notebooks  
  EDA, modeling, guardrails, anomaly surfacing, tiering, classification

> Note: In deployment, only the subset of artifacts required by the app is bundled.

---

## ▶️ Run locally
### 1) Create and activate an environment
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2) Run the app
```bash
python -m shiny run --reload app.py
```

---

## 🚀 Deployment (shinyapps.io)


This repo deploys using Option A: artifacts included in the app bundle (but carefully excluded to keep bundle size manageable).

A working pattern is to deploy from repo root while excluding heavy folders:

```bash
rsconnect deploy shiny . \
  --name ozkangelincikshinyapp \
  --title medicare-provider-benchmarking-engine \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.ipynb" \
  --exclude "catboost_info" \
  --exclude "data" \
  --exclude "models" \
  --exclude "eval_pack" \
  --exclude "holdout_sanity_pack" \
  --exclude "artifacts/failure_analysis_v2" \
  --exclude "artifacts/models" \
  --exclude "artifacts/guardrails" \
  --exclude ".DS_Store" \
  --exclude "src/bench" \
  --exclude "data/nppes"

```

---

## 🧭 How to read outputs (mental model)

1. Benchmark Explorer first
Validate expected vs observed behavior at the same row grain used downstream.
2. Anomaly Surfacing second
Trust-building: anomalies are not magic, they are rule-based on residual/tail logic.
3. Tiering third
The cluster is a segmentation of eligible providers based on anomaly footprint.
4. Explainer last
A transparent “why cluster_1” model that avoids leakage and over-claiming.

--- 

✅ What this project demonstrates (portfolio signals)
- ✅ End-to-end data product thinking (auditability, frozen artifacts, reproducibility)
- ✅ Robust modeling mindset (guardrails, stability checks, eligibility gating)
- ✅ Multiple anomaly definitions (robust vs magnitude) instead of one fragile metric
- ✅ Explainability without ML theater (logistic + bootstrap stability)
- ✅ Product layer delivery (Shiny app with narrative UX and provenance)

---

## 📌 Disclaimer

This project surfaces statistical anomalies and model variance signals.
It does not claim fraud, wrongdoing, or intent. Any surfaced result requires domain review and additional context.

---

## 👨🏻‍💻 Author

Ozkan Gelincik — Data Scientist

🔗 LinkedIn: https://www.linkedin.com/in/ozkangelincik

🌎 Visit my website: https://ozkangelincik.com









