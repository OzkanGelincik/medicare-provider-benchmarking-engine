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
This repo deliberately ships **two row-level** and **two provider-level** anomaly products, each frozen from artifacts to be reproducible.

### Row-level anomalies (worklists)
1. **Primary: ANOM.2.a.1 (robust + size-aware)**  
   Goal: surface rows that are extreme on a robust tail definition while respecting slice support (peer group size).

2. **Alternative: ANOM.2.b.1 (magnitude-aware + size-aware)**  
   Goal: contrast view that emphasizes magnitude/severity of over-expected behavior.

### Provider-level anomalies (watchlists)
3. **ANOM.3.a.1 (repeat offenders)**  
   Providers ranked by how often they show up as robust row-level tail events across codes and years.

4. **ANOM.3.b.1 (shock/severity providers)**  
   Providers ranked by severe “shock” behavior (magnitude tail).

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









