# Narrative Geometry

**Algebraic Orthogonalization on the Stiefel Manifold for Monitoring the Relative Dominance of Competing Narratives**

This repository contains the methodology, empirical pipelines, and reproducible notebooks for **Narrative Geometry**—an algebraic framework designed to measure the relative prevalence of competing narratives within time-evolving text corpora.

---

## 1. Overview & Methodology

When tracking narratives in a corpus using modern bidirectional text encoders (such as `ModernBERT`), standard cosine similarity between article embeddings and textual descriptions of narratives ("raw narratives" $N_i$) suffers from a severe **structural correlation bias**. Because narrative descriptions are written in the same language, share grammatical structures, and address related macro-themes, their embedding vectors exhibit high pairwise cosine similarities (often $0.60\text{--}0.96$). Consequently, naive cosine projections produce synchronized time series that reflect overall topic volume or shared background drift rather than the idiosyncratic dynamics of individual narratives.

### Key Methodological Pillars

1. **Linear Generative Model in Embedding Space**:
   Each article embedding $X_n \in \mathbb{R}^d$ is modeled as a linear combination of orthonormal **pure narratives** $\{V_i\}_{i=1}^K$ weighted by binary presence indicators $\alpha_{i,n} \in \{0,1\}$, plus an orthogonal residual $\varepsilon_n \in \operatorname{Vect}(V_1,\ldots,V_K)^\perp$ capturing style and unmodeled topics:
   $$X_n = \sum_{i=1}^K \alpha_{i,n}\, V_i + \varepsilon_n$$
   Under orthonormality ($V_i^\top V_j = \delta_{ij}$), the empirical conditional inner product $\hat{p}_i(t) = \frac{1}{n_t}\sum_{n \in \mathcal{C}_t} V_i^\top X_n$ is an **unbiased estimator** of true narrative prevalence $p_i(t) = \mathbb{E}[\alpha_{i,n} \mid \mathcal{F}_t]$, free of signal leakage across narratives.

2. **Max-Min Orthogonalization on the Stiefel Manifold**:
   Unlike sequential Gram–Schmidt (which is order-dependent and severely distorts later narratives) or Orthogonal Procrustes (which maximizes the sum of alignments and can sacrifice minority narratives), we solve a global **max-min problem** on the Stiefel manifold $\mathrm{St}(K, d) = \{V \in \mathbb{R}^{K \times d} : V V^\top = I_K\}$:
   $$\max_{V \in \mathrm{St}(K, d)} \min_{i=1,\ldots,K} \langle N_i^*, V_i \rangle$$
   Using a smooth soft-min surrogate with temperature $\tau = 0.05$:
   $$f_\tau(V) = -\tau \log \sum_{i=1}^K \exp\!\left(-\frac{\langle N_i^*, V_i \rangle}{\tau}\right)$$
   optimized via Riemannian Conjugate Gradient (`pymanopt`) initialized from Gram–Schmidt, this constructs a collection of **pure narratives** $\{V_i\}_{i=1}^K$ that removes shared semantic background while preserving worst-case fidelity to the original narrative directions.

3. **Quantitative & Statistical Validation**:
   - **Covariation Reduction**: Evaluates pairwise second-moment co-variation $M_{ij}(t) = \mathbb{E}[(X^\top W_i)(X^\top W_j) \mid \mathcal{F}_t]$, reducing artificial cross-narrative correlation by an order of magnitude.
   - **Semantic Specificity ($\Delta_{\text{gain}}$)**: Uses labeled or clustered thematic groups ("chapters") as a counterfactual to verify that orthogonalized vectors increase the separation between in-theme and out-of-theme similarity: $\Delta^i(V_i) > \Delta^i(N_i)$.
   - **Rupture Detection & Jaccard Overlap**: Applies moving-window two-sided Welch $t$-tests with Benjamini–Hochberg FDR correction to isolate narrative-specific historical structural breaks.

---

## 2. Project Roadmap: Reference Paper vs. CNN Case Study

This repository supports two complementary studies that share the core Stiefel orthogonalization engine but differ in how narratives and thematic structures are constructed:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          NARRATIVE GEOMETRY ROADMAP                                 │
├──────────────────────────────────────────┬──────────────────────────────────────────┤
│  STUDY A: Reference Paper (HuffPost)     │  STUDY B: Worked Case Study (CNN Iran)   │
│  "The Algebra of Narratives and the      │  "Clustering, Prototype Narratives, and  │
│   Monitoring of Their Relative Dominance"│   Stiefel-Manifold Orthogonalization"    │
├──────────────────────────────────────────┼──────────────────────────────────────────┤
│ • Corpus: News Category Dataset          │ • Corpus: CNN Live-Blog Coverage of the  │
│   (~200k HuffPost headlines, 2012–2018)  │   2026 US–Israel–Iran War (1,572         │
│ • Ground Truth: Exogenous human-assigned │   snippets, March–August 2026)           │
│   category labels (Politics,             │ • Thematic Discovery: Unsupervised       │
│   Entertainment, Parenting, Travel)      │   PCA (50D) + K-Means (K=8 clusters) in  │
│ • Embeddings: answerdotai/ModernBERT-base│   natural embedding space                │
│ • Narrative Construction: 12 single      │ • Embeddings:                            │
│   representative paragraphs (3 per       │   nomic-ai/modernbert-embed-base         │
│   chapter)                               │ • Narrative Construction: 3 curated      │
│ • Key Focus:                             │   sentences averaged into 1 stable       │
│   - Probabilistic generative theory      │   "prototype of narrative" per cluster   │
│   - Temporal resampling & EMA prevalence │ • Key Focus:                             │
│   - Bootstrap specificity CI (1,000x)    │   - End-to-end unsupervised pipeline     │
│   - 90-day Welch t-test + BH-FDR         │   - Prototype averaging                  │
│     rupture detection (2016 US Election, │   - Decomposition into Raw, Pure (V),    │
│     2017 ACA repeal)                     │     and Residual (shared war jargon)     │
├──────────────────────────────────────────┼──────────────────────────────────────────┤
│ Notebook:                                │ Notebook & Docs:                         │
│ notebooks/                               │ notebooks/CNN_Iran_War_scrapping.py      │
│  narratives_project_clean_version_...py  │ docs/CNN_Iran/                           │
└──────────────────────────────────────────┴──────────────────────────────────────────┘
```

| Feature | Study A: HuffPost Reference Paper | Study B: CNN Iran War Case Study |
| :--- | :--- | :--- |
| **Authors** | Juliette Anglade, Charles-Albert Lehalle, Mario Massy | Charles-Albert Lehalle |
| **Primary Script** | [`notebooks/narratives_project_clean_version_shared_3.py`](notebooks/narratives_project_clean_version_shared_3.py) | [`notebooks/CNN_Iran_War_scrapping.py`](notebooks/CNN_Iran_War_scrapping.py) |
| **HTML Output** | [`docs/notebooks/narratives_project_clean_version_shared_3.html`](docs/notebooks/narratives_project_clean_version_shared_3.html) | [`docs/notebooks/CNN_Iran_War_scrapping.html`](docs/notebooks/CNN_Iran_War_scrapping.html) |
| **Documentation** | Reference manuscript (*The Algebra of Narratives...*) | [`docs/CNN_Iran/stiefel_narrative_methodology_v3.tex`](docs/CNN_Iran/stiefel_narrative_methodology_v3.tex) & [`stiefel_narrative_slides.pdf`](docs/CNN_Iran/stiefel_narrative_slides.pdf) |
| **Validation Mode** | Supervised counterfactual (human-labeled HuffPost categories) | Label-free *ex post* validation (in-cluster vs. out-of-cluster separation) |

---

## 3. Repository Structure

```text
narrative-geometry/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── narratives_project_clean_version_shared_3.py   # Study A: HuffPost experiments (Jupytext percent format)
│   └── CNN_Iran_War_scrapping.py                      # Study B: CNN Iran War scraping, clustering & Stiefel pipeline
└── docs/
    ├── CNN_Iran/
    │   ├── stiefel_narrative_methodology_v3.tex       # Detailed 5-stage methodology report for Study B
    │   ├── stiefel_narrative_slides.tex               # Beamer presentation source
    │   ├── stiefel_narrative_slides.pdf               # Compiled presentation slides
    │   └── images/                                    # Generated figures (t-SNE, heatmaps, cluster histograms)
    └── notebooks/
        ├── narratives_project_clean_version_shared_3.html  # Pre-rendered HTML execution of Study A
        └── CNN_Iran_War_scrapping.html                     # Pre-rendered HTML execution of Study B
```

---

## 4. Installation & Reproduction Instructions

### 4.1 Environment Setup

Install the required dependencies in a virtual environment (Python 3.9+ recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Working with Jupytext Notebooks

The scripts in `./notebooks/` are stored in the `py:percent` [Jupytext](https://jupytext.org/using/cli/) format for clean version control.

- **Convert `.py` to `.ipynb` notebook**:
  ```bash
  jupytext --to notebook notebooks/narratives_project_clean_version_shared_3.py
  jupytext --to notebook notebooks/CNN_Iran_War_scrapping.py
  ```
- **Synchronize changes between `.py` and `.ipynb`**:
  ```bash
  jupytext --sync notebooks/<NOTEBOOK_NAME>.ipynb
  ```
- **Execute and export to HTML**:
  ```bash
  jupyter nbconvert --to html --execute --output=../docs/notebooks/<NOTEBOOK_NAME>.html notebooks/<NOTEBOOK_NAME>.ipynb
  ```

### 4.3 Reproducing Study A (`narratives_project_clean_version_shared_3.py`)

1. **Dataset**: Uses the publicly available [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset/data) (Misra, 2022) via `kagglehub.dataset_download("rmisra/news-category-dataset")`.
2. **Precomputation Step (Section 0)**:
   - On first run, the script downloads `News_Category_Dataset_v3.json`, computes 768-dimensional embeddings using `answerdotai/ModernBERT-base`, and saves them to `./data_narratives/news_with_embeddings.json`.
   - *Note*: Embedding ~200k headlines takes significant compute time on CPU; a CUDA or MPS GPU is strongly recommended. Subsequent sections load `./data_narratives/news_with_embeddings.json` directly.
3. **Self-Contained Sections**:
   - **Section III** (Stiefel optimization, Procrustes, Gram–Schmidt comparisons, and Figures 1 & 2 heatmaps on the 12 narratives) only embeds the 12 narrative descriptions and runs in a few seconds on CPU or GPU.
   - **Sections IV–VI** compute the weekly resampled prevalence curves, covariance tables, 1,000-draw bootstrap specificity confidence intervals, and 90-day Welch $t$-test rupture detections.

### 4.4 Reproducing Study B (`CNN_Iran_War_scrapping.py`)

1. **Stage 1 (Web Scraping)**: Fetches live-blog updates from CNN URLs covering March–August 2026 and stores per-day JSON files.
2. **Stages 2–5 (Embedding, PCA + K-Means, Prototype Averaging, Stiefel Orthogonalization)**:
   - Note that parts of the notebook were originally authored on Google Colab (`google.colab.drive` and `google.colab.files` calls are included for persisting scraped JSONs to Google Drive). When running locally, adjust the data folder path (`folder_path`) to a local directory containing the scraped JSON snippets.
   - Uses `nomic-ai/modernbert-embed-base` via `sentence-transformers` and `pymanopt` on $\mathrm{St}(8, 768)$.
