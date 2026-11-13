# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: myenv (3.9.25)
#     language: python
#     name: python3
# ---

# %% id="0de4409f"
import os
import sys
import json
from pathlib import Path

repo_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from IPython.display import display, Markdown

from narrative_geometry import (
    SPECTRAL_CMAP,
    add_colorbar,
    bootstrap_delta_ci,
    delta,
    detect_ruptures,
    df_to_latex_booktabs,
    draw_chapter_heatmap,
    fmt_ci,
    get_embedding,
    gram_schmidt,
    load_model,
    normalize_rows,
    pairwise_jaccard,
    pairwise_mean_product_matrix,
    plot_narrative_timeseries,
    plot_narrative_vs_events,
    plot_narratives_ruptures,
    procrustes_stiefel,
    resample_constant_weekly,
    run_moving_tests_fdr,
    short_label,
    stiefel_optimize,
    upper_triangle_values,
    weekly_ema,
)

# %% [markdown] id="aeb1cee2"
# # 0. First loading of news dataset + compute embeddings

# %% [markdown] id="50806f8b"
# The idea is to load the dataset and compute the article embeddings once, then store them in a JSON file. Since this involves a significant amount of computation, it’s best to perform it only once.

# %% [markdown] id="ab985929"
# https://www.kaggle.com/datasets/rmisra/news-category-dataset/data
#
# > This dataset contains around 210k news headlines from 2012 to 2022 from HuffPost. This is one of the biggest news datasets and can serve as a benchmark for a variety of computational linguistic tasks. HuffPost stopped maintaining an extensive archive of news articles sometime after this dataset was first collected in 2018, so it is not possible to collect such a dataset in the present day. Due to changes in the website, there are about 200k headlines between 2012 and May 2018 and 10k headlines between May 2018 and 2022.
#
# ```
# @article{misra2022news,
#   title={News Category Dataset},
#   author={Misra, Rishabh},
#   journal={arXiv preprint arXiv:2209.11429},
#   year={2022}
# }
# ```

# %% id="02711806" outputId="61869bc4-83f1-453c-e4f0-b27893f18579"
# Download latest version
path = kagglehub.dataset_download("rmisra/news-category-dataset")

print("Path to dataset files:", path)

# %% id="ff892af2"
# Configuration
MODEL_NAME = "answerdotai/ModernBERT-base"
OUTPUT_JSON = './data_narratives/articles_with_embeddings.json'
BATCH_SIZE = 32

def compute_embeddings_from_kaggle():
    """
    Load articles from Kaggle dataset and compute embeddings.
    Saves results to JSON file.
    """
    # Load model once
    tokenizer, model, device = load_model(MODEL_NAME)

    # Load Kaggle dataset
    fname = '.../.cache/kagglehub/datasets/rmisra/news-category-dataset/versions/3' #adapt to your path
    full_path = os.path.join(fname, 'News_Category_Dataset_v3.json')

    print(f"Loading dataset from: {full_path}")
    df = pd.read_json(full_path, lines=True)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create output directory
    output_dir = Path(OUTPUT_JSON).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process articles and compute embeddings
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing embeddings"):
        # Combine headline and description
        combined_text = (row.get('headline', '') or '') + ' ' + (row.get('short_description', '') or '')
        combined_text = combined_text.strip()

        if not combined_text:
            continue

        # Compute embedding
        embedding = get_embedding(combined_text, tokenizer, model, device)

        # Store result
        results.append({
            'idx': idx,
            'headline': row.get('headline'),
            'short_description': row.get('short_description'),
            'category': row.get('category'),
            'date': str(row.get('date')),
            'link': row.get('link'),
            'combined_text': combined_text,
            'embedding': embedding.astype(float).tolist()
        })

        # Save progressively every 5000 articles
        if (idx + 1) % 5000 == 0:
            print(f"Processed {idx + 1} articles, saving checkpoint...")
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save
    print(f"Saving {len(results)} articles with embeddings...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Completed! Results saved to: {OUTPUT_JSON}")



# %% id="57cfcce4" outputId="0d60753e-14f3-4a44-b87c-d97a02455ee1"
""" Run only once comment the cell when done to avoid re-computation. """
#compute_embeddings_from_kaggle()

# %% [markdown] id="ed745a10"
# # I. Data : news dataset  

# %% [markdown] id="e92907bd"
# Now, we assume that the dataset has been loaded and saved as explained above.

# %% [markdown] id="4d88cd54"
# **I.1 dataset**

# %% id="e9cfb6ae"
""" Loading data """
data = pd.read_json("data_narratives/resampled_articles_with_embeddings.json")

# %% id="18a33bb2" outputId="b97ca942-bd5d-451d-beb9-15a401f78cc4"
""" Count articles in dataset """
data.groupby('date').category.count().plot(
    figsize=(13,4), grid=True, logy=True,
)

# %% id="e7791b32" outputId="c915cd80-b74c-4ae4-bb2f-6bc65ce386ac"
data.category.value_counts()[:7]
data.groupby([data['date'].dt.year, 'category']).size()

# %% id="61f55da4" outputId="cbda10fe-74d3-4f2b-9dc7-55d5c17112f6"
""" display description of dataset in a latex table """

data['year'] = data['date'].dt.year


table = data.groupby(['year', 'category']).size().unstack(fill_value=0)


latex_table = table.to_latex()

print(latex_table)

# %% id="ab74da10" outputId="f91b81ad-af4d-4f34-946f-8e292acc85fd"
""" Count number of articles per category"""
daily_politics_df = data.groupby(['date', 'category']).link.count().unstack()[:'2018-06-01']
daily_politics_df.plot(
    figsize=(13,4), grid=True, logy=False,
)

# %% id="a53d4eca"
""" load the data with embeddings """
data_with_embeddings = pd.read_json("data_narratives/resampled_articles_with_embeddings.json")
#data_with_embeddings.head(5)

# %% [markdown] id="cbe4868a"
# **I.2 Resampling**

# %% [markdown] id="01d07abc"
# We are resampling the dataset so that, the number of articles is the same every week. We are keeping the relative proportions of all categories

# %% id="4ec4e0d1" outputId="b68bf518-a76e-4bca-bf42-c1850ecb99d9"
resampled_data = resample_constant_weekly(data)
resampled_data.head(5)

# %% id="cc5fd0d9" outputId="7e6cb8f2-ed9e-41d2-fa94-34250f8678cc"
""" Count articles in dataset """
resampled_data.groupby('date').category.count().plot(
    figsize=(13,4), grid=True, logy=True,
)

# %% id="5587c3ec" outputId="c379aa94-881f-4b5c-9b32-557dfd98f286"
resampled_data.groupby([resampled_data['date'].dt.year, 'category']).size()

# %% id="085ac854" outputId="e624c78d-5dcc-48e9-ca0f-f8795df9075f"
""" display description of dataset in a latex table """
resampled_data['year'] = resampled_data['date'].dt.year


table = resampled_data.groupby(['year', 'category']).size().unstack(fill_value=0)


latex_table = table.to_latex()

print(latex_table)

# %% [markdown] id="2135626f"
# ## II. Narratives and orthogonalization ##
#

# %% [markdown] id="4b709a1a"
# **I.1 Narratives**

# %% [markdown] id="adf0af82"
# We chose 3 narratives per chapter corresponding to the categories of the news in the dataset

# %% id="62841853"
narratives_by_chapter = {
    "politics": {
        "Trump_as_Disruptor_and_Institutional_Stress":
            "Donald Trump rises to and governs from power by channeling widespread distrust of political elites, media institutions, and bureaucratic authority, positioning himself as a disruptive outsider who challenges democratic norms, centralizes executive power, attacks critics, and reframes institutional resistance as partisan sabotage.",

        "Culture_War_and_Identity_Conflict":
            "American politics between 2012 and 2018 is structured around intensifying culture wars in which abortion, LGBTQ rights, immigration, race, religion, Confederate symbols, gun control, and voting rights become symbolic battlegrounds through which parties mobilize supporters by framing policy disagreements as existential moral conflicts.",

        "Welfare_State_Under_Press":
            "Debates over health care reform, Medicaid, food assistance, drug pricing, refugee policy, and regulatory protections reflect a broader struggle over the future of the American welfare state, as conservative efforts to roll back social protections clash with progressive attempts to defend or expand them in the face of economic insecurity."
    },

    "entertainment": {
        "Entertainment_as_Political_Resistance":
            "Film, television, comedy, and celebrity culture increasingly function as arenas of political contestation, where artists publicly criticize elected officials, late-night shows frame political narratives, awards ceremonies become stages for activism, and cultural production blends entertainment with explicit political messaging.",

        "Franchise_Nostalgia_and_Serial_Dominance":
            "The entertainment industry relies heavily on reboots, sequels, cinematic universes, and anniversary revivals, capitalizing on nostalgia and established intellectual property to secure audience loyalty in an era of streaming competition and franchise-driven box office economics.",

        "Celebrity_Image_Management_in_the_Social_Media_Age":
            "Celebrities navigate a hyper-visible media ecosystem in which personal relationships, identity declarations, scandals, and social media controversies become central to their public personas, requiring constant image management amid shifting norms around gender, race, and accountability."
    },

    "parenting": {
        "The_Overwhelmed_Modern_Parent":
            "Parenthood in the 2012–2018 period is portrayed as emotionally intense and structurally unsupported, with working parents—especially mothers—struggling to balance careers, childcare costs, sleep deprivation, and social expectations while navigating guilt, burnout, and economic pressure.",

        "Intensive_and_Expert_Guided_Parenting":
            "Raising children becomes an increasingly expert-driven project in which parents rely on scientific studies, psychological advice, educational technologies, and developmental optimization strategies to manage screen time, stress, discipline, mental health, and academic performance.",

        "Raising_Children_in_a_Politicized_Society":
            "Parents confront a social environment shaped by political polarization, gun violence debates, racial tensions, immigration fears, and changing gender norms, prompting concerns about children’s safety, values formation, civic identity, and exposure to divisive public discourse."
    },
    "travel": {
        "Experiential_and_Transformational_Travel":
            "Travel is framed as a path to self-discovery and cultural immersion, encouraging encounters with remote landscapes, wildlife, historic sites, and local communities as a way to cultivate authenticity, global awareness, and personal growth in an interconnected world.",

        "Travel_Optimization_and_Consumer_Strategy":
            "Modern travel is treated as a strategic consumer activity, with emphasis on cheapest destinations, loyalty programs, airline hacks, peak-season timing, gadget recommendations, hotel rankings, and efficiency tips that promise to maximize value, comfort, and status.",

        "Aesthetic_and_Instagrammable_Destinations":
            "Destinations are increasingly valued for their visual spectacle and shareability, as pink beaches, dramatic glaciers, themed hotels, architectural landmarks, and unusual experiences become curated backdrops for digital photography and social media performance."
    }
}


# %% [markdown] id="8e32caef"
# **II.2 Orthogonalization**

# %% [markdown] id="8dbe2224"
# Our goal is to find a representative $V_i$ of each normalized narrative's embedding $N_i$. We are adding the mean of (N_i)_i as a last "narrative". Thus, we solve the following optimization under constraints problem:
#
# $$ \text{argmax}_{(V_1, ..., V_n)} \text{min}_{i} N_i^TV_i$$
# $$s.t, ||V_i|| \le 1, N_i^TV_i \ge 0, V_i^TV_j = 0$$
#
# Functions `gram_schmidt`, `stiefel_optimize`, and `procrustes_stiefel` are imported from `narrative_geometry.manifold`.

# %% [markdown] id="19e333a4"
# # III. Compute the narratives and orthogonal representatives

# %% [markdown] id="d99c0a3b"
# The narratives need to be normalized before the orthogonalization, bacause we are optimizing on the dot product thus, if the norm of one $N_i$ is arbitrarly bigger than the others, this is inducing a bias. As our metric will be cosine similarity, we don't care about the norm.

# %% id="1ef1226f"
chapter_names = narratives_by_chapter.keys()

# %% id="387346ca" outputId="e1cf9e65-c234-4a10-d6c9-cf2fa4a9686e"
""" Compute narratives embeddings and solve orthogonality problem """
tokenizer, model, device = load_model("answerdotai/ModernBERT-base")

narrative_rows = []
for chapter, narratives in narratives_by_chapter.items():
    for title, text in narratives.items():
        narrative_rows.append({"chapter": chapter, "title": title, "text": text})

narrative_matrix_unorm = np.array([get_embedding(row["text"], tokenizer, model, device) for row in narrative_rows])
narrative_matrix = normalize_rows(narrative_matrix_unorm)

""" normalize each narrative"""
print(narrative_matrix.shape)

orthogonal_matrix, score = stiefel_optimize(narrative_matrix)

# %% id="97cb50b3"
narratives = pd.DataFrame([{**narrative_rows[i], "embedding": narrative_matrix[i]} for i in range(len(narrative_rows))])
narratives_orth = pd.DataFrame([{"chapter": narrative_rows[i]["chapter"], "title": f"{narrative_rows[i]['title']}_orth", "text": narrative_rows[i]["text"], "embedding": orthogonal_matrix[i]} for i in range(len(narrative_rows))])

os.makedirs("./data_narratives", exist_ok=True)
narratives.to_csv("./data_narratives/narratives.csv", index=False)
narratives_orth.to_csv("./data_narratives/narratives_orth.csv", index=False)

# %% id="d4720eff"
narrative_names = (narratives.chapter + ':' + narratives.title).to_list()

orthogonal_names = (narratives_orth.chapter + ':' + narratives_orth.title).to_list()


# %% [markdown] id="fdae85ef"
# **III.2 Visualization**

# %% id="f38927c2" outputId="187c114b-8d04-41d3-906b-50c5e459df38"
cosine_sim_all = cosine_similarity(np.vstack([narrative_matrix, orthogonal_matrix]))

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

CHAPTER_PAL = {
    "politics":      "#4878CF",
    "entertainment": "#6ACC65",
    "parenting":     "#D65F5F",
    "travel":        "#B47CC7",
}

_ABBREV = {
    "politics":      "Politics",
    "entertainment": "Enter",
    "parenting":     "Parenting",
    "travel":        "travel",
}

chapters      = narratives.chapter.to_list()
titles_raw    = narratives.title.to_list()
chapter_names = list(narratives_by_chapter.keys())
chapter_size  = len(narratives_by_chapter["politics"])  # 3
n             = len(narratives_by_chapter)              # 12

tick_labels = [short_label(t) for t in titles_raw]

sim_NN    = cosine_similarity(narrative_matrix,  narrative_matrix)
sim_N_ON  = cosine_similarity(narrative_matrix,  orthogonal_matrix)
sim_ON_ON = cosine_similarity(orthogonal_matrix, orthogonal_matrix)
sim_GS    = cosine_similarity(gram_schmidt(narrative_matrix), narrative_matrix)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  (a) N × N   |   (b) ON × ON  [Stiefel]
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(
    1, 2,
    figsize=(17, 8),
    gridspec_kw={"wspace": 0.65}
)

panels_fig1 = [
    (axes1[0], sim_NN,    tick_labels, tick_labels,
     "(a)  Narratives  ×  Narratives",
     0,  1),
    (axes1[1], sim_ON_ON, tick_labels, tick_labels,
     "(b)  Orthogonal Narratives  ×  Orthogonal Narratives  [Stiefel]",
     -1, 1),
]

for ax, mat, rl, cl, title, vmin, vmax in panels_fig1:
    im = draw_chapter_heatmap(
        ax, mat, rl, cl, title,
        chapter_names=chapter_names, chapter_size=chapter_size,
        chapter_pal=CHAPTER_PAL, chapter_abbrev=_ABBREV,
        vmin=vmin, vmax=vmax,
    )
    add_colorbar(fig1, im, ax, vmin, vmax)

fig1.suptitle(
    "Cosine Similarity Matrices — Raw Narratives and Stiefel Orthogonal Basis",
    fontsize=12, fontweight="bold", y=1.02
)
fig1.savefig("fig1_N_and_ON.pdf", bbox_inches="tight")
fig1.savefig("fig1_N_and_ON.png", bbox_inches="tight", dpi=300)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  (a) N × ON  [Stiefel]   |   (b) GS-ON × N  [Gram-Schmidt]
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(
    1, 2,
    figsize=(17, 8),
    gridspec_kw={"wspace": 0.65}
)

panels_fig2 = [
    (axes2[0], sim_N_ON,  tick_labels, tick_labels,
     "(a)  Narratives  ×  Orthogonal Narratives  [Stiefel]",
     -1, 1),
    (axes2[1], sim_GS.T,  tick_labels, tick_labels,
     "(b)  Gram-Schmidt Orthogonal Narratives  ×  Narratives",
     -1, 1),
]

for ax, mat, rl, cl, title, vmin, vmax in panels_fig2:
    im = draw_chapter_heatmap(
        ax, mat, rl, cl, title,
        chapter_names=chapter_names, chapter_size=chapter_size,
        chapter_pal=CHAPTER_PAL, chapter_abbrev=_ABBREV,
        vmin=vmin, vmax=vmax,
    )
    add_colorbar(fig2, im, ax, vmin, vmax)

fig2.suptitle(
    "Cosine Similarity Matrices — Cross Similarities and Gram-Schmidt Comparison",
    fontsize=12, fontweight="bold", y=1.02
)
fig2.savefig("fig2_cross_and_GS.pdf", bbox_inches="tight")
fig2.savefig("fig2_cross_and_GS.png", bbox_inches="tight", dpi=300)

plt.show()
print("✓  Two figures saved as PDF and PNG.")

# %% [markdown] id="6fd7eefd"
# **III.3 Example with other narratives**

# %% [markdown] id="2cfe922e"
# Let's choose narratives that are independent from our data base to put in evidence the interest of our method compared to Gram Schmidt method

# %% id="d795e8ad"
narratives_by_chapter2 = {
    "climate": {
        "Extreme_Heat_and_Emergency_Policy_Response":
            "Record-breaking temperatures across southern Europe this summer triggered unprecedented heatwaves, forcing governments to activate emergency climate plans and accelerate the transition toward renewable energy sources in order to meaningfully reduce carbon emissions and limit further global warming.",

        "International_Climate_Commitments_Under_Pressure":
            "As climate-related disasters multiply worldwide, the United Nations urgently calls for stronger adherence to the Paris Agreement, emphasizing that industrialized nations must deliver substantial and verifiable reductions in greenhouse gas emissions before irreversible climate tipping points are reached.",

        "Rising_Seas_and_Coastal_Vulnerability":
            "The accelerating melting of Arctic glaciers, driven by sustained global warming, is causing a gradual but measurable rise in sea levels that threatens tens of millions of people living in low-lying coastal zones, according to a major scientific report released this week."
    },

    "artificial_intelligence": {
        "AI_Disruption_of_Labor_Markets":
            "The rapid rise of generative artificial intelligence models is profoundly reshaping global labor markets, prompting technology companies to invest billions of dollars in autonomous systems capable of gradually replacing skilled workers across a wide range of industries and professional sectors.",

        "Big_Tech_Race_and_Regulatory_Pushback":
            "Silicon Valley's leading technology firms are intensifying their competition in artificial intelligence by unveiling increasingly powerful language models, while European regulators work to constrain these developments through strict legislative frameworks designed to protect citizens' fundamental rights and digital freedoms.",

        "AI_Ethics_Privacy_and_Public_Sector_Governance":
            "Recent breakthroughs in artificial intelligence are raising critical ethical concerns around personal data privacy and cybersecurity, prompting several national governments to establish dedicated regulatory frameworks governing the deployment of these technologies within public institutions and essential services."
    },

    "geopolitics": {
        "NATO_Eastern_Flank_and_the_Russia_Ukraine_War":
            "The ongoing military conflict between Russia and Ukraine continues to destabilize the European geopolitical order, compelling NATO allies to reinforce their eastern flank with additional troops and to significantly increase national defense budgets in response to sustained and escalating Russian aggression.",

        "Middle_East_Conflict_and_Humanitarian_Crisis":
            "The armed conflict in the Middle East has triggered a humanitarian catastrophe of alarming proportions, prompting urgent international calls for an immediate ceasefire and the rapid deployment of emergency aid to civilian populations trapped in active combat zones across the region.",

        "US_China_Rivalry_and_Indo_Pacific_Tensions":
            "Growing geopolitical rivalry between Washington and Beijing over the status of Taiwan is seriously threatening stability across the Indo-Pacific region, with both superpowers escalating military demonstrations and economic sanctions in what increasingly resembles a renewed cold war confrontation."
    },

    "economics": {
        "Inflation_Interest_Rates_and_Household_Strain":
            "Persistent inflation across the world's major economies is forcing central banks to maintain elevated interest rates, placing significant pressure on economic growth and disproportionately burdening lower-income households struggling to absorb the rising cost of everyday goods and essential services.",

        "Financial_Market_Volatility_and_Fed_Signaling":
            "Global financial markets experienced sharp volatility this week as investors reacted to the latest monetary policy signals from the United States Federal Reserve, triggering significant fluctuations across major stock indices and a notable depreciation of several emerging market currencies.",

        "IMF_Forecasts_and_Global_Growth_Slowdown":
            "Global economic growth is decelerating at a concerning pace according to the latest International Monetary Fund projections, which identify trade tensions, rising public debt levels, and slowing emerging market performance as the primary drivers behind deteriorating worldwide economic prospects."
    }
}

# %% id="fc89216c" outputId="ed483d06-f752-47da-8f59-21c883a454ec"
""" Compute narratives embeddings and solve orthogonality problem """
tokenizer, model, device = load_model("answerdotai/ModernBERT-base")

rows = []
for chapter, narratives_dict in narratives_by_chapter2.items():
    for title, text in narratives_dict.items():
        rows.append({"chapter": chapter, "title": title, "text": text})

narrative_matrix_unorm = np.array([get_embedding(row["text"], tokenizer, model, device) for row in rows])

""" normalize each narrative"""
narrative_matrix2 = normalize_rows(narrative_matrix_unorm)

orthogonal_matrix2, score = stiefel_optimize(narrative_matrix2)

#orthogonal_matrix_GS2 = gram_schmidt(narrative_matrix2)

orthogonal_matrix_GS2 = procrustes_stiefel(narrative_matrix2)

# %% id="95527831" outputId="06b32717-c2c2-4cca-efb3-90ab10060f08"

CHAPTER_PAL = {
    "climate":                 "#4878CF",
    "artificial_intelligence": "#6ACC65",
    "geopolitics":             "#D65F5F",
    "economics":               "#B47CC7",
}

_ABBREV = {
    "climate":                 "Climate",
    "artificial_intelligence": "AI",
    "geopolitics":             "Geopolitics",
    "economics":               "Economics",
}

chapters      = [1 for row in rows]
titles_raw    = [title for narratives_dict in narratives_by_chapter2.values() for title in narratives_dict.keys()]
chapter_names = list(narratives_by_chapter2.keys())
chapter_size  = 3
n             = len(rows)   # 12

tick_labels = [short_label(t) for t in titles_raw]

sim_NN    = cosine_similarity(narrative_matrix2,  narrative_matrix2)
sim_N_ON  = cosine_similarity(narrative_matrix2,  orthogonal_matrix2)
sim_ON_ON = cosine_similarity(orthogonal_matrix2, orthogonal_matrix2)
sim_GS    = cosine_similarity(gram_schmidt(narrative_matrix2), narrative_matrix2)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  (a) N × N   |   (b) ON × ON  [Stiefel]
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(
    1, 2,
    figsize=(17, 8),
    gridspec_kw={"wspace": 0.65}
)

panels_fig1 = [
    (axes1[0], sim_NN,    tick_labels, tick_labels,
     "(a)  Narratives  ×  Narratives",
     0,  1),
    (axes1[1], sim_ON_ON, tick_labels, tick_labels,
     "(b)  Orthogonal Narratives  ×  Orthogonal Narratives  [Stiefel]",
     -1, 1),
]

for ax, mat, rl, cl, title, vmin, vmax in panels_fig1:
    im = draw_chapter_heatmap(
        ax, mat, rl, cl, title,
        chapter_names=chapter_names, chapter_size=chapter_size,
        chapter_pal=CHAPTER_PAL, chapter_abbrev=_ABBREV,
        vmin=vmin, vmax=vmax,
    )
    add_colorbar(fig1, im, ax, vmin, vmax)

fig1.suptitle(
    "Cosine Similarity Matrices — Raw Narratives and Stiefel Orthogonal Basis",
    fontsize=12, fontweight="bold", y=1.02
)
fig1.savefig("fig1_N_and_ON.pdf", bbox_inches="tight")
fig1.savefig("fig1_N_and_ON.png", bbox_inches="tight", dpi=300)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  —  (a) N × ON  [Stiefel]   |   (b) GS-ON × N  [Gram-Schmidt]
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(
    1, 2,
    figsize=(17, 8),
    gridspec_kw={"wspace": 0.65}
)

panels_fig2 = [
    (axes2[0], sim_N_ON,  tick_labels, tick_labels,
     "(a)  Narratives  ×  Orthogonal Narratives  [Stiefel]",
     -1, 1),
    (axes2[1], sim_GS.T,  tick_labels, tick_labels,
     "(b)  Gram-Schmidt Orthogonal Narratives  ×  Narratives",
     -1, 1),
]

for ax, mat, rl, cl, title, vmin, vmax in panels_fig2:
    im = draw_chapter_heatmap(
        ax, mat, rl, cl, title,
        chapter_names=chapter_names, chapter_size=chapter_size,
        chapter_pal=CHAPTER_PAL, chapter_abbrev=_ABBREV,
        vmin=vmin, vmax=vmax,
    )
    add_colorbar(fig2, im, ax, vmin, vmax)

fig2.suptitle(
    "Cosine Similarity Matrices — Cross Similarities and Gram-Schmidt Comparison",
    fontsize=12, fontweight="bold", y=1.02
)
fig2.savefig("fig2_cross_and_GS.pdf", bbox_inches="tight")
fig2.savefig("fig2_cross_and_GS.png", bbox_inches="tight", dpi=300)

plt.show()
print("✓  Two figures saved as PDF and PNG.")

# %% [markdown] id="699346c0"
# This is just an example as we are not using those narratives for our main experiments

# %% [markdown] id="e59a9987"
# # IV Main computations: cosine similarities with $N_i$, $V_i$

# %% [markdown] id="81158e2d"
# Here we are:
#
# 1. Computing the embeddings of all articles in our news dataset, and then obtaining a collection $(X_n)_n$ of article embeddings;
# 2. Computing the cosine similarities of :
#
#     a) Article embeddings with raw narratives $N_i$: $cos(N_i, X_n)$
#
#     b) Article embeddings with orthogonalized narratives $V_i$: $cos(V_i, X_n)$

# %% id="1f33174e" outputId="10704cbd-4560-4436-a75f-2fedc2e4de40" colab={"referenced_widgets": ["ce4754a1b4a2429292a7e83934b70f32"]}
"""Create dataframe with the cosine similarity with orthogonal narratives"""
from tqdm.notebook import tqdm
from sklearn.metrics.pairwise import cosine_similarity

print("Computing cosine similarities with narratives and orthogonal narratives...")


time_series_data = []
gdir = './data_narratives'

for idx, row in tqdm(resampled_data.iterrows(), total=len(resampled_data)):
    # Get article embedding
    article_embedding = row['embedding']
    if isinstance(article_embedding, list):
        article_embedding = np.array(article_embedding)

    info_similarities = {
        'date': row['date'],
        'category': row['category'],
        'headline': row['headline'],
        'link': row.get('link', '')
    }

    # Compute cosine similarities with narratives (vectorized)
    similarities_narratives = cosine_similarity(
        article_embedding.reshape(1, -1),
        narrative_matrix
    )[0]

    for name, sim in zip(narrative_names, similarities_narratives):
        info_similarities[f"{name}"] = float(sim)

    # Compute cosine similarities with orthogonal narratives (vectorized)
    similarities_orthogonal = cosine_similarity(
        article_embedding.reshape(1, -1),
        orthogonal_matrix
    )[0]

    for name, sim in zip(orthogonal_names, similarities_orthogonal):
        info_similarities[f"{name}"] = float(sim)


    time_series_data.append(info_similarities)

# Save to CSV
ts_df = pd.DataFrame(time_series_data)
filepath = os.path.join(gdir, 'narrative_exposures_with_orthogonal.csv')
os.makedirs(gdir, exist_ok=True)
ts_df.to_csv(filepath, index=False)
print(f"✓ Saved to: {filepath}")
print(f"Shape: {ts_df.shape}")
print(f"Columns: {ts_df.columns.tolist()[:10]}...")  # Show first 10 columns

# %% [markdown] id="c2eeafdd"
# # V Visualization

# %% [markdown] id="b0a05cba"
# **V.1 All new, all narratives**

# %% [markdown] id="b890cd98"
# 1) Narratives $N_i$

# %% id="74b7a1f2" outputId="f2a28afc-7015-402c-9b07-9505ffdbbc5b"
""" Prepare data for visualization """
ts_df['date_dt'] = pd.to_datetime(ts_df['date'])

# Extract narrative columns (those with ':' in the name)
all_cols = [col for col in ts_df.columns if ':' in col]

# Separate narratives and orthogonal narratives
narrative_cols = [col for col in all_cols if '_orth' not in col and '_comple' not in col]
orthogonal_cols = [col for col in all_cols if '_orth' in col]
residuals_cols = [col for col in all_cols if '_comple' in col]
print(len(narrative_cols))

# Get unique news categories
news_categories = sorted(ts_df['category'].unique())


# %% id="fbb8fea9" outputId="dfdbb8d2-375c-4e01-a207-4d971fb9a419"
ts_df["date_dt"] = pd.to_datetime(ts_df["date"])

all_cols = [col for col in ts_df.columns if ":" in col]
narrative_cols = [col for col in all_cols if "_orth" not in col and "_comple" not in col]
orthogonal_cols = [col for col in all_cols if "_orth" in col]
residuals_cols = [col for col in all_cols if "_comple" in col]

# tous les chapitres présents dans les narratifs (pas de filtrage)
all_chapters = sorted(set(col.split(":")[0] for col in narrative_cols))
print(f"Chapters found : {all_chapters}")

# %% id="96281f20"
# plot_narrative_timeseries is imported from narrative_geometry


# %% id="f2ab08cd" outputId="9a2a814a-dda6-469f-ae2c-abc11c9c6c76"
""" Raw narratives, all chapters, EMA span=4 weeks, 2014–2018 """
plot_narrative_timeseries(
    narrative_cols, all_chapters, ts_df,
    title="Raw narratives $\\hat{p}$ (2014–2018, EMA span=4)",
    start="2014-07-01", end="2018-07-31"
)

# %% id="74d4291e" outputId="b45e4ba4-6d4c-4554-afa3-db84df3d47ff"
""" Orthogonal narratives, all chapters, EMA span=4 weeks, 2014–2018 """
plot_narrative_timeseries(
    orthogonal_cols, all_chapters, ts_df,
    strip_suffix="_orth",
    title="Orthogonal narratives $\\hat{p}$ (2014–2018, EMA span=4)",
    start="2014-07-01", end="2018-07-31"
)

# %% id="413b3cef" outputId="89b5069e-0e4f-4362-debd-3ac2fd48eafc"
selected_chapters = ["politics", "entertainment"]

events = {
    "2015-06-16": "Trump announces his campaign",
    "2016-07-19": "Trump as the Rep candidate",
    "2016-09-26": "First presidential debate",
    "2016-11-08": "Trump election",
    "2017-01-20": "Trump inauguration",
    "2017-03-06": "Attempt to repeal Obamacare",
}

""" Raw narratives, selected chapters, EMA span=4 weeks, 2014–2018 """
plot_narrative_timeseries(
    narrative_cols, selected_chapters, ts_df,
    title="Raw narratives $\\hat{p}$ (2014–2018, EMA span=4)",
    start="2014-07-01", end="2018-07-31", events=events
)

""" Orthogonal narratives, selected chapters, EMA span=4 weeks, 2014–2018 """
plot_narrative_timeseries(
    orthogonal_cols, selected_chapters, ts_df,
    strip_suffix="_orth",
    title="Orthogonal narratives $\\hat{p}$ (2014–2018, EMA span=4)",
    start="2014-07-01", end="2018-07-31", events=events
)

# %% [markdown] id="39a2635d"
# # VI. Tables

# %% [markdown] id="f81748c1"
# **VI.1 Table 1: pairwise narrative covariation before and after orthogonalization**

# %% id="482748c8" outputId="d0255eb6-bcb0-4dbb-cfdd-58f215bead2b"
# ============================================================
# Mean over time of pairwise products:
#   M_N[i,j] = mean_t Q_N[i](t) * Q_N[j](t)
#   M_V[i,j] = mean_t Q_V[i](t) * Q_V[j](t)
# Displayed as descriptive statistics (not the raw 8x8 matrices),
# with ready-to-paste LaTeX.
# ============================================================

ts = ts_df.copy()
ts["date_dt"] = pd.to_datetime(ts["date"])

# ------------------------------------------------------------
# 1) Find matched N / V narrative columns
# ------------------------------------------------------------
all_cols = [c for c in ts.columns if ":" in c]
cols_N_raw = [c for c in all_cols if "_orth" not in c and "_comple" not in c]

col_N_to_V = {}
for col_n in cols_N_raw:
    chapter, name = col_n.split(":", 1)
    col_v = f"{chapter}:{name}_orth"
    if col_v in ts.columns:
        col_N_to_V[col_n] = col_v

cols_N = sorted(col_N_to_V.keys(), key=lambda c: (c.split(":", 1)[0], c.split(":", 1)[1]))
cols_V = [col_N_to_V[c] for c in cols_N]

if len(cols_N) == 0:
    raise ValueError("No matched N / V narrative pairs were found in ts_df.")

base_labels = cols_N.copy()

ts[cols_N] = ts[cols_N].apply(pd.to_numeric, errors="coerce").fillna(0.0)
ts[cols_V] = ts[cols_V].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# ------------------------------------------------------------
# 2) Weekly aggregation
# ------------------------------------------------------------
N_total = len(ts)
weekly = ts.groupby(pd.Grouper(key="date_dt", freq="W-MON"))

Q_N = weekly[cols_N].sum() / N_total
Q_V = weekly[cols_V].sum() / N_total

if len(Q_N.index) > 0:
    full_index = pd.date_range(start=Q_N.index.min(), end=Q_N.index.max(), freq="W-MON")
    Q_N = Q_N.reindex(full_index, fill_value=0.0)
    Q_V = Q_V.reindex(full_index, fill_value=0.0)

Q_N = Q_N.fillna(0.0)
Q_V = Q_V.fillna(0.0)

if Q_N.shape[0] == 0:
    raise ValueError("No weekly observations found after aggregation.")

# ------------------------------------------------------------
# 3) Pairwise mean product matrices (from narrative_geometry)
# ------------------------------------------------------------
M_N = pairwise_mean_product_matrix(Q_N, base_labels).fillna(0.0)
M_V = pairwise_mean_product_matrix(Q_V, base_labels).fillna(0.0)

# ------------------------------------------------------------
# 4) Rescale by 1e-7 for display
# ------------------------------------------------------------
scale = 1e-7
M_N_disp = M_N / scale
M_V_disp = M_V / scale

# ------------------------------------------------------------
# 5) Descriptive statistics table (main text) + LaTeX export
# ------------------------------------------------------------
offdiag_N = upper_triangle_values(M_N_disp)
offdiag_V = upper_triangle_values(M_V_disp)

stats_table = pd.DataFrame(
    {
        "N (raw)": [
            np.mean(offdiag_N), np.std(offdiag_N), np.median(offdiag_N),
            np.min(offdiag_N), np.max(offdiag_N),
        ],
        "V (purified)": [
            np.mean(offdiag_V), np.std(offdiag_V), np.median(offdiag_V),
            np.min(offdiag_V), np.max(offdiag_V),
        ],
    },
    index=["mean", "std", "median", "min", "max"],
).round(3)

display(Markdown(r"### Descriptive statistics — covariation between narrative pairs ($\times 10^{-7}$)"))
display(stats_table)

latex_code = df_to_latex_booktabs(
    stats_table,
    caption=r"Descriptive statistics of pairwise narrative covariation ($\times 10^{-7}$), before (N) and after (V) orthogonalization.",
    label="tab:covariation_stats",
)
print(latex_code)


# %% [markdown] id="5e08288d"
# **VI.2 Table 2: to assess semantic specificity**

# %% id="d9d0b0f6"
# delta and bootstrap_delta_ci are imported from narrative_geometry
boot_rows = []
n_boot = 1000
frac = 1
for chapter in narratives_by_chapter.keys():
    for narrative in narratives_by_chapter[chapter].keys():
        stats_boot = bootstrap_delta_ci(ts_df, chapter, narrative, n_bootstrap=n_boot, frac=frac, alpha=0.05)
        boot_rows.append({"chapter": chapter, "narrative": narrative, **stats_boot})

df_boot = pd.DataFrame(boot_rows)


# %% id="176561a7" outputId="aa39ded4-5aaa-41ba-ef55-085f99e28a4f"
# ------------------------------------------------------------
# Paper-ready display: mean [CI lo, CI hi], significance flagged
# ------------------------------------------------------------
display_rows = []
for _, row in df_boot.iterrows():
    display_rows.append({
        "Chapter": "Right" if "droite" in row["chapter"] else "Left" if "gauche" in row["chapter"] else row["chapter"],
        "Narrative": row["narrative"].replace("_", " ").capitalize(),
        r"$\Delta^i(N_i)$": fmt_ci(row["mean_delta_N"], row["ci_delta_N_low"], row["ci_delta_N_high"]),
        r"$\Delta^i(V_i)$": fmt_ci(row["mean_delta_V"], row["ci_delta_V_low"], row["ci_delta_V_high"]),
        r"$\Delta_{\text{gain}}$": fmt_ci(row["mean_gain"], row["ci_gain_low"], row["ci_gain_high"]),
    })

boot_display = pd.DataFrame(display_rows)

display(Markdown(f"### Sensitivity analysis — {n_boot} bootstrap draws, {int(frac*100)}% of articles, 95% CI"))
display(boot_display)

# %% [markdown] id="9976d3cb"
# **VI.3 Rupture test**

# %% id="a2f9f44a"

TRUMP_CAMPAIGN_EVENTS = [
    ("2015-06-16", "Campaign announcement",
     "Trump officially announces his candidacy for the Republican presidential nomination from Trump Tower in New York City."),

    ("2015-08-06", "First Republican debate",
     "First televised Republican primary debate hosted by Fox News; Trump engages in a widely publicized exchange with journalist Megyn Kelly."),

    ("2015-12-07", "Proposal to ban Muslim entry",
     "Trump calls for a total suspension of Muslims entering the United States, triggering nationwide controversy."),

    ("2016-02-01", "Iowa caucus defeat",
     "Trump finishes second behind Ted Cruz in the first contest of the Republican primaries."),

    ("2016-03-15", "Super Tuesday II and Marco Rubio's withdrawal",
     "Trump wins several primaries, including Florida; Marco Rubio suspends his campaign after losing in his home state."),

    ("2016-05-04", "Ted Cruz and John Kasich withdraw",
     "Following Trump's victory in Indiana on May 3, Ted Cruz suspends his campaign, followed by John Kasich the next day; Trump becomes the only active Republican candidate."),

    ("2016-05-26", "Delegate threshold reached",
     "The Associated Press announces that Trump has secured enough delegates to guarantee his nomination as the Republican presidential candidate."),

    ("2016-08-01", "Controversy over Humayun Khan's parents",
     "Trump's remarks targeting the parents of Captain Humayun Khan, a Muslim American soldier killed in Iraq, trigger strong political and media reactions."),

    ("2016-09-26", "First presidential debate",
     "First televised debate between Donald Trump and Hillary Clinton, covering issues including the economy, national security, and the candidates' policy positions."),

    ("2016-10-07", "Release of the Access Hollywood recording",
     "The release of a 2005 recording reveals Trump's vulgar remarks about women and triggers a major political and media crisis."),

    ("2016-10-28", "James Comey's letter to Congress",
     "FBI Director James Comey informs Congress that the FBI is examining additional emails potentially related to the investigation into Hillary Clinton's email practices, reigniting controversy days before the election."),

    ("2016-11-08", "Election Day",
     "Donald Trump wins the 2016 United States presidential election, defeating Democratic nominee Hillary Clinton."),

    ("2017-01-20", "Inauguration",
     "Donald Trump is inaugurated as the 45th President of the United States in a ceremony held at the United States Capitol.")
]

# %% id="821b45bf"
# Article embeddings already available in the notebook
embeddings = [np.asarray(e, dtype=float) for e in data_with_embeddings["embedding"]]
dates = pd.to_datetime(data_with_embeddings["date"])

narrative_labels = [f"{r['chapter']}:{r['title']}" for r in narrative_rows]  # même ordre que narrative_matrix / orthogonal_matrix

def get_narrative_vectors(label):
    idx = narrative_labels.index(label)
    return narrative_matrix[idx], orthogonal_matrix[idx]

t_begin = pd.Timestamp("2014-07-01")
t_end = pd.Timestamp("2018-07-31")

# %% id="fcbe1cff"
# run_moving_tests_fdr, detect_ruptures, weekly_ema are imported from narrative_geometry

# %% id="1e2b7e4e" outputId="4252a989-efb0-4474-ceb5-acbf22278711"
# All narratives, one subplot each (the Trump narrative is the first panel)
all_ruptures = plot_narratives_ruptures(
    embeddings, dates, narrative_labels, t_begin, t_end,
    get_narrative_vectors=get_narrative_vectors,
    bandwidth_days=90, alpha=0.05, trim_days=90, span_weeks=4,
    events=TRUMP_CAMPAIGN_EVENTS,
)


# %% id="cc3c7041" outputId="614ca823-dd37-4cdd-8692-d5e9c8481651"
plot_narrative_vs_events(
    embeddings, dates, "politics:Trump_as_Disruptor_and_Institutional_Stress",
    TRUMP_CAMPAIGN_EVENTS, t_begin, t_end,
    get_narrative_vectors=get_narrative_vectors,
)


# %% id="d85d70b1"
def analyze_narrative(embeddings, dates, label, t_begin, t_end, bandwidth_days=90, alpha=0.05, trim_days=90):
    """Une seule passe par narratif -- calcule N et V en même temps, pas de doublon."""
    N_vec, V_vec = get_narrative_vectors(label)
    results = run_moving_tests_fdr(embeddings, dates, N_vec, V_vec, t_begin, t_end, bandwidth_days=bandwidth_days)
    df_N, ruptures_N = detect_ruptures(results, test_col="p_delta_N", alpha=alpha, trim_days=trim_days)
    df_V, ruptures_V = detect_ruptures(results, test_col="p_delta_V", alpha=alpha, trim_days=trim_days)
    return {
        "sig_N": df_N.set_index("date")["significant"],
        "sig_V": df_V.set_index("date")["significant"],
        "n_ruptures_N": len(ruptures_N),
        "n_ruptures_V": len(ruptures_V),
    }


analyses = {label: analyze_narrative(embeddings, dates, label, t_begin, t_end) for label in narrative_labels}

sig_N = pd.DataFrame({label: a["sig_N"] for label, a in analyses.items()})
sig_V = pd.DataFrame({label: a["sig_V"] for label, a in analyses.items()})

breadth_N = sig_N.sum(axis=1)
breadth_V = sig_V.sum(axis=1)


# %% id="c18e4825" outputId="8296709e-539f-477f-e316-48ef12fad7cc"
jaccard_N = pairwise_jaccard(sig_N)
jaccard_V = pairwise_jaccard(sig_V)

offdiag_N = upper_triangle_values(jaccard_N)
offdiag_V = upper_triangle_values(jaccard_V)

# Tableau de synthèse (même esprit que celui de la covariation)
jaccard_stats = pd.DataFrame({
    "N (brut)": [np.nanmean(offdiag_N), np.nanmedian(offdiag_N), np.nanmax(offdiag_N)],
    "V (purifié)": [np.nanmean(offdiag_V), np.nanmedian(offdiag_V), np.nanmax(offdiag_V)],
}, index=["Jaccard mean", "Jaccard median", "Jaccard max"]).round(3)

display(Markdown("### Overlap between detected ruptures by pairs of narratives (Jaccard index)"))
display(jaccard_stats)

counts_table = pd.DataFrame({
    "N (brut)": {label: a["n_ruptures_N"] for label, a in analyses.items()},
    "V (purifié)": {label: a["n_ruptures_V"] for label, a in analyses.items()},
})
display(Markdown("### Number of detected ruptures per narrative (N vs V)"))
display(counts_table)
