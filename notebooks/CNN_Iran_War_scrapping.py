# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="TiQoXTCQ6WrY"
# # STEP 1. Scrapping CNN War in Iran updates
#
# 1. thanks to Perplexity for the URL list (Claude has been very bad at that)
# 2. thanks to Safari Dev URL explorer to detect the HTML TAG to focus on
# 3. thanks to DeepSeek for the scrapping snippet (as usual: compact and efficient)

# %% [markdown] id="1Y3Ix8zQjfKt"
# ## Configuration & Future update
#
# Set `SCRAPE_CNN_WEB_PAGES = True` below to scrape the CNN live-news web pages listed in `resources/cnn_news_pages.json`, or `False` to load the freshest `cnn_news_<date>.json` directly from storage (Google Drive if available, or `resources/` locally).
#
# To be future proof, note this new pattern:
# - https://www.cnn.com/2026/08/25/world/live-news/iran-war-trump-sanctions
#
# The last date I ran this code (for a paper) was `2026-08-19`

# %% id="-GgAk238zW3y"
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import HTML, display
from tqdm.notebook import tqdm, trange

# Ensure the repository root is on sys.path so narrative_geometry can be imported
for _candidate in [Path.cwd().resolve(), Path.cwd().resolve().parent]:
    if (_candidate / "narrative_geometry").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from narrative_geometry import (
    average_cluster_prototypes,
    decompose_narratives,
    find_freshest_dated_file,
    get_resources_dir,
    gram_schmidt,
    normalize_rows,
    resolve_storage_dir,
    save_and_download_figure,
    stiefel_optimize,
)

# Top-of-notebook flag:
# - True: retrieve news from CNN web pages and save cnn_news_<today>.json
# - False: load the freshest cnn_news_<date>.json from Google Drive or resources/
SCRAPE_CNN_WEB_PAGES = False

# %% [markdown] id="02nCn_Id6TIC"
# # Load consolidated URL list from `resources/cnn_news_pages.json`

# %% id="kt39ynRkHT_F"
resources_dir = get_resources_dir()
cnn_pages_path = resources_dir / "cnn_news_pages.json"

with open(cnn_pages_path, "r") as f:
    updated_url = json.load(f)

cnn_iran_war_urls = updated_url
print(f"Loaded {len(updated_url)} CNN URLs from {cnn_pages_path}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["7dbf27a8589c4030b502591ea3abfc10", "d3e07e2192f5491db071958666555c2d", "bab09fbbf5414e07bcd21beab16b2aac", "a9354553cf9b4fc6a7afabcc56aa4cad", "836797c7e146469d82e161733b498f68", "57663f1d5b054051bfc2c6d97a95fd7b", "ecd2a5bd204a457b90c421645a468a6f", "c3f1c04d9b354a218c7092ee83f5b6ff", "754feb2a9d6441f5b750b811e4525642", "d86f2a3473914a78bd764196effd8f85", "8041207d1df248a5bca809f2570789cd"]} id="K99HjbwdHX3p" outputId="c688d7f9-6b06-43db-ddb2-0d1e6deb3e67"
cnn_news = {}
bad_news = {}

if SCRAPE_CNN_WEB_PAGES:
    pbar = tqdm(updated_url.items(), desc="description")
    for day, url in pbar:
        pbar.set_description(day)

        # get the file
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            s = BeautifulSoup(r.content, "html.parser")
        except Exception as e:
            print(f"Error scraping [{day}] - {url}: {e}")
            bad_news[day] = (url, None, None)
            continue

        # parse the file
        try:
            title = s.find("h1", {"data-editable": "headlineText"}).get_text(strip=True)
            contents = [
                c.get_text(strip=True)
                for c in s.find_all("div", class_="live-story-post__content")
            ]
            cnn_news[day] = (url, title, contents)
        except Exception as e:
            print(f"Error parsing [{day}] - {url}: {e}")
            bad_news[day] = (url, None, None)
else:
    print("SCRAPE_CNN_WEB_PAGES is False — skipping live web scraping and using cached cnn_news_<date>.json.")

# %% colab={"base_uri": "https://localhost:8080/"} id="roLMyHEPHeJ-" outputId="f3ea88a4-8411-49d0-db90-aafa0a0f1b20"
if SCRAPE_CNN_WEB_PAGES:
    print(f"{len(cnn_news)} found over {len(updated_url)}")

# %% colab={"base_uri": "https://localhost:8080/", "height": 241} id="KXY6WorLHeHn" outputId="1e92f4f4-cc7f-4c9b-9bb4-2422464183ee"
if SCRAPE_CNN_WEB_PAGES and len(cnn_news) > 0:
    CNN_04_df = pd.DataFrame(
        [
            {"date": date, "title": title, "contents": contents, "url": url}
            for date, (url, title, contents) in cnn_news.items()
        ]
    )
    display(CNN_04_df[:4])

# %% id="X5FukZ8-HeE0"
# [url for date, (url, _, _) in bad_news.items()]

# %% [markdown] id="XjFEb0rSdR0V"
# # SAVE STEP 1. Save/Load the most complete version (Google Drive or local `resources/`)

# %% colab={"base_uri": "https://localhost:8080/"} id="Srxlr5wOeR28" outputId="e161d239-0513-468c-8fdd-b5575596c641"
# Resolve storage directory: mounts Google Drive if available, otherwise uses local resources/
drive_path = str(resolve_storage_dir(local_dir=get_resources_dir()))

today_str = datetime.today().strftime("%Y-%m-%d")
file_path = os.path.join(drive_path, f"cnn_news_{today_str}.json")

# %% id="xwW_etJOHd8c"
# Save the cnn_news dictionary to a JSON file if scraped
if SCRAPE_CNN_WEB_PAGES and "cnn_news" in locals() and len(cnn_news) > 0:
    if os.path.isfile(file_path):
        print(f"not overwriting {file_path}")
    else:
        with open(file_path, "w") as f:
            json.dump(cnn_news, f, indent=4)
        print(f"cnn_news dictionary saved to {file_path}")

# %% colab={"base_uri": "https://localhost:8080/"} id="K4ugIiFlnVEq" outputId="347b6744-a9c3-4f7f-e3ae-0d5f9dec1d42"
freshest_json = find_freshest_dated_file(
    drive_path,
    prefix="cnn_news_",
    suffix=".json",
    fallback_dirs=[get_resources_dir()],
)

if freshest_json is not None:
    most_recent_file = freshest_json.name
    full_path_most_recent = str(freshest_json)
    print(f"The most recent file is: {most_recent_file}")
    print(f"Full path: {full_path_most_recent}")
else:
    print("No files matching the pattern 'cnn_news_YYYY-MM-DD.json' were found.")
    full_path_most_recent = os.path.join(drive_path, "cnn_news_04.json")
    print(f"Using default path: {full_path_most_recent}")


# %% colab={"base_uri": "https://localhost:8080/"} id="maBR5nKWeWBu" outputId="53b16f6e-2408-42dc-a864-5c5fadf79223"
# Load the cnn_news dictionary from the JSON file
with open(full_path_most_recent, "r") as f:
    loaded_cnn_news = json.load(f)

print(
    f"Successfully loaded cnn_news from {full_path_most_recent}. It contains {len(loaded_cnn_news)} entries."
)

# %% colab={"base_uri": "https://localhost:8080/", "height": 146} id="cRuYUmYseqnx" outputId="2ccc0a1b-f996-48b9-b22e-4b19341ab4da"
pd.Series(1, index=pd.to_datetime(list(loaded_cnn_news.keys()))).cumsum().plot(
    figsize=(13,1), lw=3, alpha=.7, marker='o', grid=True, rot=0,
)

# %% [markdown] id="v0m3lHK99heY"
# # STEP 2.  Structure the two dataframes

# %% [markdown] id="kuJJGK7i920c"
# ### One simple dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 289} id="WFuGU1dPe1hj" outputId="d614cf2e-64e5-4841-9478-678bf76e66a2"
import pandas as pd

cnn_news_df = pd.DataFrame(loaded_cnn_news).T
cnn_news_df.index = pd.to_datetime(cnn_news_df.index)
cnn_news_df.columns = ['url', 'title', 'contents']
cnn_news_df.sort_index(inplace=True)
cnn_news_df.head()

# %% [markdown] id="OqYP5odTULul"
# ## Display and replace

# %% id="pEFUs9KjUNiK"
replaces = ['Äô', 'Äì', 'Äî', 'Äò', 'Ä¢']


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="Rx9qmq5490sh" outputId="d5260904-f02f-4af0-e385-10fb5bfca903"
from IPython.display import HTML

html_string = """<table border='1'>
<thead>
<tr>
<th>Date</th>
<th>Title</th>
</tr>
</thead>
<tbody>"""

for index, row in cnn_news_df.iterrows():
    date = index.strftime('%Y-%m-%d')
    title = row['title']
    for r in replaces:
      title = title.replace(r, '—')
    url = row['url']
    html_string += f"""<tr>
<td>{date}</td>
<td><a href=\"{url}\" target=\"_blank\">{title}</a></td>
</tr>"""

html_string += """</tbody>
</table>"""

HTML(html_string)

# %% [markdown] id="pnZi6vus97DB"
# ### An exploded dataset per content

# %% colab={"base_uri": "https://localhost:8080/", "height": 620} id="pIT9L4aBfADm" outputId="8b34a324-8fd1-443e-ee35-272aaea6eb0c"
cnn_news_exploded_df = cnn_news_df.explode('contents')

for r in replaces:
   cnn_news_exploded_df['contents'] = cnn_news_exploded_df['contents'].str.replace(r, '—')

cnn_news_exploded_df['content_order'] = cnn_news_exploded_df.groupby(level=0).cumcount()
cnn_news_exploded_df = cnn_news_exploded_df.reset_index().rename(columns={'index': 'date'})
cnn_news_exploded_df = cnn_news_exploded_df.set_index(['date', 'content_order'])

cnn_news_exploded_df

# %% [markdown] id="YiuZR2aj9puB"
# ### Remove non meaningful contents

# %% colab={"base_uri": "https://localhost:8080/"} id="FYl1tth47Kx0" outputId="59958ca2-bd89-450a-be97-92e2fd1b179f"
print(f"started with {cnn_news_exploded_df.shape[0]:,} items")
for bad_sign in ['Get', 'Our', 'Follow the latest', 'Live coverage of the']:
  cnn_news_exploded_df = cnn_news_exploded_df[cnn_news_exploded_df['contents'].str.find(bad_sign)<0]
print(f"..ended with {cnn_news_exploded_df.shape[0]:,} items")

# %% colab={"base_uri": "https://localhost:8080/", "height": 484} id="GoIkBTy5B88j" outputId="dfee8903-a317-4b31-9e90-49ba8fb49953"
all_items_ser = cnn_news_exploded_df['contents']
all_items_ser

# %% [markdown] id="uYEZgAbAYwph"
# # STEP 3. Do a t-SNE on modernBERT Embeddings

# %% id="THCvNU_CY2lw"
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# %% colab={"background_save": true, "base_uri": "https://localhost:8080/", "height": 487, "referenced_widgets": ["88e3f4a8914c4885a1bf8b562a2712f5", "ec5efc5d0d2047bdbe765e01a2f7e63d", "ee6ac31c4a6e47939fb7a86ddae506e7", "2e40ada9f3bc436e814572724fbaaad0", "995986df0a1644a7abdc55498f19aaaa", "d8414de16cc746b5ba4665d1d0483dcb", "6b56771c3121496b9b20409882ae1eed", "f1ac329a0eb642358f1371278eb99a52", "462386f730e04f5ebb3b7058ef296efd", "47edb6a2c5f84f049e4138fd1798e48b", "8e849dd5370f47b3819fbe30c825c295", "860e11a912774c27ba379637c68e3041", "f4be01aebfbd431c828f6b7cf31819e2", "af9a63d232fa4aa4ba68340852ea7bda", "d5c2c4c2853c4aae9449c7520b62bee9", "99f5bc20ad22411dbc11a8ac4cc38b60", "ea9353a8e1c446f7a4dac090706e5a31", "365ec4e26db241e7b44891729106913f", "690742b5e4344ceb8aaf9baf4b222f77", "3de477030bc34c049b7ebfa7087204ce", "c0c918d136d64c52b999f4c0cf1be326", "10189ca93dda4045ad4f05189c19ed04", "3ad269972974466cb1bcaddd7ec93f24", "4a4f72e64b644621abd35e6b9e525dd6", "057df5d8bab648f3b7c7313ca80758df", "1891315aba6640d8bc2e807c30d182c1", "63becad33391470088540d81b1b72d6a", "cd7660f733cb4294b2fc181271cadb39", "2dffd0e8a6a64cfcb38fc736240b537d", "542a8a3dd76f4b22b10ca02faead4003", "b1eae16ec68045cfa23c344fa0918820", "346caa656eb846db980f00e48b73a66b", "59f2441a6e914a508e681cbd7c6831a4", "d259c580ec4c4f5bb6de54d9d78c065d", "c0a3f371de6e41ce96676b083d9a7298", "e859859dcb084b75818b421943390c9d", "6de3a7dc5542467e8de1eb4b3210792f", "573b6d8ca27d468f814d2a534135479e", "b68c6eff560242dc9725329737b68e09", "85dea1476cb04333955bb042a1fca2df", "35a6ee0be49b4e6d87396f1a661779a1", "07e233d5f5474bf987c86160b9c0280d", "6d2bcd801ed8437888a0ca7f7d1cd521", "9a2689a55be548b29bad6a332bdd4a5f", "9a8e8cd0e8fd4c35933c5846736bd051", "c8f32fd7a27344c18601c5fd37ee81a8", "fa608b5590264b2bbd36d2c04f2590db", "ac18b473ed784ff0a1ff0ca2a28839f2", "71474f11b8b74c6ab9b1dd37d139bb6c", "ac34297bbb554ddca9c85e56eaeecf81", "0d5577b5c2ea4ecd9706b0021466830b", "b8f207269242492bad6752925fabff67", "885d3c6a67c2440fb6e207ce6276e122", "41d655e74bd142d4be1383c5e8b7e244", "0bb0e91230284590b09bd85eef427f17", "d352ac093b1842cbbfdc98cc97753513", "222b132a7010485ab92e5c83531f7623", "e4a49b8a34924ad194507fbd15e28fff", "4337d3b6ae44494c8c72b663a317f09e", "6e62b07b1c89432084dc34a9191925bf", "c16638fe97e24f1183e077e21a7d0f95", "a4be01c038a040079209b3a8642c62c2", "fdb7f628fbe84a83b294901cd755849f", "65bfc294ac62499bac8b03be10a2b2a7", "9627807c2e4f4609996d74cd83066138", "aef9edc470504a98a49e1fc080e30df3", "5c1d50974db749ae91d021aadfbd0f28", "3ffedc26a4194bb6bfe7ea70cab64165", "37364cb083ff44c0b57cb480fafd34ba", "dbbd63de1aeb49d6b866c717da84b28a", "79c6eee243774274842c861b2662a804", "2225785a0f734bb289c43b5f2d9e2dc3", "d3bf8866c08b4a21953b2faeb0d150cb", "68bd68aa7ee8437fbd4e2707b26b295f", "582e856814a3471aa9071e7975c345ac", "d89a6e3a012e4e7eabe276f60597b031", "72b56cb9089f47f081d22c1efd8015bb", "050c4a55b79f436daec500c6d7ff0518", "6803b1870e2444edab47a5a69d3144d7", "e9837f1eedab466e8ae63c0d068dce40", "8f49110c16984998a5adc6b4453bef4f", "680156319414478aacc2359c0a7b8f75", "7b0cea1f1551429696d1beb0946fd2d1", "281751bdc0144befae4c0a70bc887c17", "d5dea051e3cf46dc9dc1918f517b874c", "a0e18ff9a63a4a89863b4010b970d7fc", "d231c5469ac24d43ae48458460875077", "3d27b6675fbc475c8c4d4200938daff3", "d2df7678ff5d443ea78913b0342bb8ed", "a55512a6d1d34e97958a49177d4e634f", "0d8f9ac1ba364514a05a9bd11d737b6c", "a78ace89a8be4b708db7cbfeac484abc", "b004a69e61b5425bb7d52dc39b159d97", "4a0db4926b464500864097db07b7647e", "6da07db32ee741d0b976378d53057e11", "43d582e21c5e4373a38f1729ba5d0403", "243fcfa246df46c0bd62280f98425f7e", "d883fcd36c0a4144938cd205c91e956b", "8443dc1c086e4457a79bbe0c828eecbe", "61a28f489fdb42eabd7e1e0705859f4e", "ae205b44b8684563a05a4352398f2367", "be70716ca7d04848978327be927183da", "66c771f6f4744482b12fa531a65bd1f9", "7e6c91ed2ae34f32b44bd3413699c604", "6d2b88028d9f4eb9b18445fa76ba3e86", "8bd98204d22d4824b081ed951afb3ca2", "88d579efebfe4dd3bff7950fce210754", "b81881df7b5441f2beb98b0190b99198", "ce3dff5cfc1344da8d4a7196d91bdde4", "a9dd7644fc604de38353fd0dfcd82a66", "34be7435090649a7969f555fdee70086", "4c43b61e7ef941d59be731bf870d9bfc", "f0c609a1d07e4bcf9703e1fb409ca8c1", "ce6a6261a43a4b8aa501120ca9cf26e1", "010b1b9eb7524285a5d20273d8760472", "879c70ebd18944aebf2a406af4cef1c7", "c6794b1fe17d46dc8ccbe14c85f97765", "d02ea9d6fed8401598bd6f16c410f64d", "d63abe98078948329bc30c515dd1a6e9", "956f7e932dc544c28c318d81e2bd5783", "961c0e748e824ac5b2cba55a0f00ba6a", "dbc15fa86d1243c0a5e553edbd31f5f5", "1489d91e4be34dea92450406c8c94f31", "0a7ded8c5ac5414d9e9011682a4b7f01", "21de610bacb1481991563ea51542700f", "0c2d3cfb923443f89af4bee102474185", "4f9d281f623541639814a483310952d3", "2e227d4970204903ac18cbd17004d2a0", "038531a1441d4926a130c937100ea4bb", "82eaf7ef93d746789e189782e486dcfc", "5faa5085b93d4dc2831a154b493cdb52", "3437f1e60fce455aa78fb7b0db316d1b", "41c37c8103114bed91c381988f789b7e", "fd7f970396c24714a693b1a8365e3957", "f3a4e22665ce4a7cbf1fea4a9d8732b5", "5acc6fa2a9e5487ca903bf376081e09e", "70ebea3a6f4a4e328eeb5205c9a70f54", "c1da0489a8274d6fbe0397be1136a334", "ffdfd295799c495dbc018276da5b2136", "c1461179508042609b4a8454b6395269", "f5a8e5e6ea9346bf8924e093068002a8", "b5dc9d63c9eb4093aa5b74d276d55668", "bc95cdbd888f42bca64aec51ae0c707e"]} id="meQ6eZiGC1mE" outputId="065fba03-c0a2-4035-a257-ae277a3e0da0"

# --- 1. Embed -------------------------------------------------------------
# Using a ModernBERT variant *fine-tuned for embeddings* rather than raw
# answerdotai/ModernBERT-base (MLM-pretrained, not contrastively tuned).
cached_emb_path = full_path_most_recent.replace(".json", "-embeddings.parquet")
if not SCRAPE_CNN_WEB_PAGES and os.path.isfile(cached_emb_path):
    print(f"Loading cached article embeddings from {cached_emb_path}")
    _cached_emb_df = pd.read_parquet(cached_emb_path)
    embeddings = np.vstack(_cached_emb_df["embedding"].values)
else:
    model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    texts = all_items_ser.tolist()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize -> cosine geometry
    )
    embeddings = np.asarray(embeddings)



# %% colab={"background_save": true} id="wR-okiI6rixk"

# %% colab={"background_save": true} id="gw8kHJiCXplT"
# --- 2. Optional PCA pre-reduction (recommended for t-SNE stability/speed)
pca = PCA(n_components=50, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)



# %% colab={"background_save": true} id="fBysteOcYqal"
# --- 3. t-SNE ---------------------------------------------------------------
tsne = TSNE(
    n_components=2,
    perplexity=35,          # tune between ~20-50 for n=1572
    metric="cosine",
    init="pca",
    learning_rate="auto",
    random_state=42,
)
coords = tsne.fit_transform(embeddings_pca)



# %% colab={"background_save": true} id="fKep4xcHYsVs"
# --- 4. Reassemble with the MultiIndex for downstream analysis -------------
tsne_df = pd.DataFrame(
    coords, columns=["tsne_x", "tsne_y"], index=all_items_ser.index
)
tsne_df["date"] = tsne_df.index.get_level_values("date")
tsne_df["content"] = all_items_ser.values



# %% colab={"background_save": true} id="MrexQeS-YtL5" outputId="750d500b-68ec-4d69-f14e-c18bf9f02512"
# --- 5. Quick plot -----------------------------------------------------------
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    tsne_df["tsne_x"], tsne_df["tsne_y"],
    c=pd.to_datetime(tsne_df["date"]).astype("int64"),  # color by date
    cmap="viridis", s=10, alpha=0.7
)
plt.colorbar(sc, label="date (encoded)")
plt.title("t-SNE of content embeddings")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.show()

# %% colab={"background_save": true} id="7voxUoSCYthX"
from sklearn.cluster import KMeans
# pip install hdbscan   # if you want the HDBSCAN alternative below
# import hdbscan

# --- 6. Cluster on the embeddings (not the t-SNE coords!) -------------------
N_CLUSTERS = 8   # tune based on your data / silhouette score / elbow plot
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(embeddings_pca)

# --- Alternative: HDBSCAN (no need to pick k, flags outliers as -1) --------
# clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric="euclidean")
# cluster_labels = clusterer.fit_predict(embeddings_pca)

tsne_df["cluster"] = cluster_labels



# %% colab={"background_save": true} id="8tZy6Wsjuz_R" outputId="6ac9405d-42e1-4e35-96c1-f9f39156bd89"
# --- 7. Re-plot t-SNE colored by cluster (sanity check) ---------------------
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    tsne_df["tsne_x"], tsne_df["tsne_y"],
    c=tsne_df["cluster"], cmap="tab10", s=10, alpha=0.7
)
plt.title("t-SNE colored by cluster")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.show()



# %% [markdown] id="SjMlaytHvte_"
# ### My clusters
#
# Based on the clustering output of the news snippet collection, here is a thematic summary for each cluster, along with a detailed explanation of why each label was chosen and supported by typical examples from the text.
#
# ---
#
# ### **Cluster 0: Regional Hostilities & Secondary Gulf Targets**
#
# * **Theme:** Direct military attacks, drone/missile interceptions, evacuations, and security alerts across secondary Middle Eastern countries (Saudi Arabia, Kuwait, UAE, Bahrain, Qatar, Iraq, and Yemen).
# * **Why this summary was chosen:** Unlike clusters focused purely on central frontlines or energy logistics, this cluster captures the physical spreading of hostilities across the wider Persian Gulf and Red Sea region. It centers on civilian infrastructure impacts, drone/missile defense activations, foreign citizen evacuations, and proxy involvement (such as the Houthis in Yemen).
# * **Typical Examples:**
# * *Interceptions & Attacks in Gulf Nations:* `[2026-03-01 #3]` *"Sirens have been sounded over Bahrain..."*; `[2026-03-02 #10]` *"Saudi Arabia’s military intercepted and destroyed eight drones near the capital Riyadh..."*; `[2026-03-04 #16]` *"Authorities in Qatar are evacuating residents living 'in the vicinity' of the US Embassy..."*
# * *Evacuations & Stranded Civilians:* `[2026-03-01 #13]` *"Stranded passengers wait at Hazrat Shahjalal International Airport after flights to Dubai and Bahrain were cancelled..."*; `[2026-03-03 #7]` *"A military evacuation flight for Czechs who were stranded in the Middle East landed in Prague..."*
# * *Frontline Escalation with Houthis:* `[2026-08-07 #5]` *"The Houthis carried out an attack on Saudi forces at a base in eastern Yemen..."*
#
#
#
# ---
#
# ### **Cluster 1: Global Macroeconomic Impact & Energy Markets**
#
# * **Theme:** The economic fallout of the conflict, specifically crude oil price surges, gas station price spikes, stock market volatility, inflation concerns, and government policy responses (e.g., Strategic Petroleum Reserve releases, tax adjustments).
# * **Why this summary was chosen:** This cluster isolates financial and economic metrics, market reactions, and consumer-facing energy crises resulting from supply chain chokepoints rather than military actions themselves.
# * **Typical Examples:**
# * *Oil Price Spikes & Market Volatility:* `[2026-03-08 #18]` *"President Donald Trump dismissed concerns over rising oil prices as the price of oil surged past $100 a barrel..."*; `[2026-03-19 #7]` *"The price of oil surged, with Brent crude reaching $115 a barrel..."*
# * *Domestic Consumer & Business Impact:* `[2026-03-20 #13]` *"A driver fills their gas tank... Gas prices have climbed swiftly since the start of the war..."*; `[2026-03-21 #10]` *"Shirley Modlin was in meetings all day... Her small manufacturing facility in Powhatan, Virginia, relies on carbide..."*
# * *Government Economic Interventions:* `[2026-03-11 #15]` *"President Donald Trump has authorized the United States to release 172 million barrels of oil from the Strategic Petroleum Reserve..."*; `[2026-05-10 #18]` *"US Energy Secretary Chris Wright today suggested the administration may consider suspending the federal gas tax..."*
#
#
#
# ---
#
# ### **Cluster 2: Internal Iranian Politics, Supreme Leader Succession & Funerals**
#
# * **Theme:** Iranian domestic governance, regime power dynamics, domestic suppression/executions, the death of Supreme Leader Ali Khamenei, and the rise/mourning of his successor, Mojtaba Khamenei.
# * **Why this summary was chosen:** The dominant focus here is on internal Iranian regime affairs. It heavily features the death of Ali Khamenei, the multi-day funeral processions, political crackdowns, internet blackouts, and the transition of power to Mojtaba Khamenei.
# * **Typical Examples:**
# * *Leadership Vacuum & Succession:* `[2026-03-08 #13]` *"A body of 88 senior clerics, known as the Assembly of Experts, selected the new supreme leader [Mojtaba Khamenei]..."*
# * *State Mourning & Multi-day Funeral:* `[2026-07-03 #1]` *"Coffins containing the bodies of slain Supreme Leader Ayatollah Ali Khamenei and several family members are on display in Tehran..."*; `[2026-07-06 #5]` *"Tehran’s streets were flooded by huge crowds today, as hundreds of thousands of mourners came out..."*
# * *Domestic Repression & Executions:* `[2026-05-06 #4]` *"Iran has been carrying out the fastest wave of political executions in its recent history..."*; `[2026-05-25 #8]` *"Iranians reacted with a mix of skepticism... after internet monitors reported a partial restoration of online access..."*
#
#
#
# ---
#
# ### **Cluster 3: Direct US-Israeli & Iranian Military Escalation (Operation "Epic Fury")**
#
# * **Theme:** Direct kinetic strikes, airstrikes, missile exchanges, bombing runs, naval engagements, and tactical combat operations between US/Israeli forces and Iranian military targets.
# * **Why this summary was chosen:** This cluster is defined by operational military reports, CENTCOM updates, strike counts, strategic target hits (e.g., Kharg Island, nuclear facilities, bridges), and direct military engagements between the central combatants.
# * **Typical Examples:**
# * *Direct Combat & Strike Operations:* `[2026-03-06 #8]` *"The US military has struck more than 3,000 targets inside Iran since launching Operation ‘Epic Fury’..."*; `[2026-07-11 #1]` *"The US military said it hit 140 Iranian military targets in its third round of strikes this week..."*
# * *Strategic & Infrastructure Targeting:* `[2026-03-13 #4]` *"US strikes on Kharg Island, Iran [handling ~90% of crude exports]..."*; `[2026-07-17 #19]` *"American precision strikes on Iranian bridges, railway junctures and power lines..."*
# * *Military Engagement Announcements:* `[2026-03-10 #16]` *"CENTCOM releases video of US forces destroying Iranian naval ships and minelayers near Strait of Hormuz..."*
#
#
#
# ---
#
# ### **Cluster 4: US Executive Branch Strategy, Domestic Politics & Public Statements**
#
# * **Theme:** The political rhetoric, presidential announcements, White House press briefings, congressional oversight debates, and strategic posturing led by President Donald Trump and his administration.
# * **Why this summary was chosen:** This cluster is anchored almost entirely by political leadership in Washington. It focuses on how the US Executive Branch communicates war goals, negotiates internally, handles congressional opposition, and conducts public relations regarding the conflict.
# * **Typical Examples:**
# * *Presidential Statements & Strategy:* `[2026-03-02 #0]` *"President Donald Trump told CNN the 'big wave' of the US attack on Iran is yet to come. Trump laid out his war objectives..."*; `[2026-04-01 #1]` *"President Donald Trump used familiar talking points in his first address to the nation on the Iran war..."*
# * *White House Briefings & Political Messaging:* `[2026-03-25 #13]` *"White House Press Secretary Karoline Leavitt: 'President Trump does not bluff and he is prepared to unleash Hell'..."*
# * *Congressional Oversight & Domestic Friction:* `[2026-03-09 #12]` *"Several Senate Democrats have announced that they will continue to force war power votes..."*; `[2026-07-21 #11]` *"'You, sir, are the failure': Democratic senator slams Hegseth and Iran strategy..."*
#
#
#
# ---
#
# ### **Cluster 5: The Israeli-Lebanese/Gaza Northern & Southern Fronts**
#
# * **Theme:** Kinetic operations, missile/rocket exchanges, ground movements, and fragile ceasefire negotiations involving Israel, Hezbollah in Lebanon, and Hamas in Gaza.
# * **Why this summary was chosen:** While Cluster 3 handles the US/Israel vs. Iran main campaign, this cluster groups news centered on the Mediterranean flank—specifically Israeli defense/offense in Tel Aviv, northern Israel, Beirut, southern Lebanon, and concurrent developments in Gaza.
# * **Typical Examples:**
# * *Israel-Hezbollah Cross-Border Conflict:* `[2026-03-01 #7]` *"The Israeli military said it will increase and intensify its strikes on Lebanon..."*; `[2026-03-12 #18]` *"Hezbollah fired rockets that damaged homes in central Israel overnight..."*
# * *Attacks inside Israel:* `[2026-03-17 #9]` *"Emergency personnel at the scene of a missile strike in Ramat Gan, Israel..."*; `[2026-03-21 #14]` *"Iranian missile slams into building in Dimona in southern Israel..."*
# * *Lebanese & Gaza Ceasefire Efforts:* `[2026-06-04 #10]` *"A fragile ceasefire between the Lebanese and Israeli governments is being tested..."*; `[2026-07-31 #8]` *"The US-backed Board of Peace, tasked with overseeing a long-term peace settlement in Gaza..."*
#
#
#
# ---
#
# ### **Cluster 6: High-Level International Diplomacy, Mediated Negotiations & Ceasefire Agreements**
#
# * **Theme:** Diplomatic backchannels, multilateral summits, foreign ministry communications, and regional mediation (primarily by Qatar, Pakistan, and Oman) aimed at securing ceasefires and memorandums of understanding.
# * **Why this summary was chosen:** This cluster isolates diplomatic negotiations, official foreign ministry press statements, regional summits, and the drafting/negotiation of peace deals or temporary truces.
# * **Typical Examples:**
# * *Mediated Talks (Pakistan, Qatar, Oman):* `[2026-03-23 #12]` *"Islamabad is 'already ready to host talks' involving Iran and the United States..."*; `[2026-06-30 #1]` *"US envoy Steve Witkoff and Jared Kushner met with the Qatari prime minister today to discuss framework..."*
# * *Foreign Ministry Communications:* `[2026-05-06 #1]` *"Iran is expected to hand over its reply Thursday to mediators about the US proposal..."*; `[2026-06-12 #8]` *"A memorandum of understanding between the US and Iran could be signed by both sides remotely..."*
# * *Multilateral & Regional Coalitions:* `[2026-08-07 #1]` *"Saudi Arabia, Turkey and Pakistan signed a mutual defense pact..."*
#
#
#
# ---
#
# ### **Cluster 7: Strait of Hormuz Navigation, Maritime Security & Blockades**
#
# * **Theme:** Commercial shipping disruptions, naval blockades, mine-clearing, tanker attacks, and maritime transit management in the Strait of Hormuz, Persian Gulf, and Red Sea.
# * **Why this summary was chosen:** This is a highly domain-specific cluster focusing on maritime geography. It tracks ship tracking data, naval interceptions of cargo vessels/tankers, insurance/transit risks, and bilateral initiatives to enforce or reopen maritime corridors.
# * **Typical Examples:**
# * *Shipping Interruption & Chokepoint Traffic:* `[2026-03-03 #18]` *"The war in Iran has caused maritime traffic through the critical Strait of Hormuz to crawl to a near halt..."*; `[2026-04-20 #9]` *"Only 16 ships traversed the Strait of Hormuz on Monday, as captains and ships’ owners remain cautious..."*
# * *Attacks on Tankers & Merchant Vessels:* `[2026-03-10 #2]` *"A container vessel has sustained damage off the north coast of the United Arab Emirates..."*; `[2026-07-13 #6]` *"Iran’s IRGC said it 'struck and disabled' what it called two 'rogue supertankers'..."*
# * *Naval Blockades & Boarding Operations:* `[2026-04-20 #5]` *"US Marines boarding and seizing an Iranian-flagged cargo ship that tried to bypass the US naval blockade..."*; `[2026-07-14 #1]` *"The US has reimposed its naval blockade of Iranian ports..."*
#
# ## Summary Table
#
# | Cluster | Theme | Size |
# |---------|-------|------|
# | 0 | Gulf Regional Military Escalation | 209 |
# | 1 | Global Energy Markets & Economic Fallout | 81 |
# | 2 | Iranian Political Transition & Internal Affairs | 110 |
# | 3 | US-Israel Military Campaign in Iran | 246 |
# | 4 | US Domestic Politics & Presidential Leadership | 306 |
# | 5 | Israel-Hezbollah-Lebanon Theater | 170 |
# | 6 | Diplomacy & International Mediation | 235 |
# | 7 | Maritime Conflict & Strait of Hormuz | 215 |
#
# **Total:** 1,572 articles

# %% colab={"background_save": true} id="V7qanKbxu07W" outputId="41e7c023-cecf-4eac-eadc-53b0324c8594"
# --- 8. List original content per cluster -----------------------------------
for cluster_id in sorted(tsne_df["cluster"].unique()):
    subset = tsne_df[tsne_df["cluster"] == cluster_id]
    print(f"\n{'='*80}")
    print(f"Cluster {cluster_id}  (n={len(subset)})")
    print(f"{'='*80}")
    for idx, row in subset.iterrows():
        date, content_order = idx
        print(f"[{date} #{content_order}] {row['content'][:150]}...")

# %% [markdown] id="FEP0b551t6u7"
# # SAVE STEP 3. Save/Load embeddings (Google Drive or local `resources/`)

# %% [markdown] id="nkNfR3iXuwFy"
# ## Save t-SNE

# %% colab={"base_uri": "https://localhost:8080/"} id="GlZYeltbu1Md" outputId="5081297c-7987-4f96-f6f7-2ef950a63755"
output_parquet_path = full_path_most_recent.replace(".json", "-tsne_clusters.parquet")

if os.path.isfile(output_parquet_path):
    print(f"not overwriting {output_parquet_path}")
elif "tsne_df" in locals():
    tsne_df.to_parquet(output_parquet_path)
    print(f"tsne_df saved to {output_parquet_path}")
else:
    print("nothing to save for the t-SNE")

# %% [markdown] id="61ESh9LguzRc"
# ## Save embeddings

# %% colab={"base_uri": "https://localhost:8080/", "height": 252} id="j3YMwFIXtEaZ" outputId="fba70227-1d28-45fd-d187-dea374d152ee"
if "all_items_ser" in locals() and "embeddings" in locals():
    embeddings_df = pd.DataFrame(
        {"content": all_items_ser.values, "embedding": list(embeddings)},
        index=all_items_ser.index,
    )

    output_embeddings_parquet_path = full_path_most_recent.replace(
        ".json", "-embeddings.parquet"
    )

    if os.path.isfile(output_embeddings_parquet_path):
        print(
            f"Not overwriting existing embeddings file: {output_embeddings_parquet_path}"
        )
    elif "embeddings_df" in locals():
        embeddings_df.to_parquet(output_embeddings_parquet_path)
        print(f"Embeddings DataFrame saved to {output_embeddings_parquet_path}")

    display(embeddings_df.head())
else:
    print("nothing to save for the embeddings")

# %% [markdown] id="p4Z7-k3zvqju"
# ## Load t-SNE and embeddings

# %% colab={"base_uri": "https://localhost:8080/"} id="EMsPo5WkHDP3" outputId="bfd869bc-9906-4781-eec7-72bd8c709101"
drive_path = str(resolve_storage_dir(local_dir=get_resources_dir()))

# %% id="ZZ5z-T8jvueB" colab={"base_uri": "https://localhost:8080/"} outputId="a1c52f39-3df0-4c12-cbe7-e0cb35b3a44e"
freshest_tsne_parquet = find_freshest_dated_file(
    drive_path,
    prefix="cnn_news_",
    suffix="-tsne_clusters.parquet",
    fallback_dirs=[get_resources_dir()],
)

if freshest_tsne_parquet is not None:
    full_path_most_recent_parquet = str(freshest_tsne_parquet)
    print(f"The most recent parquet file is: {freshest_tsne_parquet.name}")
    print(f"Full path: {full_path_most_recent_parquet}")

    loaded_tsne_df = pd.read_parquet(full_path_most_recent_parquet)
    print(
        "Successfully loaded the most recent tsne_clusters.parquet file into 'loaded_tsne_df'."
    )
    print(loaded_tsne_df.head())
elif "tsne_df" in locals():
    loaded_tsne_df = tsne_df.copy()
    print("Using in-memory 'tsne_df' as 'loaded_tsne_df'.")
else:
    print(
        "No parquet files matching the pattern 'cnn_news_YYYY-MM-DD-tsne_clusters.parquet' were found."
    )
    loaded_tsne_df = None

freshest_emb_parquet = find_freshest_dated_file(
    drive_path,
    prefix="cnn_news_",
    suffix="-embeddings.parquet",
    fallback_dirs=[get_resources_dir()],
)

if freshest_emb_parquet is not None:
    full_path_most_recent_parquet = str(freshest_emb_parquet)
    print(f"The most recent parquet file is: {freshest_emb_parquet.name}")
    print(f"Full path: {full_path_most_recent_parquet}")

    loaded_embeddings_df = pd.read_parquet(full_path_most_recent_parquet)
    print(
        "Successfully loaded the most recent embeddings.parquet file into 'loaded_embeddings_df'."
    )
    print(loaded_embeddings_df.head())
elif "embeddings_df" in locals():
    loaded_embeddings_df = embeddings_df.copy()
    print("Using in-memory 'embeddings_df' as 'loaded_embeddings_df'.")
else:
    print(
        "No parquet files matching the pattern 'cnn_news_YYYY-MM-DD-embeddings.parquet' were found."
    )
    loaded_embeddings_df = None


# %% [markdown] id="NW5crQWyybim"
# # STEP 4. Monitor narratives

# %% id="I-nTyO11xDIs"
cluster_examples = {
    0: [
        "Saudi Arabia’s military intercepted and destroyed eight drones near the capital Riyadh and the central city of Al Kharj.",
        "Authorities in Qatar are evacuating residents living in the vicinity of the US Embassy as a temporary precautionary measure.",
        "Stranded passengers wait at Hazrat Shahjalal International Airport after flights to Dubai and Bahrain were cancelled.",
    ],
    1: [
        "President Donald Trump dismissed concerns over rising oil prices as the price of oil surged past $100 a barrel.",
        "The national average for regular gasoline in the United States reached $4.35 a gallon as supply disruptions from the Middle East continue.",
        "President Donald Trump has authorized the United States to release 172 million barrels of oil from the Strategic Petroleum Reserve.",
    ],
    2: [
        "A body of 88 senior clerics, known as the Assembly of Experts, selected Mojtaba Khamenei as the new supreme leader.",
        "Hundreds of thousands of mourners flooded the streets of Tehran for the multi-day funeral procession of the late Supreme Leader Ayatollah Ali Khamenei.",
        "Human rights groups report that Iran is carrying out a rapid wave of political executions and arrests amid widespread internet blackouts.",
    ],
    3: [
        "The US military has struck more than 3,000 targets inside Iran since launching Operation Epic Fury, Central Command said.",
        "US precision strikes targeted critical oil infrastructure on Kharg Island, handling roughly 90% of Iran's crude export capacity.",
        "CENTCOM forces launched a new heavy wave of strikes targeting Iranian missile sites and air defense systems around the Strait of Hormuz.",
    ],
    4: [
        "President Donald Trump told reporters in the Oval Office that the US is considering winding down military operations as objectives are near completion.",
        "White House Press Secretary Karoline Leavitt reiterated that the administration is prepared to unleash maximum pressure if negotiations stall.",
        "Senate Democrats forced a war powers resolution vote aiming to restrict further unapproved military action against Iran.",
    ],
    5: [
        "Hezbollah fired a new barrage of rockets into northern Israel, damaging residential structures in Tel Aviv and surrounding areas.",
        "The Israeli Air Force unleashed successive waves of airstrikes targeting Hezbollah infrastructure and command sites in southern Beirut.",
        "Deadly clashes along the Litani River threaten to unravel the fragile US-brokered ceasefire between Israel and Lebanon.",
    ],
    6: [
        "Negotiators from the United States and Iran met indirectly in Doha through Qatari and Pakistani mediators to discuss a framework agreement.",
        "Iranian Foreign Minister Abbas Araghchi stated that an understanding on major technical issues has been reached but key conditions remain.",
        "Saudi Arabia, Turkey, and Pakistan signed a trilateral mutual defense pact aimed at enhancing regional security coordination.",
    ],
    7: [
        "Maritime traffic through the critical Strait of Hormuz has crawled to a near halt as commercial tankers face extreme risk.",
        "US Marines boarded and seized an Iranian-flagged cargo ship attempting to run the naval blockade in the Gulf of Oman.",
        "An oil tanker was struck and disabled by an unknown projectile off the coast of the United Arab Emirates, according to UKMTO.",
    ],
}

# %% [markdown] id="cogWS-G0wsH3"
# ## Orthogonalization of the prototype narratives

# %% id="5IL0m3tpwjGM"
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %% id="dNzebh81wzkK" colab={"base_uri": "https://localhost:8080/"} outputId="5cb66322-6c34-40ee-ab50-80436df64ba7"
cluster_examples_list = []
cluster_ids_list = []
for cluster_id, examples in cluster_examples.items():
    for example in examples:
        cluster_examples_list.append(example)
        cluster_ids_list.append(cluster_id)
print(f"{len(cluster_examples_list)} prototypes available")

# %% [markdown] id="lGjglFplHpUB"
# ### Embed narratives

# %% id="MFEHUwYOw1LU" colab={"base_uri": "https://localhost:8080/", "height": 487, "referenced_widgets": ["ca5ec03ae8fe485c8b89141b291e490e", "6df6321c1b9b4d4cb80d84a2cc5b180b", "7ceb619039c941219eecb427ad311ebf", "7a567014cdd7460d945325dc2231e45f", "bd4dd86136364723bef98ca313057a51", "bc87c8b2ff644e509506d7cf011e2184", "e1eebd517be349f5a27d29e0a7859d2c", "1dfcfdac28be4ba093a3f09f3341e82d", "1c0396558ca541a2a1dccd52a3efbea1", "3b5c3ce2ad7f48b2affd2b2549d6d6aa", "975b163a4c3246c69de37cebb97fde4c", "515e926b1b684bd7be572a785c9d1474", "cd10fd99b660468ca9d9864b61a4a994", "7761cdd567d5421a9c41994d98811776", "3e8589ae0b7d49b7a092ba6236ef4913", "c95bcf0f1f964770ac1344dddbb3c06d", "46591106272a4523b80401cb682afb1e", "e268488ac74a4960a2a7d8ff7e354dad", "4403788fc0394eac8e89ea110c3d817c", "f37a806818dc44d48c8f8929ab4c04f0", "e9b2b321dc0d4a7cb40e45d706a25ecb", "05e1827aa296408a89370777d99b64c8", "d9c08650f4fb45a8ac14bce4f4de561b", "33a2edf51d554b34b7879651b62a4691", "a401adfc673c44bcb8c58bc0b95019b1", "d53065f6782240b98ab55f81d3ba9848", "4d72675bc459468489db97237b85a89c", "bace9ff97b2046719fbb5404354ab44d", "35c0c263e3104cc1a2de5be9bc276623", "852133f38b284c17b830e4a0c915d9ef", "70020df6bb004a90a7baee1f5e466d06", "8f31824f5eb148b4ab83e08e7966ea40", "9925c6bf3ebe45759b418b3fe9aebf10", "897f6fd713fd40deb23bda66b09f3aa9", "1063a7c1435046d1a70a9de12a5b120e", "ca48d0495fca40ed81ceadde4a89b562", "125fa88e3bbd489bbdb902651f7e6850", "dd2ff170457b46c7a85f06dfa8ad13d8", "66b4446467ec413980552a25a0908ae3", "40e3a8dea56e4e9db27a4722886172fc", "3fa90a6a20a54147b6ec50e07b842b17", "8cbac4951f354c60843e6f9db177bf0d", "1ec2dfb10c3c46a7a40362eb0d154139", "787b15a2b2684a678fb2a16e62fad46c", "4641fad169424b24a26169ddfffa5c10", "be9a4544ef3848548e95d5fdc3c817f9", "a540727335ca4c75a1d9fce05f8c25a0", "e7b6196eb9c846dda1da4c8c08c4ed35", "432b43c604f14c5281e23a743c9e98e9", "29f33b05aa0b4a2ba44dafffdf2df58f", "d4ca854c3f394fb5b816359c9a1fc6f5", "62dccb90673a4e3d83c2d9a61721cb43", "c2cb72b0f10c48bab923747a242ef209", "a517fbf3cc1e4724a9cf298cd3144123", "3c0bef7b9b4c4ad6a2d893db2ae5b69c", "a4a7947ff3b8436980cc5b2e0b9d85e2", "67bfd58f018f42ae880b7bb7b010adc0", "8302ced2615e401ca7770299f5b15794", "fc17fc1b8395425894969c0b88c95d2e", "fdb578b1f9ff43f2a9efaf0c10eca367", "d0d2fd3a8c56465f86bfe823f76f12c9", "c8ea25759db747f593fa1c77796aa982", "b3c754b0831c469cadfeb72ce1b26350", "ae04f1a4bc92415c8beb43330ffc791c", "acef34422bf14b30a02d11c7f8987efc", "361573ebce1c4c4c8bc78ae5f3935d6a", "a1c2516b738e4b0fb344ec86690df2f2", "a881e52df55b4d36934d9758cddeb51c", "c3b2b7701abd4cf98dcbed9735f2bbc9", "ae9f1733d5cf41a2aa640e4fa16cf64b", "ed1d0d2845a34055a24cb989eed0c78b", "fab2003904054821872eee6c7c91aea8", "e32300641d89484e80c73c28a5c6976d", "b279d075d06a4947a212bab579d0cc5b", "ce1abbaa0a424bdba1f9f4bec417e93f", "ffb82917a6d24a0ea2162f817d6d2b27", "d4aa62f764594bb6b745ba133b032107", "82297066aec04726a47239de3ab9f1fe", "17279d6f425344a59e46f714c3fe4d79", "4d7397a7b1e244b49d346a15a246105b", "f628bbb1dba84c1bb98c3f052d68d65d", "87158db41d6e420b9be70c004eb4e85d", "f98cefc5adc044c5a79d1b7ee104f062", "7998e75566704d9aba8eac2384f3901f", "2c9f71b77fd64205882cfe1c842384de", "4c1a6594f24f4e4b97e36d5b4e92e647", "80c1511e641a4f49b379610c9bea548c", "7945838faf8b4e58a720da2749b5711a", "8d843129c2fc4a86bd56aa70511ecfc4", "70c3c8ad644e4321be233b70b28ad1f6", "bbde8e0869b04546ab41cf7063704ec7", "38306329fdca44e99dfd5db0c401d51c", "9e7c6cc37f224227923db4c729349603", "d7b368e7efe6419f883faf9ef07e69a3", "3776a572b62041138f9b09111163656d", "493346aa251246458eadbed033d39c59", "c7122fb9b8cd44b0be6794b0923b55d8", "b61afafad0c6416ebb87cf83e3a2d219", "4e1e282f6405475baf29fbec39b6a3cc", "e73174d4de8a4f53a32dcf3980126963", "a93428be94804c0891ed1066e4f6482e", "b33d68516577491ea7c66a19dbd2c573", "94428b5295d74c48ae219c431a5863aa", "0fbff8322ec642218a782d89c6831c5a", "5ff4b53ba7084f0d9490f545bc0afca8", "51dd745cf6dc4c0882335380229b1c47", "0eb8e482190d43a0abb2890f69310c03", "03fa29c3bbc24d9b91629f5564d2ce8b", "53fd8ed8069f49a4980a7dc4ba4d2334", "2d1b6a430e1b487eb735700302677361", "4cbd79281d564e2e8dc29d456bde70e2", "cc8bb1b45d594d8f8061fdf99e5a026c", "d2be703a78ec43a19acd84ecc81e9354", "3c5a53c95cd54422a894bac8e512e535", "8df571f5a8734ae3bd96e62d9b6c2970", "49f87d0b50a14bb9af47929c418891db", "f3fc7c5542de4c3fbca2e424998b8e6a", "881f2a45f05b445e954f0e1c60795d37", "7a0b15cfc87c4c929682b6ade3e80ee4", "ffb75761cd184eda896c3f4ec6b1b07d", "52ca50c0ab034a7d8413a37675ae8c46", "b2426d4f89bd4f0f9b18263be0bc77ae", "d5213e58417345e08cee6f659cd5dca1", "5c24833b80cd45b581645fdc34a6481d", "b545f156a72b44cebe3497cc7da9ecd7", "2bdb7318a8524385be46438793899b94", "fc7015d705b141428126221170df0bb2", "81a69a7c50f048ab91be903e96d476a2", "a078ab5694ba497cb9ecc5418aa064af", "8412e96e48db43e4ad3d51a269818e44", "5d310dd8d2eb4aa996fd09ad9dd80c62", "dd5c80a3cfec4596b4b641c6f27b66f6", "786877977f27424f85410121611c6977", "359d350e1d2545f09eb36f31e54c0580", "076dfab298954602b606805fc411608a", "01c2c64096094a099f0ce74cc79c80b4", "87b42ff403074c7fb2f41b5a8efb3ca2", "6d15bac2b8c5465c8854d159559c9261", "d601ef05affd477ab0a9ba0823078753", "167cbd9c34374eb69d3bfb35586c7ae3", "3dbd846e963d4316911f3d446907d6eb", "9beffcd433aa4d9ab4d3a625689d19c1", "1d69e77c23434ea6b5d044d526c70c66"]} outputId="f4881808-0f12-414d-9fcf-21522d2b2734"
# Using a ModernBERT variant *fine-tuned for embeddings* rather than raw
# answerdotai/ModernBERT-base (MLM-pretrained, not contrastively tuned).
model = SentenceTransformer("nomic-ai/modernbert-embed-base")

examples_embeddings = model.encode(
    cluster_examples_list,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalize -> cosine geometry
)
examples_embeddings = np.asarray(examples_embeddings)


# %% colab={"base_uri": "https://localhost:8080/", "height": 176} id="ocbbVSeIBa77" outputId="ff5c0c5a-6fa7-407a-9cde-e908d18ab7a3"
example_embeddings_df = pd.DataFrame({
    'content': cluster_examples_list,
    'cluster_id': cluster_ids_list,
    'embedding': list(examples_embeddings)
})
example_embeddings_df.sample(3).T

# %% [markdown] id="L0aPMfQSQABZ"
# # STEP 5. Orthogonalize

# %% id="kvfbjvuDBlvQ"
from sklearn.metrics.pairwise import cosine_similarity
# gram_schmidt, stiefel_optimize, decompose_narratives, and average_cluster_prototypes
# are imported from narrative_geometry


# %% colab={"base_uri": "https://localhost:8080/"} id="to3_w4qBShRK" outputId="af50123f-0d5b-44a1-e824-880274050306"
cluster_descriptions = {
    0: "Gulf Regional Military Escalation - Comprehensive chronicle of Iran's retaliatory strikes against Gulf states and the regional defense responses, spanning six months of conflict. Includes air defense operations across Bahrain, UAE, Saudi Arabia, Kuwait, Qatar; drone and missile strikes on oil facilities, airports, and military bases; evacuations of foreign nationals and closure of airspace; civilian casualties and infrastructure damage in Gulf states; and US military deployments and naval operations in the region.",

    1: "Global Energy Markets & Economic Fallout - Coverage of the war's impact on global energy markets, focusing on oil and gas prices, supply chain disruptions, and economic consequences worldwide. Includes oil price volatility and Strait of Hormuz disruption; gasoline prices and consumer impact in the US; Strategic Petroleum Reserve releases; impact on vulnerable nations and food security; OPEC+ production decisions; and LNG tanker movements and shipping costs.",

    2: "Iranian Political Transition & Internal Affairs - Iran's domestic political upheaval following the assassination of Supreme Leader Ali Khamenei, including succession, funeral rituals, and regime consolidation. Includes Khamenei's assassination and succession by his son Mojtaba; multi-day funeral ceremonies across Tehran, Qom, Mashhad, Najaf; Iranian political repression and executions; regime propaganda and Revolutionary Guard operations; and internal dissent and human rights violations.",

    3: "US-Israel Military Campaign in Iran - Detailed operational coverage of US and Israeli military strikes against Iranian targets. Includes strikes on Iranian military, nuclear, and energy infrastructure; Operation 'Epic Fury' and ongoing air campaigns; US aircraft losses and crew recovery operations; Iranian missile launches toward Israel and Gulf states; targeting of Iranian leadership and IRGC commanders; strikes on Kharg Island oil terminal and Bushehr nuclear plant; and humanitarian impact on Iranian civilians.",

    4: "US Domestic Politics & Presidential Leadership - Coverage of President Donald Trump's leadership, communications, and the domestic political debate surrounding the war. Includes Trump's war objectives and timeline statements; congressional debates and war powers resolutions; Trump's negotiation strategy and diplomatic efforts; dignified transfers of fallen service members; White House press briefings and communications; political opposition from Democrats and some Republicans; and Trump's foreign policy interactions with China, Russia, and NATO.",

    5: "Israel-Hezbollah-Lebanon Theater - Coverage of the 'second front'—Israeli military operations against Hezbollah in Lebanon, Israeli-Iranian direct exchanges, and the Gaza/Hamas dimension. Includes Israeli strikes on Lebanon and Hezbollah targets; Iranian missile attacks on Israeli territory; Gaza ceasefire negotiations and Hamas involvement; Israeli-Lebanese border conflicts; evacuation of foreign nationals from the region; military casualties and civilian infrastructure damage; and Hezbollah rocket attacks on northern Israel.",

    6: "Diplomacy & International Mediation - Coverage of diplomatic efforts to end the war, including mediation by Pakistan, Qatar, and Oman, and international reactions. Includes Pakistan's mediation role and US-Iran talks; Qatari and Omani mediation efforts; ceasefire negotiations and memoranda of understanding; Iranian Foreign Ministry statements and diplomatic posture; international reactions from China, Russia, EU, and UN; technical negotiations on nuclear issues and Strait of Hormuz; and Iranian counter-proposals and demands.",

    7: "Maritime Conflict & Strait of Hormuz - Dedicated coverage of the maritime dimension—blockade, naval operations, shipping attacks, and the strategic Strait of Hormuz. Includes US naval blockade and Iranian counter-measures; attacks on tankers and commercial vessels; shipping traffic data and analysis; US-Iran naval confrontations; Iranian 'maritime authority' and shipping regulations; oil spills and environmental damage; and Admiralty/UKMTO incident reports."
}
cluster_short_descriptions = {
    id: description.split(' - ')[0] for id, description in cluster_descriptions.items()
}
cluster_short_descriptions

# %% [markdown] id="bGucajZVH36t"
# ## 5.1. On all the narratives

# %% colab={"base_uri": "https://localhost:8080/"} id="VLKsCVTNQUGb" outputId="9b2eb5bc-fb2f-4d1e-b877-25a8740c6a4d"
narrative_matrix_unorm = np.array([N for N in example_embeddings_df.embedding.values])
narrative_matrix, orthogonal_matrix, residuals_matrix, score = decompose_narratives(
    narrative_matrix_unorm
)
print(narrative_matrix.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 176} id="l47BSpnvRusL" outputId="2b2ee812-5301-4e12-d99f-349ce5154e39"
example_labels_ser = example_embeddings_df.reset_index()[['cluster_id', 'index']].apply(
    lambda df: f"{cluster_short_descriptions[df['cluster_id']]}[{df['cluster_id']}]:{df['index']}", axis=1)

example_labels_ser.sample(3)

# %% id="GaJ01pI5QvyZ"
narratives = pd.DataFrame([{
    "chapter": example_labels_ser[i].split('[')[0],
    "title"  : example_labels_ser[i],
    "embedding": narrative_matrix[i]} for i in range(example_labels_ser.shape[0])])
narratives_orth = pd.DataFrame([{
    "chapter": 'orth:' + example_labels_ser[i].split('[')[0],
    "title"  : example_labels_ser[i],
    "embedding": orthogonal_matrix[i]} for i in range(example_labels_ser.shape[0])])
narratives_complementary = pd.DataFrame([{
    "chapter": 'compl:' + example_labels_ser[i].split('[')[0],
    "title"  : example_labels_ser[i],
    "embedding": residuals_matrix[i]} for i in range(example_labels_ser.shape[0])])


# %% colab={"base_uri": "https://localhost:8080/"} id="DAsOLh_xQ2v-" outputId="9099f81f-12e8-48a6-b7e2-2ba0d08dd753"
narrative_names = narratives.chapter.to_list()
orthogonal_names = narratives_orth.chapter.to_list()
residuals_names = narratives_complementary.chapter.to_list()
len(narrative_names + orthogonal_names + residuals_names)

# %% id="zlMElVWKdStq"
cosine_sim_all = cosine_similarity(np.vstack([narrative_matrix, orthogonal_matrix, residuals_matrix]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 626} id="WiWtODysQzYX" outputId="983a1fdd-4942-4d03-c64f-32d501b37da9"
import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(1, 1, figsize=(20, 10))
# Heatmap for all vectors N_i and V_i, N_i - V_i
sns.heatmap(
    cosine_sim_all,
    xticklabels=narrative_names + orthogonal_names + residuals_names,
    yticklabels=narrative_names + orthogonal_names + residuals_names,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 6}
)
axes.set_title('Cosine similarities of all vectors (N_i and V_i)', fontsize=12)
axes.tick_params(axis='x', rotation=45, labelsize=8)
axes.tick_params(axis='y', rotation=0, labelsize=8)
plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
plt.setp(axes.get_yticklabels(), rotation=0)

plt.tight_layout()
save_and_download_figure('cosine_similarities_all')


# %% [markdown] id="aPTRP2q90sHq"
# ## 5.2. After an average

# %% colab={"base_uri": "https://localhost:8080/"} id="26a_v75PdFNG" outputId="c08c7854-acd0-41e8-ad8d-8177bfadccbe"
narrative_avg_matrix_unorm = average_cluster_prototypes(
    example_embeddings_df, cluster_col="cluster_id", embedding_col="embedding"
)
narrative_avg_matrix, orthogonal_avg_matrix, residuals_avg_matrix, score = decompose_narratives(
    narrative_avg_matrix_unorm
)
print(narrative_avg_matrix.shape)

# %% colab={"base_uri": "https://localhost:8080/", "height": 176} id="tzmFPks_Iwi6" outputId="dfd1b9e0-cd56-47ba-9577-a8e7aedf2e69"
example_labels_avg_ser = pd.Series([v for _, v in cluster_short_descriptions.items()])
example_labels_avg_ser.sample(3)

# %% id="il00jbXwJceA"
avg_narratives = pd.DataFrame([{
    "chapter": example_labels_avg_ser[i],
    "title"  : "avg narratives",
    "embedding": narrative_avg_matrix[i]} for i in range(example_labels_avg_ser.shape[0])])
avg_narratives_orth = pd.DataFrame([{
    "chapter": 'orth:' + example_labels_avg_ser[i],
    "title"  : "pure (orth) narratives",
    "embedding": orthogonal_avg_matrix[i]} for i in range(example_labels_avg_ser.shape[0])])
avg_narratives_complementary = pd.DataFrame([{
    "chapter": 'residuals:' + example_labels_avg_ser[i],
    "title"  : "residual narratives",
    "embedding": residuals_avg_matrix[i]} for i in range(example_labels_avg_ser.shape[0])])


# %% colab={"base_uri": "https://localhost:8080/"} id="5O1J_NwcJ_Gd" outputId="9b528042-6380-48db-bf2e-1f5782fd4618"
avg_narratives_names = avg_narratives.chapter.to_list()
avg_narratives_orth_names = avg_narratives_orth.chapter.to_list()
avg_residuals_names = avg_narratives_complementary.chapter.to_list()
avg_all_name = avg_narratives_names + avg_narratives_orth_names + avg_residuals_names
len(avg_all_name)

# %% id="yoJWIcpvKLNw"
cosine_sim_all_avg = cosine_similarity(np.vstack([
    narrative_avg_matrix, orthogonal_avg_matrix, residuals_avg_matrix]))

# %% colab={"base_uri": "https://localhost:8080/", "height": 626} id="PC8MawvDKV6l" outputId="a03f7b92-9cb5-4dd1-b03d-9b8ab2a40fd9"
import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(1, 1, figsize=(20, 10))
# Heatmap for all vectors N_i and V_i, N_i - V_i
sns.heatmap(
    cosine_sim_all_avg,
    xticklabels=avg_all_name,
    yticklabels=avg_all_name,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 6}
)
axes.set_title('Cosine similarities of all vectors (N_i and V_i)', fontsize=12)
axes.tick_params(axis='x', rotation=45, labelsize=8)
axes.tick_params(axis='y', rotation=0, labelsize=8)
plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
plt.setp(axes.get_yticklabels(), rotation=0)

plt.tight_layout()
save_and_download_figure('cosine_similarities_avg')


# %% [markdown] id="8GJnj0NtLS6e"
# # Step 6. Monitor narratives on the CNN news

# %% colab={"base_uri": "https://localhost:8080/", "height": 389} id="v6-vGEbvKidX" outputId="cef15675-983f-4583-ce3e-dc2d20d273e8"
cnn_vs_proto_cosine_ar = cosine_similarity(
    [cnn for cnn in loaded_embeddings_df.embedding.values],
    np.vstack([
    narrative_avg_matrix, orthogonal_avg_matrix, residuals_avg_matrix])
)
def create_multi_index(txt):
  txt2 = txt.split(':')
  if len(txt2)==2:
    return (txt2[0], txt2[1])
  return ('original', txt)

cnn_vs_proto_cosine_df = pd.DataFrame(
    cnn_vs_proto_cosine_ar,
    index=loaded_embeddings_df.index,
    columns= pd.MultiIndex.from_tuples([create_multi_index(ind) for ind in avg_all_name]))
cnn_vs_proto_cosine_df.sample(3)

# %% colab={"base_uri": "https://localhost:8080/"} id="YdkWGMfcNABU" outputId="7da53e55-474d-4840-a34a-334ce0e578dd"
avg_cosine = {}
for col_group in cnn_vs_proto_cosine_df.columns.get_level_values(0).unique():
  avg_cosine[col_group] = (
      cnn_vs_proto_cosine_df.loc[:,cnn_vs_proto_cosine_df.columns.get_level_values(0)==col_group]
      .reset_index().drop(columns=['content_order'])
      .groupby('date')
      .mean()
  )

# %% colab={"base_uri": "https://localhost:8080/", "height": 600} id="nQ0Jc7ogQK5i" outputId="3c330802-2c24-4c1d-8cdd-cb17efbbf718"
f, ax = plt.subplots(figsize=(12,6), nrows=3, sharex=True)

for ind, (col_group, df) in enumerate(avg_cosine.items()):
  df.plot(ax=ax[ind], lw=3, alpha=.7, grid=True, title=col_group)
  if ind==0:
    ax[ind].legend(df.columns.get_level_values(1), loc='upper center', bbox_to_anchor=(0.5, 1.75), ncol=3, fancybox=True, shadow=True)
  else:
    ax[ind].get_legend().remove()
plt.tight_layout()

# %% colab={"base_uri": "https://localhost:8080/"} id="mQVzDGNTReRV" outputId="c8f69af9-7835-4c47-c07d-72290156efd0"
replaces = ['Äô', 'Äì', 'Äî', 'Äò', 'Ä¢']

all_extrema = []
for col_group in cnn_vs_proto_cosine_df.columns.get_level_values(0).unique():
  print(f"for {col_group} embeddings:\n--------------")
  df = cnn_vs_proto_cosine_df.loc[:,cnn_vs_proto_cosine_df.columns.get_level_values(0)==col_group]
  for c, ser in df.items():
    print(f">>> for theme {c[1]}")
    ind_max = ser.argmax()
    ind_min = ser.argmin()
    index_max = ser.index[ind_max]
    index_min = ser.index[ind_min]
    value_max = float(ser.iloc[ind_max])
    value_min = float(ser.iloc[ind_min])
    txt_max = loaded_embeddings_df.loc[index_max, 'content']
    for r in replaces:
      txt_max = txt_max.replace(r, '')
    txt_min = loaded_embeddings_df.loc[index_min, 'content']
    for r in replaces:
      txt_min = txt_min.replace(r, '')
    all_extrema.append({
        'col_group': col_group,
        'theme': c[1],

        'date_max': index_max[0],
        'index_max': index_max[1],
        'value_max': value_max,
        'txt_max': txt_max,

        'date_min': index_min[0],
        'index_min': index_min[1],
        'value_min': value_min,
        'txt_min': txt_min
    })
    print(f"... most sensitive news in on date {index_max[0]}")
    print(f"... ... {txt_max}")
    # print(f"... less sensitive news in on date {index_min[0]}")
    # print(f"... {txt_min}")
  print("================================")

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="mDIHgytGUEEH" outputId="2c0e8995-2bbd-40e8-8651-597f01c9bf8e"
all_extrema_df = (
    pd.DataFrame(all_extrema)
    .sort_values(by=['theme', 'col_group'])
    .set_index(['theme', 'col_group'])
    [['date_max', 'index_max', 'value_max', 'txt_max',
      'date_min', 'index_min', 'value_min', 'txt_min']]
)

display(HTML(
  all_extrema_df[['date_max', 'value_max', 'txt_max',
                  'date_min', 'value_min', 'txt_min']].to_html()))

# %% [markdown] id="5-FvjahFc1n1"
# # STEP 6. Validate
#
# Go back to the classification to check all that.

# %% colab={"base_uri": "https://localhost:8080/", "height": 586} id="97x0__lpeMu6" outputId="b853319b-7e72-42ea-a717-536a6329b261"
cnn_vs_proto_cosine_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 658} id="ec7k-hpdHqcP" outputId="f7e3e145-3bd6-4a55-e122-bcf4da2796e6"
# --- 7. Re-plot t-SNE colored by cluster (sanity check) ---------------------
plt.figure(figsize=(10, 8))
sc = plt.scatter(
    loaded_tsne_df["tsne_x"], loaded_tsne_df["tsne_y"],
    c=loaded_tsne_df["cluster"], cmap="tab10", s=10, alpha=0.7
)
plt.grid(True)
plt.title("t-SNE colored by cluster")
plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
plt.tight_layout()
save_and_download_figure('clusters_CNN')

# %% colab={"base_uri": "https://localhost:8080/", "height": 142} id="zgC7Prsib1BO" outputId="88d595fa-d13f-4d3f-fab9-ca65f8735b2a"
cluster_attribution_list = []
for cluster_id in sorted(loaded_tsne_df["cluster"].unique()):
    subset = loaded_tsne_df[loaded_tsne_df["cluster"] == cluster_id]
    for idx, row in subset.iterrows():
      cluster_attribution_list.append({
          'cluster': cluster_id,
          'description': cluster_short_descriptions[cluster_id],
          'date': idx[0],
          'content_order': idx[1],
          'content': row['content']
      })
cluster_attribution_df = pd.DataFrame(cluster_attribution_list)
cluster_attribution_df.sample(3)

# %% colab={"base_uri": "https://localhost:8080/"} id="oN7UogtYdyp3" outputId="5d5ed91f-75c3-4737-ca43-096951a52391"
matched_news_dict = {}
for col_group in cnn_vs_proto_cosine_df.columns.get_level_values(0).unique():
  print(f"for {col_group} embeddings")
  df = cnn_vs_proto_cosine_df.loc[:,cnn_vs_proto_cosine_df.columns.get_level_values(0)==col_group]

  matched_news_dict[col_group] = df.droplevel(level=0, axis=1).merge(
    cluster_attribution_df.set_index(['date', 'content_order']),
    left_index=True, right_index=True)

# %% id="mBzjt9GzfG4G"
compare_scores_list = []
for col_group, df in matched_news_dict.items():
  for ind, row in df.iterrows():
    compare_scores_list.append({
        'col_group': col_group,
        'date': ind[0],
        'content_order': ind[1],
        'content': row['content'],
        'true_cluster': row['description'],
        'score_this_cluster': row[row['description']],
        'other_scores': row.drop(['cluster', 'description','content', row['description']]).values,
    })
compare_scores_df = pd.DataFrame(compare_scores_list)

# %% colab={"base_uri": "https://localhost:8080/", "height": 800} id="tX2FwdtjgD4_" outputId="ab66fcf0-e7f2-4146-bec6-9a8e6fc9b020"
stats_simple_lst = []

f, ax = plt.subplots(figsize=(10,6), ncols=2, nrows=3, sharex=True, sharey=True)
for ind, (cgr, df) in enumerate(compare_scores_df.groupby(['col_group'])):
  gr_in = df[['true_cluster', 'score_this_cluster']].groupby('true_cluster')
  gr_in.boxplot(subplots=False,ax=ax[ind][0], rot=90, ylabel=cgr[0])
  # all others
  all_others_df = pd.DataFrame([
    {'cluster': tc, 'other_scores': np.hstack(_df.other_scores.values)}
    for tc, _df in df[['true_cluster', 'other_scores']].groupby('true_cluster')
  ])
  # Explode the 'other_scores' column, which contains arrays of scores, into individual rows.
  exploded_others_df = all_others_df.set_index('cluster').explode('other_scores')
  # Convert the 'other_scores' column to a numeric type, as explode might leave it as 'object' (because of some empty ones?)
  exploded_others_df['other_scores'] = pd.to_numeric(exploded_others_df['other_scores'])
  # Now, create the boxplot using the numerically typed 'other_scores' column.
  exploded_others_df.reset_index().boxplot(
      column='other_scores', by='cluster', ax=ax[ind][1], rot=90)
  ax[ind][1].set_xlabel('')
  ax[ind][1].set_title('')

  gr_out = exploded_others_df.groupby('cluster')

  stats_simple_lst.append(
      pd.concat([
      gr_in.mean().rename(columns={'score_this_cluster': (cgr[0], 'mean_in')}),
      gr_in.std().rename(columns={'score_this_cluster': (cgr[0], 'std_in')}),
      gr_out.mean().rename(columns={'other_scores': (cgr[0], 'mean_out')}),
      gr_out.std().rename(columns={'other_scores': (cgr[0], 'std_out')}),
      ], axis=1)
  )
#plt.tight_layout()
save_and_download_figure('compare_cosine_orth')

# %% id="JVpmaXL9uorR" colab={"base_uri": "https://localhost:8080/", "height": 807} outputId="a126f3ac-7793-4b09-91e1-baf58c0831e2"
stats_simple_df = pd.concat(stats_simple_lst, axis=1)
stats_simple_df.columns= pd.MultiIndex.from_tuples(stats_simple_df.columns)
stats_simple_df

# %% colab={"base_uri": "https://localhost:8080/", "height": 408} id="vusjd4GS6GTk" outputId="f99e5314-462f-42cf-aa1b-c16cd064e062"
(
    (stats_simple_df.loc[:,(slice(None), 'mean_in')] - stats_simple_df.loc[:,(slice(None), 'mean_out')].values) /
    stats_simple_df.loc[:,(slice(None), 'std_in')].values
).plot(kind='barh', grid=True)
save_and_download_figure('compare_cosine_mean')

# %% colab={"base_uri": "https://localhost:8080/", "height": 759} id="j0GxeePrgk4v" outputId="8b90230d-a872-42a0-f9e9-166dff3c9af7"
compare_scores_df
