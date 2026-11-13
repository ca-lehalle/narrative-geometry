"""
Embedding utilities for Hugging Face AutoModel (ModernBERT) and SentenceTransformer,
plus prototype averaging across thematic clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def load_model(model_name: str = "answerdotai/ModernBERT-base", device: str | torch.device | None = None):
    """Load Hugging Face tokenizer and encoder model on the target device."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    return tokenizer, model, device


def get_embedding(
    text: str,
    tokenizer,
    model,
    device: torch.device | str,
    max_length: int = 512,
) -> np.ndarray:
    """Compute mean-pooled embedding for a single text string using Hugging Face AutoModel."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy().flatten()


def average_cluster_prototypes(
    example_embeddings_df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    embedding_col: str = "embedding",
) -> np.ndarray:
    """
    Average representative narrative embeddings within each cluster into a single
    prototype vector per cluster, returning a 2D array of shape (K, dim).
    """
    avg_series = example_embeddings_df.groupby(cluster_col)[embedding_col].apply(
        lambda s: np.mean(np.vstack(s.to_numpy()), axis=0)
    )
    return np.vstack(avg_series.to_numpy()).astype(float)
