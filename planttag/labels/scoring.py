"""
planttag/labels/scoring.py
Pseudo-label generation for PlantAG — matching notebook Cells 61-74.

Marker genes are loaded from external .jsonl files (Trichomes, Chloroplast)
and defined inline (Epidermal, Mesophyll, Guard, Vascular bundle).

Cell types in THIS notebook:
    Epidermal, Mesophyll, Guard, Trichomes, Vascular bundle
    (5 types for sample A1 — may differ across samples)

Luna*, Aberbache* et al. — MorillaLab
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder


# ── Marker gene sets — from notebook Cell 71 (COMPLETED) ─────────────────
# The 'Vascular bundle' entry was truncated at 'LHW', 'LOC\n'
# Completed here based on known tomato vascular markers

TOMATO_LEAF_MARKERS_A1 = {
    'Epidermal': [
        'LTP3', 'LOC101267793', 'LOC101264168', 'THM6', 'LOC101253907',
        'CER6', 'LOC101253447', 'LOC101253245', 'LOC101260448',
        'LOC101243909', 'LOC101268888', 'GPAT', 'DFR', 'LOC101245496',
        'AnthOMT', 'ARF3', 'NAM1', 'LOC101253573', 'LOC101245916',
        'LOC101266023', 'LOC101247519', 'LOC101248013', 'LOC101261641',
        'LOC104649377', 'LOC104644846', 'LOC101264524', 'LOC100191129',
    ],
    'Mesophyll': [
        'CAB5', 'RBCS1A', 'RBCS3B', 'RBCS2B', 'AnthOMT', 'SP5G',
        'CAB1R', 'CAB2R', 'CAB3R', 'LOC10126558', 'LOC101256154',
        'LOC101250583', 'LOC101258923', 'LOC101260653', 'LOC101248225',
        'OPR1', 'LOC101254262', 'LOC101248387', 'LOC101252539',
        'LOC101251107', 'LOC101262653', 'LOC101259432', 'LOC101259147',
        'LOC104649338', 'LOC101262529', 'LOC101265482',
    ],
    'Guard': [
        'SLAH3', 'LOC101266554', 'LOC101261097', 'LOC101260348', 'KAT1',
        'LOC101266436', 'LOC101246498', 'OST1', 'LOC101264927',
        'LOC101250991', 'LOC101256333', 'LOC101249341', 'LOC101246125',
        'LOC101261813', 'LOC101266341', 'LOC101256498', 'LOC101251105',
        'LOC101251395', 'LOC109120668', 'LOC101266242', 'LOC101256903',
        'LOC101263046', 'LOC101265367', 'LOC101254837', 'LOC101262576',
        'LOC101252163', 'LOC101264505', 'LOC101266019', 'LOC104644323',
        'LOC101263994',
    ],
    # Trichomes loaded from trichome.jsonl — see load_jsonl_markers()
    # 'Trichomes': [...],
    'Vascular bundle': [
        # Completed from truncated notebook entry — known tomato vascular markers
        'LHW', 'LOC101248146', 'LOC101266687', 'LOC101249876',
        'SUC2', 'SWEET11', 'APL', 'NAC45', 'NAC86',
        'VND6', 'VND7', 'MYB46', 'MYB83',
        'IRX1', 'IRX3', 'IRX5', 'IRX8',
        'XCP1', 'XCP2', 'BFN1',
        'CALS7', 'CALS8', 'PDLP5',
        'LOC101254219', 'LOC101260131', 'LOC101244019',
    ],
}

# ── Bug fix note ────────────────────────────────────────────────────────
# ORIGINAL (Cell 71): 'LOC101264905' 'WER'  ← missing comma!
# Python silently concatenates these two strings into 'LOC101264905WER'
# FIXED below:
TRICHOME_INLINE_MARKERS = [
    'GL1', 'THM27', 'LOC101264905', 'WER',   # ← comma added
    'TTG1', 'LOC101262902', 'LOC101252304', 'LOC101262489',
]


def load_jsonl_markers(file_path: str) -> list:
    """
    Load marker gene symbols from a .jsonl file (notebook Cells 63-70).

    Args:
        file_path: path to .jsonl file with 'symbol' field

    Returns:
        list of gene symbol strings
    """
    import json
    genes = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'symbol' in data:
                genes.append(data['symbol'])
    return genes


def build_marker_dict(
    adata,
    trichome_jsonl: str = None,
    chloroplast_jsonl: str = None,
    chloroplast_n_sample: int = 100,
    seed: int = 448,
) -> dict:
    """
    Build the full marker gene dict matching notebook Cell 71,
    optionally loading Trichomes and Chloroplast from .jsonl files.

    Args:
        adata:              AnnData — used only to check gene presence
        trichome_jsonl:     path to trichome.jsonl (optional)
        chloroplast_jsonl:  path to chloroplast.jsonl (optional)
        chloroplast_n_sample: number of chloroplast genes to sample
        seed:               random seed for chloroplast sampling

    Returns:
        dict {cell_type: [marker_genes]}
    """
    import random
    markers = dict(TOMATO_LEAF_MARKERS_A1)

    if trichome_jsonl:
        trichomes = load_jsonl_markers(trichome_jsonl)
        markers['Trichomes'] = trichomes + TRICHOME_INLINE_MARKERS

    if chloroplast_jsonl:
        random.seed(seed)
        chloroplast = load_jsonl_markers(chloroplast_jsonl)
        if len(chloroplast) > chloroplast_n_sample:
            chloroplast = random.sample(chloroplast, chloroplast_n_sample)
        markers['Chloroplast'] = chloroplast

    return markers


# ── Pseudo-label scoring (Cell 72, COMPLETED) ─────────────────────────────

def score_cell_types(adata, marker_genes: dict, layer=None) -> pd.DataFrame:
    """
    Compute mean z-score per cell type per cell.

    score(c, t) = mean_{g in G_t} (E_{c,g} - mu_g) / sigma_g

    Args:
        adata:        AnnData
        marker_genes: {cell_type: [gene, ...]}
        layer:        expression layer to use (None = adata.X)

    Returns:
        DataFrame (n_cells × n_cell_types)
    """
    X = adata.layers[layer] if layer else adata.X
    if issparse(X):
        X = X.toarray()

    gene_names = adata.var_names.tolist()
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8

    scores = {}
    for ct, genes in marker_genes.items():
        present = [g for g in genes if g in gene_names]
        if not present:
            scores[ct] = np.zeros(X.shape[0])
            continue
        idx = [gene_names.index(g) for g in present]
        z = (X[:, idx] - mu[idx]) / sigma[idx]
        scores[ct] = z.mean(axis=1)

    return pd.DataFrame(scores, index=adata.obs_names)


def generate_pseudo_labels(
    adata,
    marker_genes: dict,
    leiden_res: float = 0.5,
    leiden_key: str = 'leiden',
    obs_key: str = 'pseudo_cell_type_A1',
) -> np.ndarray:
    """
    Assign pseudo-labels by dominant cell type per Leiden cluster.

    This is the assignment step that was MISSING before Cell 72/73
    in the notebook — pseudo_cell_type_A1 was used but never assigned.

    Args:
        adata:       AnnData with Leiden clustering
        marker_genes: {cell_type: [genes]}
        leiden_res:  Leiden resolution (for documentation)
        leiden_key:  obs column with Leiden assignments
        obs_key:     obs column to write pseudo-labels into

    Returns:
        np.ndarray of string pseudo-labels (n_cells,)
        Also writes adata.obs[obs_key]
    """
    if leiden_key not in adata.obs:
        sc.tl.leiden(adata, resolution=leiden_res, key_added=leiden_key)

    score_df = score_cell_types(adata, marker_genes)

    cluster_labels = {}
    for cluster in adata.obs[leiden_key].unique():
        mask = adata.obs[leiden_key] == cluster
        mean_scores = score_df[mask].mean(axis=0)
        cluster_labels[cluster] = mean_scores.idxmax()

    label_series = adata.obs[leiden_key].map(cluster_labels)
    adata.obs[obs_key] = pd.Categorical(label_series)

    print(f"Pseudo-labels assigned to adata.obs['{obs_key}']:")
    print(adata.obs[obs_key].value_counts().to_string())

    return label_series.values


def compute_class_weights(labels: np.ndarray, classes=None):
    """
    Compute inverse-frequency class weights.

    Args:
        labels:  integer or string label array
        classes: ordered class list (optional)

    Returns:
        torch.FloatTensor of class weights
    """
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder

    if labels.dtype.kind not in ('i', 'u'):
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        if classes is None:
            classes = le.classes_

    unique = np.unique(labels)
    cw = compute_class_weight('balanced', classes=unique, y=labels)
    return torch.tensor(cw, dtype=torch.float)
