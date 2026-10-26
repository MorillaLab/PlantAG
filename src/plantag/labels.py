"""Training labels: marker scores over Leiden clusters of the persistence images (original notebook, cell 81)."""
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def marker_scores(adata, marker_genes):
    """Mean z-scored expression of the markers of each list that are present in adata (spots x types)."""
    marker_genes_in_data = {ct: [g for g in gs if g in adata.var_names] for ct, gs in marker_genes.items()}
    cell_scores = pd.DataFrame(index=adata.obs_names, columns=marker_genes_in_data.keys(), dtype=float)
    for ct, genes in marker_genes_in_data.items():
        if not genes:
            continue
        expr = adata[:, genes].X
        if issparse(expr):
            expr = expr.toarray()
        expr_z = (expr - expr.mean(axis=0)) / expr.std(axis=0)
        expr_z = np.nan_to_num(expr_z)
        cell_scores[ct] = expr_z.mean(axis=1)
    return cell_scores


def training_labels(adata, marker_genes, n_pcs=50, n_neighbors=30, resolution=1.5, random_state=None):
    """obsm['X_tda_pca'] (PCA of the standardised persistence images), obs['tda_leiden'] and obs['pseudo_cell_type']
    (the type with the highest mean marker score in each Leiden cluster)."""
    cell_scores = marker_scores(adata, marker_genes)
    X_tda_scaled = StandardScaler().fit_transform(adata.obsm["images_vec1"])
    adata.obsm["X_tda_pca"] = PCA(n_components=n_pcs, random_state=random_state).fit_transform(X_tda_scaled)
    sc.pp.neighbors(adata, use_rep="X_tda_pca", n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=resolution, key_added="tda_leiden")
    cluster_labels = {}
    for cl in adata.obs["tda_leiden"].cat.categories:
        cells = adata.obs["tda_leiden"] == cl
        cluster_labels[cl] = cell_scores.loc[cells].mean(axis=0).idxmax()
    adata.obs["pseudo_cell_type"] = adata.obs["tda_leiden"].map(cluster_labels)
    return cell_scores
