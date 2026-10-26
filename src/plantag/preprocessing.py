"""Normalisation, deviant-gene selection and PCA (original notebook, cells 8, 23-30)."""
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

from ._residuals import pearson_residuals


def normalize(adata):
    """Library-size normalisation and log1p into layer 'log1p_norm' (cell 8)."""
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)
    return adata


def select_deviant_genes(adata, n_top=4000, theta=100):
    """Analytic Pearson residuals of the raw counts in adata.X and the n_top genes with the largest summed squared
    residuals (cells 23-27): var['highly_deviant'], var['deviance_score']."""
    adata.layers["analytic_pearson_residuals"] = csr_matrix(pearson_residuals(adata.X, theta))
    X_resid = adata.layers["analytic_pearson_residuals"]
    if hasattr(X_resid, "toarray"):
        X_resid = X_resid.toarray()
    gene_scores = np.sum(X_resid ** 2, axis=0)
    idx = np.argsort(gene_scores)[-n_top:]
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[idx] = True
    adata.var["highly_deviant"] = mask
    adata.var["deviance_score"] = gene_scores
    return adata


def pca(adata):
    """log1p layer as X, deviant genes as highly variable, PCA with ARPACK (cells 29-30)."""
    adata.X = adata.layers["log1p_norm"]
    adata.var["highly_variable"] = adata.var["highly_deviant"]
    try:
        sc.pp.pca(adata, svd_solver="arpack", mask_var="highly_variable")
    except TypeError:                                   # scanpy < 1.10
        sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=True)
    return adata
