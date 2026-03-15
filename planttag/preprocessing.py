"""
planttag/preprocessing.py
Preprocessing utilities for PlantAG — with all notebook bugs fixed.

Bugs fixed:
  - Cell 20: sparse matrix division by size_factors → explicit toarray()
  - Cell 25: pearson_residuals called on sparse adata.X → toarray()
  - Cell 117: spliced/unspliced layer assignment without barcode alignment

Luna*, Aberbache* et al. — MorillaLab
"""

import numpy as np
from scipy.sparse import issparse, csr_matrix


# ── BUG FIX: Cell 24 — pearson_residuals ─────────────────────────────────
# Original: pearson_residuals(adata.X, 100)
# adata.X is sparse at this point → np.sum fails on sparse input
# Fix: always convert to dense before computing residuals

def pearson_residuals(counts, theta: float, clipping: bool = True):
    """
    Compute analytical Pearson residuals for NB model with fixed theta.

    BUG FIX from notebook Cell 25:
        Original call was pearson_residuals(adata.X, 100) where adata.X
        is a sparse matrix. np.sum(sparse) returns a matrix, not an array,
        causing incorrect broadcasting in the mu computation.
        This version handles both sparse and dense input.

    Args:
        counts:   (n_cells, n_genes) expression matrix — sparse or dense
        theta:    NB dispersion parameter
        clipping: clip residuals to ±sqrt(n)

    Returns:
        np.ndarray (n_cells, n_genes) of Pearson residuals
    """
    # Convert sparse to dense
    if issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=float)

    counts_sum0 = np.sum(counts, axis=0)   # sum per gene
    counts_sum1 = np.sum(counts, axis=1)   # sum per cell
    counts_sum  = np.sum(counts)

    mu = np.outer(counts_sum1, counts_sum0) / counts_sum
    z = (counts - mu) / np.sqrt(mu + mu / theta + 0.01)

    if clipping:
        n = counts.shape[0]
        z = np.clip(z, -np.sqrt(n), np.sqrt(n))

    return z


# ── BUG FIX: Cell 20 — Scran normalisation ───────────────────────────────
# Original: scran = adata.X / adata.obs["size_factors"].values[:, None]
# If adata.X is sparse, this produces a dense matrix via implicit conversion
# but the result is incorrect — dividing a CSR matrix by a vector does not
# broadcast correctly in all scipy versions.
# Fix: explicit toarray() before division.

def apply_scran_normalisation(adata, size_factors_key: str = 'size_factors'):
    """
    Apply Scran size-factor normalisation and store as layer.

    BUG FIX from notebook Cell 20:
        Sparse matrix division requires explicit dense conversion.

    Args:
        adata:             AnnData with size_factors in .obs
        size_factors_key:  obs column containing size factors

    Returns:
        adata with 'scran_normalization' layer added
    """
    size_factors = adata.obs[size_factors_key].values

    X = adata.X
    if issparse(X):
        X = X.toarray()

    # Safe division: explicit float array / column vector
    scran = X.astype(float) / size_factors[:, None]
    log_scran = np.log1p(scran)
    adata.layers['scran_normalization'] = csr_matrix(log_scran)

    return adata


# ── BUG FIX: Cell 117 — RNA velocity barcode alignment ───────────────────
# Original:
#     adata.layers['spliced']   = vlm.layers['spliced'][:].tocsr()
#     adata.layers['unspliced'] = vlm.layers['unspliced'][:].tocsr()
# This assumes vlm and adata have barcodes in the same order — no check.
# If the loom file has different barcode ordering or extra barcodes,
# the layers are silently misaligned.

def align_velocity_layers(adata, vlm):
    """
    Safely assign spliced/unspliced layers from a loom AnnData to adata,
    aligning by barcode.

    BUG FIX from notebook Cell 117:
        Direct layer assignment vlm.layers['spliced'][:] assumes identical
        barcode ordering. This function aligns explicitly.

    Args:
        adata: main AnnData (Visium)
        vlm:   loom-loaded AnnData with spliced/unspliced layers

    Returns:
        adata with 'spliced' and 'unspliced' layers aligned to its barcodes
    """
    # Find common barcodes
    common_barcodes = adata.obs_names.intersection(vlm.obs_names)

    if len(common_barcodes) == 0:
        raise ValueError(
            "No common barcodes between adata and vlm. "
            "Check that barcode formats match (e.g. trailing -1 suffix)."
        )

    if len(common_barcodes) < len(adata.obs_names):
        n_missing = len(adata.obs_names) - len(common_barcodes)
        print(f"Warning: {n_missing} barcodes in adata not found in loom file. "
              f"Subsetting adata to {len(common_barcodes)} common barcodes.")
        adata = adata[common_barcodes].copy()

    # Align vlm to adata barcode order
    vlm_aligned = vlm[adata.obs_names]

    spliced   = vlm_aligned.layers['spliced']
    unspliced = vlm_aligned.layers['unspliced']

    adata.layers['spliced']   = spliced.tocsr()   if issparse(spliced)   else csr_matrix(spliced)
    adata.layers['unspliced'] = unspliced.tocsr() if issparse(unspliced) else csr_matrix(unspliced)

    print(f"Velocity layers aligned: {len(adata.obs_names)} barcodes, "
          f"{adata.n_vars} genes")
    return adata
