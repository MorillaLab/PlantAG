"""Tests for preprocessing bug fixes."""
import numpy as np
import scipy.sparse as sp
import pytest

def test_pearson_residuals_sparse_input():
    from planttag.preprocessing import pearson_residuals
    X_sparse = sp.random(50, 100, density=0.3, format='csr')
    result = pearson_residuals(X_sparse, theta=100)
    assert result.shape == (50, 100)
    assert np.all(np.isfinite(result))

def test_pearson_residuals_dense_input():
    from planttag.preprocessing import pearson_residuals
    X = np.random.poisson(5, (50, 100)).astype(float)
    result = pearson_residuals(X, theta=100)
    assert result.shape == (50, 100)

def test_apply_scran_normalisation():
    from planttag.preprocessing import apply_scran_normalisation
    import anndata as ad
    n, g = 30, 50
    adata = ad.AnnData(sp.random(n, g, density=0.5, format='csr'))
    adata.obs['size_factors'] = np.random.uniform(0.5, 2.0, n)
    adata = apply_scran_normalisation(adata)
    assert 'scran_normalization' in adata.layers
    assert adata.layers['scran_normalization'].shape == (n, g)
