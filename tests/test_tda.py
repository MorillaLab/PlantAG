"""Tests for TDA feature extraction."""
import numpy as np
import pytest

def test_features_from_image_shape():
    from planttag.tda.features import features_from_image
    img = np.random.rand(10, 10)
    feats = features_from_image(img)
    assert feats.shape == (8,), f"Expected (8,), got {feats.shape}"

def test_features_from_image_empty():
    from planttag.tda.features import features_from_image
    img = np.zeros((5, 5))
    feats = features_from_image(img)
    assert feats.shape == (8,)

def test_clean_diagrams_no_inf():
    from planttag.tda.features import clean_diagrams
    diagrams = [[(0, (0.1, float('inf'))), (0, (0.2, 0.5)), (1, (0.3, 0.8))]]
    cleaned = clean_diagrams(diagrams)
    for diag in cleaned:
        assert np.all(np.isfinite(diag)), "Infinite values remain after cleaning"

def test_diagrams_birth_to_death():
    from planttag.tda.features import diagrams_birth_to_death
    clean = [np.array([[0.1, 0.4], [0.2, 0.3]])]  # (birth, lifetime)
    bd = diagrams_birth_to_death(clean)
    assert bd[0][0, 1] > bd[0][0, 0], "death must be > birth"

def test_compute_final_tda_features_shape():
    from planttag.tda.features import compute_final_tda_features_filtered_only
    import anndata as ad
    n_cells, n_genes = 20, 50
    adata = ad.AnnData(np.random.rand(n_cells, n_genes))
    images = [np.random.rand(10, 10) for _ in range(n_cells)]
    adata = compute_final_tda_features_filtered_only(adata, images, n_levels=5)
    expected_cols = 8 + 8 * 5  # 48
    assert adata.obsm['tda_features_final_A1'].shape == (n_cells, expected_cols)
