"""Synthetic-data tests. The GCN test runs only where PyTorch Geometric is installed."""
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import plantag


def synthetic(n_spots=240, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 3, n_spots)
    mu = rng.gamma(1.0, 1.0, (3, n_genes)) * 2
    X = sp.csr_matrix(rng.poisson(mu[groups]).astype(np.float32))
    obs = pd.DataFrame({"sample": np.where(np.arange(n_spots) < n_spots // 2, "a", "b")}, index=[f"s{i}" for i in range(n_spots)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    markers = {"A": [f"g{i}" for i in np.argsort(-(mu[0] - mu[1:].mean(0)))[:10]],
               "B": [f"g{i}" for i in np.argsort(-(mu[1] - mu[[0, 2]].mean(0)))[:10]],
               "C": [f"g{i}" for i in np.argsort(-(mu[2] - mu[:2].mean(0)))[:10]]}
    return ad.AnnData(X=X, obs=obs, var=var), markers


def test_markers_static_and_json(tmp_path):
    m = plantag.load_markers()
    assert len(m) == 14 and "Mesophyll-Ch" in m
    (tmp_path / "trichome.jsonl").write_text(json.dumps({"symbol": "T1"}) + "\n")
    (tmp_path / "chlorop_random.jsonl").write_text(json.dumps("C1") + "\n")
    m2 = plantag.load_markers(tmp_path)
    assert m2["Trichomes"][0] == "T1" and m2["Mesophyll-Ch"][0] == "C1"


def test_preprocessing_topology_labels():
    adata, markers = synthetic()
    plantag.preprocessing.normalize(adata)
    plantag.preprocessing.select_deviant_genes(adata, n_top=100)
    plantag.preprocessing.pca(adata)
    adata, images, _ = plantag.compute_topology(adata)
    assert adata.obsm["images_vec1"].shape[0] == adata.n_obs and adata.obsm["tda_features_final"].shape[1] == 40
    plantag.labels.training_labels(adata, markers, random_state=0)
    assert set(adata.obs["pseudo_cell_type"].dropna()) <= set(markers)


def test_thinning_reduces_depth():
    adata, _ = synthetic()
    before = np.median(np.asarray(adata.X.sum(1)).ravel())
    thinned = plantag.thin_to_median(adata.copy(), before / 2)
    assert np.median(np.asarray(thinned.X.sum(1)).ravel()) < before


def test_full_pipeline():
    pytest.importorskip("torch_geometric")
    adata, markers = synthetic()
    out, result = plantag.run_plantag(adata, markers, n_top_genes=100, epochs=3, verbose=False)
    assert "predicted_cell_type" in out.obs and out.obsm["X_gnn_embed"].shape[0] == out.n_obs
