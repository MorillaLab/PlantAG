"""Tests for graph construction."""
import numpy as np
import pytest

def test_build_weighted_graph_structure():
    from planttag.graph.builder import build_weighted_graph
    import anndata as ad
    import scipy.sparse as sp

    n, g = 40, 200
    X = sp.random(n, g, density=0.3, format='csr')
    adata = ad.AnnData(X.toarray())
    adata.obs['pseudo_cell_type_A1'] = (['TypeA'] * 20 + ['TypeB'] * 20)
    adata.var['highly_variable'] = np.ones(g, dtype=bool)
    adata.obsm['images_vec1'] = np.random.rand(n, 20)
    adata.obsm['tda_features_final_A1'] = np.random.rand(n, 48)

    graph, train_mask, test_mask, cw, le = build_weighted_graph(adata, n_neighbors=5)

    assert graph.num_nodes == n
    assert graph.edge_index.shape[0] == 2
    assert train_mask.sum() + test_mask.sum() == n
    assert len(le.classes_) == 2
