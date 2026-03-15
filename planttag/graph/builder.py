"""
planttag/graph/builder.py
Weighted k-NN graph construction for PlantAG.

Matches notebook Cell 72/77 build_weighted_graph() — completed with the
truncated label encoding, train/test split, class weights, and PyG Data object.

Feature concatenation order (from notebook):
    X_aug = np.hstack([X_hvg_pca,                        # 50-dim
                       adata.obsm['images_vec1'],          # flattened images
                       adata.obsm['tda_features_final_A1'] # TDA features])

Luna*, Aberbache* et al. — MorillaLab
"""

import numpy as np
import torch
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data


def build_weighted_graph(
    adata,
    label_key: str = 'pseudo_cell_type_A1',
    n_neighbors: int = 30,
    n_hvg_pca: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build a weighted k-NN graph from HVG PCA + persistence images + TDA features.

    This is the COMPLETE version of the truncated Cell 72/77 function.

    Steps:
        1. Extract HVG expression matrix, apply PCA (50 components)
        2. Concatenate with images_vec1 and tda_features_final_A1
        3. Standardise the concatenated feature matrix
        4. Build k-NN graph with cosine similarity edge weights
        5. Encode pseudo-labels as integers
        6. Create train/test masks (stratified)
        7. Compute inverse-frequency class weights for weighted cross-entropy

    Args:
        adata:        AnnData with highly_variable, images_vec1,
                      tda_features_final_A1, and pseudo-label column
        label_key:    adata.obs column containing pseudo-labels
        n_neighbors:  k for k-NN graph
        n_hvg_pca:    PCA components for HVG reduction
        test_size:    Fraction of cells held out for evaluation
        random_state: Random seed

    Returns:
        graph_data:   torch_geometric.data.Data with x, edge_index,
                      edge_weight, y
        train_mask:   Boolean tensor (n_cells,)
        test_mask:    Boolean tensor (n_cells,)
        class_weights: FloatTensor for weighted NLLLoss
        label_encoder: fitted LabelEncoder (to recover class names)
    """

    # ── 1. HVG expression → PCA ──────────────────────────────────────────
    hvg_mask = adata.var['highly_variable']
    X_hvg = adata[:, hvg_mask].X
    if issparse(X_hvg):
        X_hvg = X_hvg.toarray()

    X_hvg_scaled = StandardScaler().fit_transform(X_hvg)
    pca = PCA(n_components=min(n_hvg_pca, X_hvg_scaled.shape[1]),
              random_state=random_state)
    X_hvg_pca = pca.fit_transform(X_hvg_scaled)   # (n_cells, 50)

    # ── 2. Concatenate features ───────────────────────────────────────────
    X_aug = np.hstack([
        X_hvg_pca,                             # HVG PCA
        adata.obsm['images_vec1'],              # flattened persistence images
        adata.obsm['tda_features_final_A1'],    # TDA scalar features
    ])                                           # (n_cells, 50 + img_dim + tda_dim)
    X_aug = StandardScaler().fit_transform(X_aug)

    # ── 3. k-NN with cosine similarity ───────────────────────────────────
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                             metric='cosine').fit(X_aug)
    distances, indices = nbrs.kneighbors(X_aug)

    edge_index_list, edge_weight_list = [], []
    for i in range(len(indices)):
        for j_idx, j in enumerate(indices[i]):
            edge_index_list.append([i, j])
            edge_weight_list.append(1.0 - distances[i][j_idx])   # cosine sim

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    # ── 4. Label encoding ── COMPLETING the truncated Cell 72/77 ─────────
    raw_labels = adata.obs[label_key].values

    le = LabelEncoder()
    y_int = le.fit_transform(raw_labels)    # integer labels 0..n_classes-1
    y = torch.tensor(y_int, dtype=torch.long)

    # ── 5. Node feature tensor ────────────────────────────────────────────
    x = torch.tensor(X_aug, dtype=torch.float)

    # ── 6. Train / test masks (stratified) ───────────────────────────────
    n_cells = x.shape[0]
    idx = np.arange(n_cells)
    idx_train, idx_test = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_int,
        random_state=random_state,
    )
    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    test_mask  = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[idx_train] = True
    test_mask[idx_test]   = True

    # ── 7. Class weights ──────────────────────────────────────────────────
    classes = np.unique(y_int)
    cw = compute_class_weight('balanced', classes=classes, y=y_int)
    class_weights = torch.tensor(cw, dtype=torch.float)

    # ── 8. Assemble PyG Data object ───────────────────────────────────────
    graph_data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
    )

    print(f"Graph built: {graph_data.num_nodes} nodes | "
          f"{graph_data.num_edges} edges | "
          f"feature dim={x.shape[1]} | "
          f"{len(le.classes_)} classes")
    print(f"Train: {train_mask.sum().item()}  Test: {test_mask.sum().item()}")

    return graph_data, train_mask, test_mask, class_weights, le
