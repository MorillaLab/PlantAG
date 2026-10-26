"""Graph construction and graph convolutional network (verbatim from the PlantAG notebook, cells 87 and 109;
the only change is the optional random_state passed to the PCA in build_weighted_graph)."""
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def build_weighted_graph(adata, n_neighbors=30, n_hvg_pca=50, random_state=None):
    # --- 1️⃣ HVG ---
    hvg = adata.var_names[adata.var['highly_variable']].tolist()
    if issparse(adata.X):
        X_hvg = adata[:, hvg].X.toarray()
    else:
        X_hvg = adata[:, hvg].X

    # Réduction dimensionnelle HVG avec PCA pour limiter le bruit
    X_hvg_scaled = StandardScaler().fit_transform(X_hvg)
    pca = PCA(n_components=min(n_hvg_pca, X_hvg_scaled.shape[1]), random_state=random_state)
    X_hvg_pca = pca.fit_transform(X_hvg_scaled)

    # --- 2️⃣ Concaténation avec images aplaties et TDA features finales ---
    X_aug = np.hstack([X_hvg_pca, adata.obsm['images_vec1'], adata.obsm['tda_features_final']])
    X_aug = StandardScaler().fit_transform(X_aug)

    # --- 3️⃣ k-NN graph ---
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(X_aug)
    distances, indices = nbrs.kneighbors(X_aug)

    edge_index = []
    edge_weight = []
    for i in range(len(indices)):
        for j_idx, j in enumerate(indices[i]):
            edge_index.append([i, j])
            edge_weight.append(1.0 - distances[i][j_idx])  # cosine similarity

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # --- 4️⃣ Labels ---
    y = pd.Categorical(adata.obs['pseudo_cell_type']).codes
    y = torch.tensor(y, dtype=torch.long)

    # --- 5️⃣ Coordonnées spatiales (optionnel) ---
    pos = torch.tensor(adata.obsm['spatial'], dtype=torch.float) if 'spatial' in adata.obsm else None

    return Data(
        x=torch.tensor(X_aug, dtype=torch.float),
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        pos=pos
    )


class PlantCellGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_weight', None)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        return self.classifier(x)


def get_gnn_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index, edge_weight = data.x, data.edge_index, getattr(data, 'edge_weight', None)
        x = torch.relu(model.conv1(x, edge_index, edge_weight))
        x = torch.dropout(x, p=model.dropout, train=False)
        x = torch.relu(model.conv2(x, edge_index, edge_weight))
        x = torch.dropout(x, p=model.dropout, train=False)
        x = torch.relu(model.conv3(x, edge_index, edge_weight))
        # x = représentation finale (n_cells, hidden_dim)
        return x.cpu().numpy()
