"""
planttag/model/gcn.py
PlantAG GNN — PlantCellGNN architecture matching the actual notebook (Cell 87).

CORRECTED parameters vs earlier versions:
  hidden_dim : 256  (was 64)
  lr         : 0.005 with weight_decay=1e-4  (was 0.001)
  epochs     : 100  (was 200)
  loss       : nn.CrossEntropyLoss  (was NLLLoss; no log-softmax in forward)
  obsm key   : 'tda_features_final'  (was 'tda_features_final_A1')
  obs key    : 'pseudo_cell_type'  (was 'pseudo_cell_type_A1')

Luna*, Aberbache* et al. — MorillaLab / IHSM-UMA-CSIC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PlantAG(nn.Module):
    """
    Semi-supervised Graph Convolutional Network for plant spatial
    transcriptomics cell-type annotation.

    Three GCNConv layers + linear classifier.
    forward() returns raw logits (no log-softmax) — use CrossEntropyLoss.

    Ablation (held-out Leiden clusters):
        1-layer: 82.4%  |  2-layer: 87.3%
        3-layer: 90.1%  |  4-layer: 88.6%

    Args:
        input_dim:   Input feature dimension
        hidden_dim:  Hidden dimension (default 256)
        num_classes: Number of cell types
        dropout:     Dropout probability (default 0.5)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        """
        Args:
            data: PyG Data object with .x, .edge_index, .edge_weight

        Returns:
            Tensor (n_cells, num_classes) — raw logits for CrossEntropyLoss
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3 — penultimate activations = 256-dim latent space
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        latent = x

        return self.classifier(latent)   # raw logits, no log-softmax

    def get_embeddings(self, data, device='cpu'):
        """
        Extract 256-dim penultimate-layer (conv3) activations.
        Stored as adata.obsm['X_gnn_embed'].

        Returns:
            np.ndarray (n_cells, hidden_dim)
        """
        self.eval()
        self.to(device)
        data = data.to(device)
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            edge_weight = getattr(data, 'edge_weight', None)
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, train=False)
            x = F.relu(self.conv2(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, train=False)
            x = F.relu(self.conv3(x, edge_index, edge_weight))
        return x.cpu().numpy()

    @torch.no_grad()
    def predict(self, data, device='cpu'):
        """Return integer class predictions."""
        self.eval()
        data = data.to(device)
        logits = self(data)
        return logits.argmax(dim=1).cpu().numpy()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
