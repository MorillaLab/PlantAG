"""
planttag/model/gcn.py
PlantAG GNN architecture — matches the actual notebook implementation.

Architecture confirmed from notebook Cell 98 (get_gnn_embeddings):
    conv1 → conv2 → conv3 → classifier (linear)
All layers use edge_weight (cosine similarity).

Luna*, Aberbache* et al. — MorillaLab / IHSM-UMA-CSIC
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PlantAG(torch.nn.Module):
    """
    Semi-supervised Graph Convolutional Network for plant spatial
    transcriptomics cell-type annotation.

    Input features per cell (from notebook Cell 77):
        X_aug = np.hstack([X_hvg_pca,               # 50-dim  HVG PCA
                           adata.obsm['images_vec1'], # flattened persistence images
                           adata.obsm['tda_features_final_A1']])  # TDA features

    Three GCNConv layers propagate context up to 3 hops.
    Penultimate activations (after conv3) = latent space used for
    diffusion pseudotime and RNA velocity (Cell 98–100).

    Ablation (held-out Leiden clusters):
        1-layer: 82.4%  |  2-layer: 87.3%
        3-layer: 90.1%  |  4-layer: 88.6%

    Args:
        in_channels:     Input feature dimension
        hidden_channels: Hidden dimension (default 64)
        num_classes:     Number of cell types (default 10)
        dropout:         Dropout probability (default 0.5)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        """
        Accept a PyG Data object (matches notebook usage: model(graph_data1)).
        Returns log-softmax probabilities (n_cells, num_classes).
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_weight))
        # x here is the latent space — do NOT dropout before returning embeddings
        latent = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)

    def get_embeddings(self, data, device='cpu'):
        """
        Extract penultimate-layer (conv3) activations.
        Matches notebook Cell 98 get_gnn_embeddings() exactly.

        Returns:
            np.ndarray of shape (n_cells, hidden_channels)
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

    def fit(
        self,
        data,
        train_mask,
        class_weights=None,
        epochs: int = 200,
        lr: float = 0.001,
        patience: int = 30,
        device: str = 'cpu',
    ):
        """
        Training loop matching notebook Cell 78 expectations.

        Args:
            data:          PyG Data object with .x, .edge_index, .edge_weight, .y
            train_mask:    Boolean tensor (n_cells,) — training nodes
            class_weights: Tensor for weighted cross-entropy
            epochs:        Max epochs
            lr:            Adam learning rate
            patience:      Early stopping patience
            device:        'cpu' or 'cuda'
        """
        self.to(device)
        data = data.to(device)
        if class_weights is not None:
            class_weights = class_weights.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr,
                                     betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, factor=0.5)
        criterion = torch.nn.NLLLoss(weight=class_weights)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            out = self(data)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if epoch % 10 == 0:
                        print(f"Early stopping at epoch {epoch + 1} "
                              f"(best loss: {best_loss:.4f})")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1:3d} | loss: {loss.item():.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)

    @torch.no_grad()
    def predict(self, data, device: str = 'cpu'):
        """Return integer class predictions for all cells."""
        self.eval()
        data = data.to(device)
        out = self(data)
        return out.argmax(dim=1).cpu().numpy()

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model
