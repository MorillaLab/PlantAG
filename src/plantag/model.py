"""Graph convolutional network: training, prediction and latent embedding (original notebook, cells 87-110).

PyTorch and PyTorch Geometric are imported here only, so the rest of PlantAG works without them."""
import numpy as np
import pandas as pd


def fit_predict(adata, epochs=100, hidden_dim=256, dropout=0.5, lr=0.005, weight_decay=1e-4, test_size=0.2,
                split_seed=42, seed=0, device="cpu", verbose=True):
    """Train on a stratified (1 - test_size) share of spots, predict every spot and store obs['predicted_cell_type']
    and obsm['X_gnn_embed']. Returns the model, the per-epoch history and the accuracy on the held-out spots."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from torch_geometric.loader import DataLoader

    from ._network import PlantCellGNN, build_weighted_graph, get_gnn_embeddings

    torch.manual_seed(seed)
    np.random.seed(seed)
    graph_data1 = build_weighted_graph(adata, random_state=seed)
    encoded_labels = graph_data1.y
    num_classes = len(np.unique(encoded_labels))
    train_idx, test_idx = train_test_split(np.arange(len(encoded_labels)), test_size=test_size,
                                           stratify=encoded_labels, random_state=split_seed)
    train_mask = torch.zeros(len(encoded_labels), dtype=torch.bool); train_mask[train_idx] = True
    test_mask = torch.zeros(len(encoded_labels), dtype=torch.bool); test_mask[test_idx] = True
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(encoded_labels), y=encoded_labels.numpy())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
    torch.manual_seed(seed)
    model = PlantCellGNN(input_dim=graph_data1.x.shape[1], hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader([graph_data1], batch_size=1)
    test_loader = DataLoader([graph_data1], batch_size=1)

    def train():
        model.train(); total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward(); optimizer.step(); total_loss += loss.item()
        return total_loss / len(train_loader)

    def test():
        model.eval(); correct = 0
        with torch.no_grad():
            for data in test_loader:
                pred = model(data).argmax(dim=1)
                correct += (pred[test_mask] == data.y[test_mask]).sum().item()
        return correct / test_mask.sum().item()

    history = []
    for epoch in range(epochs):
        loss = train(); acc = test(); history.append({"epoch": epoch + 1, "loss": loss, "held_out_accuracy": acc})
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Epoch {epoch + 1:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    model.eval()
    dev = torch.device(device)
    graph_data1 = graph_data1.to(dev); model = model.to(dev)
    with torch.no_grad():
        preds = model(graph_data1).argmax(dim=1).cpu().numpy()
    categories = pd.Categorical(adata.obs["pseudo_cell_type"]).categories
    adata.obs["predicted_cell_type"] = pd.Categorical.from_codes(preds, categories)
    adata.obsm["X_gnn_embed"] = get_gnn_embeddings(model, graph_data1)
    return {"model": model, "history": pd.DataFrame(history), "held_out_accuracy": history[-1]["held_out_accuracy"] if history else None}
