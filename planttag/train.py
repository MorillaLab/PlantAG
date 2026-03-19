"""
planttag/train.py
train() and test() functions for PlantAG GNN.

These were CALLED in notebook Cell 78 but never defined anywhere.
Defined here to match the notebook's usage pattern.

Luna*, Aberbache* et al. — MorillaLab
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def make_train_test_functions(model, graph_data, train_mask, test_mask,
                               class_weights=None, lr=0.001, device='cpu'):
    """
    Factory that returns train() and test() closures matching
    notebook Cell 78 usage:

        acc_test_final = test()

    Args:
        model:         PlantAG instance
        graph_data:    PyG Data object
        train_mask:    Boolean tensor
        test_mask:     Boolean tensor
        class_weights: FloatTensor for weighted NLLLoss
        lr:            Adam learning rate
        device:        'cpu' or 'cuda'

    Returns:
        train: callable() → float training loss
        test:  callable() → float test accuracy
    """
    model.to(device)
    graph_data = graph_data.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def test():
        model.eval()
        out = model(graph_data)
        preds = out.argmax(dim=1)
        y_true = graph_data.y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        return float(accuracy_score(y_true, y_pred))

    return train, test


def run_training(
    model,
    graph_data,
    train_mask,
    test_mask,
    class_weights=None,
    epochs: int = 100,
    lr: float = 0.005,
    patience: int = 30,
    device: str = 'cpu',
    verbose: bool = True,
):
    """
    Full training loop with early stopping.

    Equivalent to running the training loop manually as the notebook
    implies but never explicitly defines.

    Args:
        model:         PlantAG instance
        graph_data:    PyG Data object
        train_mask:    Boolean tensor
        test_mask:     Boolean tensor
        class_weights: FloatTensor
        epochs:        Max epochs
        lr:            Learning rate
        patience:      Early stopping patience
        device:        'cpu' or 'cuda'
        verbose:       Print progress every 20 epochs

    Returns:
        dict with 'train_losses' and 'test_accuracies' lists
    """
    train_fn, test_fn = make_train_test_functions(
        model, graph_data, train_mask, test_mask,
        class_weights=class_weights, lr=lr, device=device,
    )

    # LR scheduler requires direct access to optimizer
    # (re-wiring here for the scheduler)
    model.to(device)
    graph_data = graph_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, verbose=False)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    train_losses, test_accuracies = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        train_losses.append(float(loss.item()))

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best loss: {best_loss:.4f})")
                break

        if verbose and epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(graph_data).argmax(dim=1)
            acc = accuracy_score(
                graph_data.y[test_mask].cpu().numpy(),
                preds[test_mask].cpu().numpy()
            )
            test_accuracies.append(acc)
            print(f"Epoch {epoch:3d} | loss: {loss.item():.4f} | "
                  f"test acc: {acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {'train_losses': train_losses, 'test_accuracies': test_accuracies}
