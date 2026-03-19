"""Tests for PlantAG GNN."""
import torch
import pytest
from torch_geometric.data import Data

def make_graph(n=30, f=100, c=5):
    x = torch.randn(n, f)
    src = torch.randint(0, n, (90,))
    dst = torch.randint(0, n, (90,))
    y = torch.randint(0, c, (n,))
    return Data(x=x, edge_index=torch.stack([src, dst]),
                edge_weight=torch.rand(90), y=y), n, c

def test_forward_output_shape():
    from planttag.model.gcn import PlantAG
    graph, n, c = make_graph()
    model = PlantAG(input_dim=100, num_classes=c)
    out = model(graph)
    assert out.shape == (n, c)

def test_log_softmax():
    from planttag.model.gcn import PlantAG
    graph, n, c = make_graph()
    model = PlantAG(input_dim=100, num_classes=c)
    out = model(graph)
    assert torch.allclose(out.exp().sum(dim=1), torch.ones(n), atol=1e-5)

def test_get_embeddings_shape():
    from planttag.model.gcn import PlantAG
    graph, n, c = make_graph()
    model = PlantAG(input_dim=100, hidden_channels=64, num_classes=c)
    emb = model.get_embeddings(graph)
    assert emb.shape == (n, 64)

def test_predict_valid_range():
    from planttag.model.gcn import PlantAG
    graph, n, c = make_graph()
    model = PlantAG(input_dim=100, num_classes=c)
    preds = model.predict(graph)
    assert len(preds) == n
    assert all(0 <= p < c for p in preds)
