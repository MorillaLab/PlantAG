"""Tests for PlantAG GNN architecture."""
import torch, pytest
from torch_geometric.data import Data


@pytest.fixture
def small_graph():
    n, f, c = 20, 100, 5   # small dims for unit test speed
    x = torch.randn(n, f)
    src = torch.randint(0, n, (60,))
    dst = torch.randint(0, n, (60,))
    return Data(x=x, edge_index=torch.stack([src, dst]),
                edge_attr=torch.rand(60)), n, c


def test_import():
    import planttag
    assert planttag.__version__ == "1.0.0"


def test_markers_loaded():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS
    assert len(TOMATO_LEAF_MARKERS) == 10
    assert "Parenchyma" in TOMATO_LEAF_MARKERS
