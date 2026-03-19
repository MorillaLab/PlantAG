"""
planttag/tda/features.py
TDA feature extraction for PlantAG.

Uses gudhi (RipsComplex) + persim (PersistenceImager) —
matching the actual notebook implementation (Cells 44-52).

Pipeline per cell:
  PCA coords → local k-NN → RipsComplex → persistence diagram
  → clean_diagrams → diagrams_to_images (persim)
  → features_from_image (CubicalComplex) → compute_final_tda_features_filtered_only

Luna*, Aberbache* et al. — MorillaLab
"""

import numpy as np
import gudhi as gd
from persim import PersistenceImager
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ── 1. Local persistence diagrams (Cell 44) ──────────────────────────────

def compute_local_diagrams(X, n_neighbors=30, max_dim=2):
    """
    Compute local persistence diagrams for each cell in X.

    For each cell, takes its k nearest neighbours in PCA space,
    builds a Vietoris-Rips complex, and computes H0+H1 persistence.

    Args:
        X:           ndarray (n_cells, n_pcs) — PCA coordinates
        n_neighbors: number of neighbours for local complex
        max_dim:     maximum homology dimension

    Returns:
        list of persistence diagrams (gudhi format: [(dim, (birth, death)), ...])
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors,
                             algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    diagrams = []
    for i in range(X.shape[0]):
        points = X[indices[i], :]
        rips = gd.RipsComplex(points=points,
                               max_edge_length=float(np.max(distances[i])))
        st = rips.create_simplex_tree(max_dimension=max_dim)
        st.compute_persistence()
        diagrams.append(st.persistence())

    return diagrams


# ── 2. Clean diagrams (Cell 48) ───────────────────────────────────────────

def clean_diagrams(diagrams):
    """
    Clean persistence diagrams: replace infinite death values and
    convert to (birth, lifetime) format for persim compatibility.

    Args:
        diagrams: list of gudhi persistence diagrams

    Returns:
        list of np.ndarray, each shape (n_points, 2) in (birth, lifetime)
    """
    cleaned = []
    for diag in diagrams:
        finite_vals = [d for _, (b, d) in diag if np.isfinite(d)]
        max_val = max(finite_vals) if finite_vals else 1.0
        points = []
        for _, (b, d) in diag:
            if not np.isfinite(d):
                d = b + max_val
            if d > b:
                points.append([b, d - b])   # (birth, lifetime)
        if points:
            cleaned.append(np.array(points, dtype=float))
        else:
            cleaned.append(np.array([[0.0, 1e-6]], dtype=float))
    return cleaned


# ── 3. Convert to (birth, death) format (Cell 47) ────────────────────────

def diagrams_birth_to_death(diagrams_clean):
    """
    Convert (birth, lifetime) diagrams to (birth, death) format
    required by persim.wasserstein / bottleneck.

    Args:
        diagrams_clean: list of np.ndarray in (birth, lifetime) format

    Returns:
        list of np.ndarray in (birth, death) format
    """
    diagrams_bd = []
    for diag in diagrams_clean:
        if len(diag) == 0:
            diagrams_bd.append(np.array([[0.0, 1e-6]]))
            continue
        births = diag[:, 0]
        deaths = diag[:, 0] + diag[:, 1]
        deaths = np.maximum(deaths, births + 1e-8)
        diagrams_bd.append(np.column_stack([births, deaths]))
    return diagrams_bd


# ── 4. Persistence images (Cell 46) ──────────────────────────────────────

def diagrams_to_images(diagrams_clean, pixel_size=0.1):
    """
    Vectorise persistence diagrams as 2D persistence images using persim.

    Args:
        diagrams_clean: list of np.ndarray (birth, lifetime) format
        pixel_size:     pixel resolution of the persistence image

    Returns:
        list of 2D np.ndarray (normalised to [0, 1])
    """
    pimgr = PersistenceImager(pixel_size=pixel_size)
    pimgr.fit(diagrams_clean)
    images = [pimgr.transform(d) for d in diagrams_clean]
    images = [img / (img.max() + 1e-8) for img in images]
    return images


# ── 5. Per-image features (Cell 48, COMPLETED) ───────────────────────────

def persistence_from_image(img):
    """
    Compute persistence diagram of a 2D image via gudhi CubicalComplex.
    """
    cc = gd.CubicalComplex(
        dimensions=img.shape,
        top_dimensional_cells=img.flatten()
    )
    diag = cc.persistence()
    return diag


def features_from_image(img):
    """
    Extract 8 scalar TDA features from the persistence diagram of a
    2D persistence image using CubicalComplex filtration.

    Features:
        0: number of finite persistence points
        1: total persistence (sum of lifetimes)
        2: max persistence
        3: mean persistence
        4: std of persistence
        5: entropy of persistence distribution
        6: number of points with persistence > 0.1 (prominent features)
        7: Betti-0 proxy (H0 points count)

    Args:
        img: 2D np.ndarray (persistence image)

    Returns:
        np.ndarray of shape (8,)
    """
    diag = persistence_from_image(img)
    points = np.array([pt for _, pt in diag if np.isfinite(pt[1])])

    if points.size == 0:
        return np.zeros(8, dtype=float)

    births, deaths = points[:, 0], points[:, 1]
    pers = deaths - births
    pers = pers[pers > 0]

    if len(pers) == 0:
        return np.zeros(8, dtype=float)

    total = pers.sum()
    probs = pers / (total + 1e-8)
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    # Betti-0 proxy: count H0 points (dim=0 entries)
    h0_count = sum(1 for (dim, _) in diag if dim == 0)

    return np.array([
        float(len(pers)),           # 0: n finite points
        float(total),               # 1: total persistence
        float(pers.max()),          # 2: max persistence
        float(pers.mean()),         # 3: mean persistence
        float(pers.std()),          # 4: std persistence
        entropy,                    # 5: persistence entropy
        float((pers > 0.1).sum()),  # 6: prominent features
        float(h0_count),            # 7: Betti-0 proxy
    ], dtype=float)


# ── 6. Superlevel filtration (Cell 48) ───────────────────────────────────

def superlevel_filtration(img, n_levels=5, mode='quantile'):
    """
    Create binary filtered images at adaptive or linear thresholds.

    Args:
        img:      2D np.ndarray
        n_levels: number of threshold levels
        mode:     'quantile' (adaptive) or 'linear'

    Returns:
        list of (threshold, binary_image) tuples
    """
    if mode == 'linear':
        thresholds = np.linspace(0, 1, n_levels)
    elif mode == 'quantile':
        thresholds = np.quantile(img, np.linspace(0, 1, n_levels))
    else:
        raise ValueError("mode must be 'linear' or 'quantile'")
    return [(t, (img >= t).astype(int)) for t in thresholds]


# ── 7. Multi-level TDA features (Cell 50, COMPLETED) ─────────────────────

def compute_final_tda_features_filtered_only(adata, images, n_levels=5):
    """
    MISSING FUNCTION — called in notebook Cell 50 but never defined.

    Computes multi-level TDA features by combining:
      (a) 8 scalar features from the full persistence image (features_from_image)
      (b) 8 scalar features per superlevel-filtered image × n_levels

    Total per cell: 8 + (8 × n_levels) = 8 + 40 = 48 features (default n_levels=5)

    Stores result in adata.obsm['tda_features_final'].

    Args:
        adata:    AnnData object
        images:   list of 2D np.ndarray persistence images (one per cell)
        n_levels: number of superlevel filtration thresholds

    Returns:
        adata with adata.obsm['tda_features_final'] populated
    """
    all_features = []

    for img in images:
        # Base features from full image
        base_feats = features_from_image(img)           # shape (8,)

        # Multi-level superlevel features
        level_feats = []
        for _, binary_img in superlevel_filtration(img, n_levels=n_levels,
                                                    mode='quantile'):
            level_feats.append(features_from_image(binary_img.astype(float)))

        # Concatenate base + all level features
        cell_feats = np.concatenate([base_feats] +
                                     [f for f in level_feats])  # (8 + 8*n_levels,)
        all_features.append(cell_feats)

    tda_matrix = np.array(all_features, dtype=float)  # (n_cells, 8 + 8*n_levels)

    # Replace any NaN/inf that slipped through
    tda_matrix = np.nan_to_num(tda_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    adata.obsm['tda_features_final'] = tda_matrix

    print(f"TDA features computed: {tda_matrix.shape} "
          f"({tda_matrix.shape[0]} cells × {tda_matrix.shape[1]} features)")
    return adata


# ── 8. Persistence entropy (Cell 58) ─────────────────────────────────────

def persistence_entropy(diagram):
    """
    Compute normalised persistence entropy for a single diagram.

    Args:
        diagram: np.ndarray (N, 2) in (birth, lifetime) format

    Returns:
        float — entropy value
    """
    if diagram is None or len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1]
    total = np.sum(lifetimes)
    if total == 0:
        return 0.0
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


# ── 9. Wasserstein distance matrix (Cell 45) ─────────────────────────────

def compute_wasserstein_matrix(diagrams_bd):
    """
    Compute pairwise Wasserstein distance matrix between persistence diagrams.

    Args:
        diagrams_bd: list of np.ndarray in (birth, death) format

    Returns:
        np.ndarray (n_cells, n_cells) symmetric distance matrix
    """
    from persim import wasserstein

    n = len(diagrams_bd)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dist = wasserstein(diagrams_bd[i], diagrams_bd[j], matching=False)
            if dist < 0 or not np.isfinite(dist):
                dist = wasserstein(diagrams_bd[i], diagrams_bd[j], matching=True)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix
