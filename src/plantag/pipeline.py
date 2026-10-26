"""PlantAG end to end: AnnData of raw counts in, cell-type labels and a latent embedding out."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import labels, preprocessing, topology


def thin_to_median(adata, target, sample_key="sample", seed=0):
    """Binomial thinning of each sample (capture area) to a median of `target` UMIs per spot (never upwards)."""
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(adata.X).tocoo()
    sec = adata.obs[sample_key].astype(str).to_numpy()
    med = pd.Series(np.asarray(adata.X.sum(axis=1)).ravel()).groupby(sec).median()
    fac = np.minimum(1.0, target / med.reindex(sec).to_numpy())
    adata.X = sp.csr_matrix((rng.binomial(np.rint(X.data).astype(np.int64), fac[X.row]).astype(np.float32), (X.row, X.col)),
                            shape=adata.shape)
    adata.X.eliminate_zeros()
    return adata[np.asarray(adata.X.sum(axis=1)).ravel() > 0].copy()


def compute_topology(adata, use_rep="X_pca", n_levels=5):
    """Local persistence diagrams on `use_rep`, persistence images (obsm['images_vec1']) and the scalar descriptors
    (obsm['tda_features_final']), as in the original notebook (cells 49, 51, 60)."""
    diagrams = topology.compute_local_diagrams(adata.obsm[use_rep])
    diagrams_clean = topology.clean_diagrams(diagrams)
    images = topology.diagrams_to_images(diagrams_clean)
    adata = topology.compute_final_tda_features_filtered_only(adata, images, n_levels=n_levels)
    adata.obsm["images_vec1"] = np.array([img.ravel() for img in images])
    return adata, images, topology.diagrams_birth_to_death(diagrams_clean)


def run_plantag(adata, marker_genes, *, n_top_genes=4000, theta=100, leiden_resolution=1.5, n_neighbors=30,
                epochs=100, seed=0, device="cpu", thin_to_median_umis=None, sample_key="sample", verbose=True):
    """Run PlantAG on an AnnData whose X holds raw counts. Adds obs['tda_leiden'], obs['pseudo_cell_type'],
    obs['predicted_cell_type'], obsm['X_pca'], obsm['images_vec1'], obsm['tda_features_final'], obsm['X_tda_pca'] and
    obsm['X_gnn_embed']. Returns the annotated AnnData and a dict with the trained model, the training history and the
    accuracy on held-out spots."""
    from .model import fit_predict
    np.random.seed(seed)
    if thin_to_median_umis:
        adata = thin_to_median(adata, thin_to_median_umis, sample_key, seed)
    adata = adata[np.asarray(adata.X.sum(axis=1)).ravel() > 0, :].copy()
    adata.obs["total_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
    preprocessing.normalize(adata)
    preprocessing.select_deviant_genes(adata, n_top=n_top_genes, theta=theta)
    preprocessing.pca(adata)
    adata, _, _ = compute_topology(adata)
    adata.var_names_make_unique()
    labels.training_labels(adata, marker_genes, n_neighbors=n_neighbors, resolution=leiden_resolution, random_state=seed)
    result = fit_predict(adata, epochs=epochs, seed=seed, device=device, verbose=verbose)
    return adata, result
