# Output fields

`run_plantag` returns `(adata, result)`.

**adata** gains:

- `obs["total_counts"]`: UMIs per spot.
- `layers["log1p_norm"]`, `layers["analytic_pearson_residuals"]`.
- `var["highly_deviant"]`, `var["deviance_score"]`, `var["highly_variable"]` (= highly_deviant).
- `obsm["X_pca"]`: 50 expression principal components.
- `obsm["images_vec1"]`: flattened persistence image of each spot (pixel count depends on the data).
- `obsm["tda_features_final"]`: 40 standardised topological descriptors.
- `obsm["X_tda_pca"]`: 50 principal components of the standardised persistence images.
- `obs["tda_leiden"]`, `obs["pseudo_cell_type"]`: Leiden clusters and training labels.
- `obs["predicted_cell_type"]`: PlantAG cell types.
- `obsm["X_gnn_embed"]`: latent embedding (256 dimensions).

**result** holds `model` (the trained network), `history` (loss and held-out accuracy per epoch) and
`held_out_accuracy` (accuracy on the 20% of spots withheld from training).
