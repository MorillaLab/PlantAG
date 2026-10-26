# PlantAG method and defaults

All defaults reproduce the article. Function names refer to the `plantag` package.

| Step | Function | Default |
|---|---|---|
| Spots without counts removed | `run_plantag` | — |
| Optional depth matching | `thin_to_median(adata, target)` | off (`thin_to_median_umis=None`) |
| Library-size normalisation, log1p | `preprocessing.normalize` | `target_sum=None` |
| Deviant genes | `preprocessing.select_deviant_genes` | analytic Pearson residuals, θ = 100, 4,000 genes by summed squared residuals |
| Expression PCA | `preprocessing.pca` | 50 components, ARPACK, deviant genes |
| Local persistence | `topology.compute_local_diagrams` | the spot and its 29 nearest neighbours (30 points) in `X_pca`; Rips complex up to dimension 2, truncated at the farthest neighbour (GUDHI) |
| Infinite deaths | `topology.clean_diagrams` | birth + largest finite death; H₀ and H₁ pooled |
| Persistence images | `topology.diagrams_to_images` | persim, pixel size 0.1, each image scaled to a maximum of 1 |
| Scalar descriptors | `topology.compute_final_tda_features_filtered_only` | cubical persistence of each image at 5 quantile superlevel thresholds, 8 statistics each (40), standardised |
| Marker scores | `labels.marker_scores` | mean z-score of the detected markers of each list |
| Training labels | `labels.training_labels` | 50 PCs of the standardised images; 30-neighbour graph; Leiden resolution 1.5; cluster argmax |
| Graph | `_network.build_weighted_graph` | standardised [50 expression PCs, image pixels, 40 descriptors]; 30 nearest neighbours, cosine; edge weight = similarity |
| Network | `_network.PlantCellGNN` | 3 GCNConv layers, 256 units, ReLU, dropout 0.5; linear classifier |
| Training | `model.fit_predict` | stratified 80/20 split (seed 42); class-weighted cross-entropy; Adam, learning rate 0.005, weight decay 1e-4; 100 epochs |
| Embedding | `_network.get_gnn_embeddings` | output of the third graph-convolution layer (256 dimensions) |

The functions in `topology.py` and `_network.py` are copied verbatim from the notebook used for the article; the only
change is an optional `random_state` for the PCA inside `build_weighted_graph`, so that seeds can be fixed.
