"""PlantAG in a few lines: raw-count AnnData in, imputed cell types and a latent embedding out."""
import scanpy as sc

import plantag

adata = sc.read_h5ad("my_visium_raw_counts.h5ad")         # X: raw counts; obs['sample']: capture area of each spot
markers = plantag.load_markers("json_data")                # tomato leaf lists; or any {cell type: [genes]} dictionary
adata, result = plantag.run_plantag(adata, markers, seed=0, epochs=100, device="cpu")
print(adata.obs["predicted_cell_type"].value_counts())
print(f"accuracy on held-out spots: {result['held_out_accuracy']:.3f}")
adata.write_h5ad("my_visium_plantag.h5ad")
