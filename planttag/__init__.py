"""
PlantAG — Plant Tissue Annotation via Graph Topology
=====================================================
Automated cell-type annotation for plant spatial transcriptomics.
No reference atlas required.

    from planttag import PlantAG, run_pipeline
    adata = run_pipeline(adata, marker_genes=my_markers)

GitHub:  https://github.com/MorillaLab/PlantAG
Contact: ian.morilla@ihsm.uma-csic.es
"""

from planttag.model.gcn        import PlantAG
from planttag.tda.features     import (compute_local_diagrams, clean_diagrams,
                                        diagrams_to_images, diagrams_birth_to_death,
                                        compute_final_tda_features_filtered_only,
                                        persistence_entropy, compute_wasserstein_matrix)
from planttag.graph.builder    import build_weighted_graph
from planttag.labels.scoring   import (generate_pseudo_labels, score_cell_types,
                                        compute_class_weights, build_marker_dict,
                                        TOMATO_LEAF_MARKERS_A1)
from planttag.train            import run_training, make_train_test_functions
from planttag.preprocessing    import (pearson_residuals, apply_scran_normalisation,
                                        align_velocity_layers)

__version__ = "1.0.0"

__all__ = [
    "PlantAG",
    "run_training", "make_train_test_functions",
    "compute_local_diagrams", "clean_diagrams",
    "diagrams_to_images", "diagrams_birth_to_death",
    "compute_final_tda_features_filtered_only",
    "persistence_entropy", "compute_wasserstein_matrix",
    "build_weighted_graph",
    "generate_pseudo_labels", "score_cell_types",
    "compute_class_weights", "build_marker_dict",
    "TOMATO_LEAF_MARKERS_A1",
    "pearson_residuals", "apply_scran_normalisation",
    "align_velocity_layers",
]
