"""
PlantAG — Plant Tissue Annotation via Graph Topology
=====================================================
Automated cell-type annotation for plant spatial transcriptomics.
No reference atlas required.

    from planttag import PlantAG, run_pipeline
    from planttag.labels.markers import TOMATO_LEAF_MARKERS

GitHub:  https://github.com/MorillaLab/PlantAG
Contact: ian.morilla@ihsm.uma-csic.es
"""

from planttag.model.gcn        import PlantAG
from planttag.tda.features     import (
    compute_local_diagrams, clean_diagrams,
    diagrams_to_images, diagrams_birth_to_death,
    compute_final_tda_features_filtered_only,
    persistence_entropy, compute_wasserstein_matrix,
)
from planttag.graph.builder    import build_weighted_graph
from planttag.labels.scoring   import (
    generate_pseudo_labels, score_cell_types,
    compute_class_weights, load_external_markers,
)
from planttag.labels.merge     import (
    merge_cell_types, MERGE_MAP, FINAL_CELL_TYPES,
)
from planttag.labels.markers   import (
    TOMATO_LEAF_MARKERS, TOMATO_LEAF_MARKERS_10,
    CELL_TYPE_COLORS, CELL_TYPE_ORDER_10,
    DISCARDED_TYPES,
    MT_GENE_LIST, RIBO_GENE_LIST,
    build_full_marker_dict,
)
from planttag.train            import run_training, make_train_test_functions
from planttag.preprocessing    import (
    pearson_residuals, apply_scran_normalisation, align_velocity_layers,
)

__version__ = "1.0.0"
