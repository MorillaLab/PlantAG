"""
Post-GNN cell type merging — collapses 14 raw GNN predictions to 10 reported types.

Biological rationale for each merge:

  Guard     → Stomatal
    Guard cells are the inner cells of the stomatal complex; in Visium at 35 µm
    resolution they are not spatially resolved from the surrounding stomatal
    lineage cells. Transcriptionally, Guard markers (KAT1, OST1, SLAH3) overlap
    heavily with the Stomatal cluster at this resolution.

  Trichomes → Epidermal
    Trichomes are epidermal appendages. At Visium spot resolution, trichome-rich
    spots are indistinguishable from general epidermal spots; the Trichomes
    pseudo-label was used to improve Epidermal annotation recall but is
    reported as Epidermal in the final atlas.

  Vascular bundle → Vein
    Vascular bundle markers (LHW, WOX4) resolve vascular identity broadly;
    at Visium resolution Vein captures the full vascular bundle signal.
    Xylem and Phloem are distinguished separately; the residual Vascular bundle
    cluster maps to the Vein compartment in the spatial tissue.

  Mesophyll-Ch → Mesophyll
    Mesophyll-Ch was introduced to improve recovery of chloroplast-dense
    mesophyll cells using plastid-encoded marker genes. The biological
    identity is Mesophyll; the Ch designation reflects a marker strategy,
    not a distinct cell type.
"""

# Merge map: raw GNN label → reported atlas label
MERGE_MAP = {
    # 10 types that are kept as-is
    'Parenchyma':    'Parenchyma',
    'Vein':          'Vein',
    'Xylem':         'Xylem',
    'Phloem':        'Phloem',
    'Mesophyll':     'Mesophyll',
    'Epidermal':     'Epidermal',
    'Bundle sheath': 'Bundle sheath',
    'Stomatal':      'Stomatal',
    'Subsidiary':    'Subsidiary',
    'Hydathodes':    'Hydathodes',
    # 4 types collapsed
    'Guard':           'Stomatal',
    'Trichomes':       'Epidermal',
    'Vascular bundle': 'Vein',
    'Mesophyll-Ch':    'Mesophyll',
}

FINAL_CELL_TYPES = [
    'Parenchyma', 'Vein', 'Xylem', 'Phloem', 'Mesophyll',
    'Epidermal', 'Bundle sheath', 'Stomatal', 'Subsidiary', 'Hydathodes',
]

def merge_cell_types(adata,
                     raw_key='planttag_celltype_raw',
                     merged_key='planttag_celltype',
                     merge_map=None):
    """
    Collapse 14 raw GNN-predicted cell types to the 10 reported atlas types.

    This step is applied after PlantAG prediction and before all downstream
    biological analyses (cosine distance, pseudotime, RNA velocity, DE).

    Args:
        adata:      AnnData with raw predictions in obs[raw_key]
        raw_key:    obs column containing 14-type raw GNN predictions
        merged_key: obs column to write 10-type merged labels into
        merge_map:  dict mapping raw → merged (defaults to MERGE_MAP)

    Returns:
        adata with adata.obs[merged_key] populated
    """
    import pandas as pd

    if merge_map is None:
        merge_map = MERGE_MAP

    raw = adata.obs[raw_key]
    merged = raw.map(merge_map)

    unmapped = merged[merged.isna()].index.tolist()
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} cells have raw labels not in merge_map: "
            f"{set(raw[merged.isna()].values)}"
        )

    adata.obs[merged_key] = pd.Categorical(
        merged, categories=FINAL_CELL_TYPES
    )

    print(f"Merged 14 → 10 cell types (stored in adata.obs['{merged_key}']):")
    print(adata.obs[merged_key].value_counts().to_string())
    return adata
