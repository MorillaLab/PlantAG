"""Tests for marker gene dictionary — validates all bug fixes."""
import pytest


def test_all_cell_types_present():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS
    expected = {
        'Epidermal', 'Mesophyll', 'Guard', 'Trichomes',
        'Vascular bundle', 'Xylem', 'Phloem', 'Stomatal',
        'Parenchyma', 'Bundle sheath', 'Subsidiary',
        'Hydathodes', 'Vein', 'Mesophyll-Ch',
    }
    assert set(TOMATO_LEAF_MARKERS.keys()) == expected


def test_no_concatenated_strings():
    """All three missing-comma bugs must be fixed."""
    from planttag.labels.markers import TOMATO_LEAF_MARKERS
    all_genes = [g for genes in TOMATO_LEAF_MARKERS.values() for g in genes]
    assert 'LOC101264905WER'    not in all_genes, "Trichomes comma bug not fixed"
    assert 'LOC101246250MUTE'   not in all_genes, "Stomatal comma bug not fixed"
    assert 'LOC101257875RBCS3B' not in all_genes, "Bundle sheath comma bug not fixed"


def test_trichomes_count():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS, TRICHOMES
    # 63 from jsonl + 8 inline
    assert len(TOMATO_LEAF_MARKERS['Trichomes']) == len(TRICHOMES) + 8


def test_mesophyll_ch_count():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS, CHLOROPLAST_SAMPLE_448
    # 100 sampled + 18 inline
    assert len(TOMATO_LEAF_MARKERS['Mesophyll-Ch']) == len(CHLOROPLAST_SAMPLE_448) + 18


def test_mt_ribo_lists_populated():
    from planttag.labels.markers import MT_GENE_LIST, RIBO_GENE_LIST
    assert len(MT_GENE_LIST) == 665
    assert len(RIBO_GENE_LIST) == 408


def test_colors_cover_all_types():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS, CELL_TYPE_COLORS
    for ct in TOMATO_LEAF_MARKERS:
        assert ct in CELL_TYPE_COLORS, f"No color for {ct}"


def test_no_duplicate_genes_within_type():
    from planttag.labels.markers import TOMATO_LEAF_MARKERS
    for ct, genes in TOMATO_LEAF_MARKERS.items():
        dups = [g for g in genes if genes.count(g) > 1]
        assert not dups, f"Duplicate genes in {ct}: {set(dups)}"
