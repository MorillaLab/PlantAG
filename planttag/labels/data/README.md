# Marker gene data files

| File | Genes | Description |
|------|------:|-------------|
| `trichome.jsonl` | 63 | Trichome-specific markers |
| `chlorop.jsonl` | 1,295 | Chloroplast-encoded genes (100 sampled with seed=448) |
| `mt.jsonl` | 665 | Mitochondrial genes (used for QC filtering) |
| `ribo.jsonl` | 408 | Ribosomal genes (used for QC filtering) |

All files sourced from Sol Genomics Network / ITAG4.0 annotation,
formatted as JSON Lines with a `symbol` field per entry.

Usage:
```python
from planttag.labels.markers import load_external_markers
ext = load_external_markers(
    trichome_path='planttag/labels/data/trichome.jsonl',
    chlorop_path='planttag/labels/data/chlorop.jsonl',
    mt_path='planttag/labels/data/mt.jsonl',
    ribo_path='planttag/labels/data/ribo.jsonl',
)
```
