"""Literature-curated marker lists used to name PlantAG cell types.

`tomato_leaf_markers_static.json` holds the dictionary of the original notebook (cell 80) without the two lists that
were read from JSON-lines files; `load_markers(json_dir)` adds them exactly as the notebook did (cells 74-77):
`Trichomes = trichome.jsonl + fixed extras` and `Mesophyll-Ch = chlorop_random.jsonl + fixed extras`.
The lists are kept verbatim, including three concatenated gene names (missing commas), so results reproduce exactly.
"""
import json
from importlib import resources
from pathlib import Path


def load_markers(json_dir=None):
    """Marker dictionary {cell type: [genes]}; pass the folder with trichome.jsonl and chlorop_random.jsonl."""
    static = json.loads(resources.files("plantag").joinpath("data/tomato_leaf_markers_static.json").read_text())
    markers = {k: list(v) for k, v in static.items()}
    if json_dir is not None:
        d = Path(json_dir)
        with open(d / "trichome.jsonl") as fh:
            trichomes = [json.loads(line)["symbol"] for line in fh if line.strip()]
        with open(d / "chlorop_random.jsonl") as fh:
            chloroplast = [json.loads(line) for line in fh if line.strip()]
        markers["Trichomes"] = trichomes + markers["Trichomes"]
        markers["Mesophyll-Ch"] = chloroplast + markers["Mesophyll-Ch"]
    return markers
