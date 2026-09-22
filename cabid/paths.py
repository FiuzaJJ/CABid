from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
FIGURES = ROOT / "figures"
RAW = ROOT / "raw"  # Foldseek benchmark tables, not tracked (see README)

SCOP_LOOKUP = DATA / "scop_lookup.fix.tsv"
DOMAIN_LENGTHS = DATA / "domain_lengths.npy"

POOLED = CACHE / "pooled"
BOOTSTRAP = CACHE / "bootstrap"
OVERLAP = CACHE / "overlap"
CV = CACHE / "cv"
SELF = CACHE / "self"
HIT_COLUMNS = CACHE / "hit_columns"  # ~800 MB, rebuilt by scripts/extract_hits.py
