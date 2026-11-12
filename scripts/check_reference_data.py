"""Optional quick check for downloaded official corpus and evaluation rows."""

from pathlib import Path

from crlab.datasets.anthropic import load_dataset

base = Path("data/raw/anthropic")
ds = load_dataset(base / "codebase_chunks.json", base / "evaluation_set.jsonl")
expected = (90, 737, 248)
actual = (ds.document_count, len(ds.chunks), len(ds.queries))
print(f"documents/chunks/queries: {actual}, expected reference: {expected}")
if actual != expected:
    raise SystemExit("Dataset counts differ from the historical reference")
