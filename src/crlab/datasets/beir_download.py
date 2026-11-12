"""Verified SciFact download, isolated extraction and provenance manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

OFFICIAL_SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/"
    "thakur/BEIR/datasets/scifact.zip"
)
# Official BEIR data wiki checksum (MD5; preserved solely for version matching).
OFFICIAL_SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
MAX_ZIP_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_BYTES = 150 * 1024 * 1024


def download_scifact(root: str | Path) -> dict:
    """Download only canonical SciFact; never auto-run a script from the archive.

    Fails closed if source checksum changes, target exists, or paths are unsafe.
    Does not overwrite previous downloads. Requires access to official UKP host.
    """
    root = Path(root)
    destination = root / "scifact"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite BEIR SciFact dataset at {destination}"
        )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crlab-scifact-", dir=root) as tmp:
        staging = Path(tmp)
        archive = staging / "scifact.zip"
        request = urllib.request.Request(
            OFFICIAL_SCIFACT_URL, headers={"User-Agent": "crlab/0.1"}
        )
        md5 = hashlib.md5()  # noqa: S324 - matching upstream published checksum
        sha256 = hashlib.sha256()
        consumed = 0
        with urllib.request.urlopen(request, timeout=120) as response:
            with archive.open("wb") as writer:
                while chunk := response.read(1024 * 1024):
                    consumed += len(chunk)
                    if consumed > MAX_ZIP_BYTES:
                        raise ValueError("SciFact archive exceeds maximum download size")
                    md5.update(chunk)
                    sha256.update(chunk)
                    writer.write(chunk)
        if md5.hexdigest() != OFFICIAL_SCIFACT_MD5:
            raise ValueError("SciFact archive checksum differs from official BEIR MD5")
        extracted = 0
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if "\\" in info.filename:
                    raise ValueError("Backslashes prohibited in archive paths")
                path = PurePosixPath(info.filename)
                if not path.parts or path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {info.filename}")
                if path.parts[0] != "scifact":
                    raise ValueError(f"Unexpected SciFact archive root: {info.filename}")
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Symlinks are prohibited in the archive")
                extracted += info.file_size
                if extracted > MAX_EXTRACTED_BYTES:
                    raise ValueError("SciFact archive exceeds extracted-size limit")
                output = staging.joinpath(*path.parts)
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        root_dataset = staging / "scifact"
        for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            if not (root_dataset / name).is_file():
                raise ValueError(f"SciFact archive missing required file {name}")
        manifest = {
            "dataset": "scifact",
            "source_url": OFFICIAL_SCIFACT_URL,
            "official_zip_md5": OFFICIAL_SCIFACT_MD5,
            "downloaded_zip_sha256": sha256.hexdigest(),
            "archive_bytes": consumed,
            "extracted_bytes": extracted,
            "label_granularity": "document",
        }
        (root_dataset / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        root_dataset.rename(destination)
        return manifest
