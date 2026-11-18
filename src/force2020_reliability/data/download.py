from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from force2020_reliability.paths import RAW_DIR

ZENODO_RECORD_API = "https://zenodo.org/api/records/4351156"
EXPECTED_FILE = "LAS_files_Force_2020_all_wells_train_test_blind_hidden_final.zip"
EXTRA_FILES = (
    "NPD_Casing_depth_most_wells.xlsx",
    "lithology scoring matrix cost function.xlsx",
    "NPD_Lithostratigraphy_member_formations_all_wells.xlsx",
    "NPD_Lithostratigraphy_groups_all_wells.xlsx",
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_official_data(raw_dir: Path = RAW_DIR) -> dict[str, object]:
    """Download official Zenodo files and verify their published MD5 checksums."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD_API) as response:
        metadata = json.load(response)

    by_name = {item["key"]: item for item in metadata["files"]}
    selected = (EXPECTED_FILE, *EXTRA_FILES)
    verification: dict[str, object] = {}
    for name in selected:
        if name not in by_name:
            raise KeyError(f"Zenodo record does not contain expected file: {name}")
        item = by_name[name]
        destination = raw_dir / name
        if not destination.exists():
            urllib.request.urlretrieve(item["links"]["self"], destination)
        expected = item["checksum"].removeprefix("md5:").lower()
        actual = _md5(destination)
        verification[name] = {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "md5_expected": expected,
            "md5_actual": actual,
            "valid": actual == expected,
        }
        if actual != expected:
            raise ValueError(f"MD5 mismatch for {name}: {actual} != {expected}")

    record_path = raw_dir / "zenodo_record.json"
    record_path.write_text(
        json.dumps(
            {
                "doi": metadata.get("doi"),
                "record_id": metadata.get("id"),
                "title": metadata.get("title"),
                "files": verification,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return verification