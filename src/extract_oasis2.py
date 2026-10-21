"""Extract only spreadsheet-matched OASIS-2 RAW NIfTI scans, locally."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from src.oasis_ai.pipeline import read_table


SESSION_ID = re.compile(r"^OAS2_\d{4}_MR\d+$")
NIFTI_SUFFIXES = (".nifti.hdr", ".nifti.img")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract spreadsheet-matched OASIS-2 RAW scans locally; outputs are ignored by Git."
    )
    parser.add_argument("--archive", required=True, help="Path to OAS2_RAW_PART1.tar.gz or PART2.tar.gz")
    parser.add_argument("--table", required=True, help="OASIS demographics .xlsx or .csv containing MRI ID")
    parser.add_argument("--output", default="data/raw/oasis2_mri", help="Local extraction directory")
    args = parser.parse_args()

    table = read_table(args.table)
    if "MRI ID" not in table:
        raise SystemExit("The demographics table must contain an 'MRI ID' column.")
    wanted = set(table["MRI ID"].dropna().astype(str).str.strip())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    found_sessions: set[str] = set()
    extracted: list[str] = []
    extracted_bytes = 0
    archive_path = Path(args.archive)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            session = next((part for part in parts if SESSION_ID.fullmatch(part)), None)
            if session not in wanted or "RAW" not in parts:
                continue
            name = PurePosixPath(member.name).name
            if not name.endswith(NIFTI_SUFFIXES):
                continue

            destination = output / session / "RAW" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            found_sessions.add(session)
            extracted.append(f"{session}/RAW/{name}")
            extracted_bytes += member.size

    if not extracted:
        raise SystemExit("No matching RAW NIfTI files were found in this archive.")

    pairs: dict[str, dict[str, set[str]]] = {}
    for relative in extracted:
        session, _, filename = relative.split("/", 2)
        suffix = next(suffix for suffix in NIFTI_SUFFIXES if filename.endswith(suffix))
        scan_stem = filename[: -len(suffix)]
        extension = suffix.rsplit(".", 1)[-1]
        pairs.setdefault(session, {}).setdefault(scan_stem, set()).add(extension)
    incomplete = sorted(
        f"{session}/{scan}"
        for session, scans in pairs.items()
        for scan, extensions in scans.items()
        if extensions != {"hdr", "img"}
    )
    manifest = {
        "source_archive": archive_path.name,
        "source_table": Path(args.table).name,
        "matched_sessions": len(found_sessions),
        "nifti_files": len(extracted),
        "extracted_bytes": extracted_bytes,
        "scans_missing_hdr_or_img_pair": incomplete,
        "files": sorted(extracted),
    }
    (output / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    print(f"Extracted files locally to {output}")
    print("These participant-level scans must remain access-controlled and are excluded from Git.")


if __name__ == "__main__":
    main()
