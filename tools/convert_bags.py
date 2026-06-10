#!/usr/bin/env python3
"""
Convert rosbag2 sqlite3 bags to MCAP format so Lichtblick can open them via HTTP URL.

Usage:
    # Convert only processed chunks (default):
    python3 convert_bags.py <chunks_dir>

    # Convert only raw (unprocessed) chunks:
    python3 convert_bags.py <chunks_dir> --raw

    # Convert both raw and processed:
    python3 convert_bags.py <chunks_dir> --both

    # Convert a single chunk by name (raw or processed):
    python3 convert_bags.py <chunks_dir> --chunk chunk_000
    python3 convert_bags.py <chunks_dir> --chunk chunk_000_processed

Output layout (siblings inside chunks_dir):
    chunk_000_processed/      ← original sqlite3 bag
    chunk_000_processed_mcap/ ← new MCAP bag (Lichtblick-ready)
    chunk_000_mcap/           ← raw MCAP bag (only with --raw or --both)

Requires: ros-humble-rosbag2-storage-mcap
    sudo apt install ros-humble-rosbag2-storage-mcap
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _copy_topics(reader, writer) -> int:
    """Copy all topic metadata and messages. Returns message count."""
    import rosbag2_py  # noqa: local import so error is per-call, not at module load

    for topic in reader.get_all_topics_and_types():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic.name,
            type=topic.type,
            serialization_format="cdr",
        ))
    n = 0
    while reader.has_next():
        writer.write(*reader.read_next())
        n += 1
    return n


def convert_bag(src: Path, dst: Path, force: bool = False) -> bool:
    """
    Convert rosbag2 sqlite3 bag at src to MCAP at dst.
    Returns True on success. dst is a rosbag2-style directory containing a .mcap file.
    force: overwrite an existing MCAP (needed after reprocessing the source bag).
    """
    # Already done?
    if not force and dst.exists() and next(dst.glob("*.mcap"), None):
        print(f"  {src.name} → already converted, skipping (use --force to redo)")
        return True
    if dst.exists():
        shutil.rmtree(dst)

    print(f"  {src.name} → {dst.name} ...", end=" ", flush=True)
    try:
        import rosbag2_py

        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(src), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", ""),
        )

        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(dst), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )

        n = _copy_topics(reader, writer)
        del writer
        del reader

        mcap_file = next(dst.glob("*.mcap"), None)
        size_mb = mcap_file.stat().st_size / 1e6 if mcap_file else 0
        print(f"ok  {n} msgs  {size_mb:.1f} MB")
        return True

    except Exception as exc:
        print(f"FAILED: {exc}")
        if "mcap" in str(exc).lower() or "storage" in str(exc).lower():
            print("    → Install the MCAP plugin: sudo apt install ros-humble-rosbag2-storage-mcap")
        if dst.exists():
            shutil.rmtree(dst)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert db3 rosbag2 bags to MCAP")
    ap.add_argument("chunks_dir", help="Path to the session chunks directory")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true",
                      help="Convert only raw (unprocessed) chunks")
    mode.add_argument("--both", action="store_true",
                      help="Convert both raw and processed chunks")
    ap.add_argument("--chunk", metavar="CHUNK_NAME",
                    help="Convert a single named chunk (e.g. chunk_000 or chunk_000_processed)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing MCAP output (use after reprocessing)")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        print(f"Directory not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)

    ok = errors = skipped = 0

    # Single-chunk mode
    if args.chunk:
        cdir = chunks_dir / args.chunk
        if not cdir.exists():
            print(f"Chunk not found: {cdir}", file=sys.stderr)
            sys.exit(1)
        dst = chunks_dir / (args.chunk + "_mcap")
        sys.exit(0 if convert_bag(cdir, dst, force=args.force) else 1)

    for cdir in sorted(chunks_dir.iterdir()):
        if not cdir.is_dir() or not cdir.name.startswith("chunk_"):
            continue
        # Skip already-converted and scenario dirs
        if any(cdir.name.endswith(s) for s in ("_mcap", "_scenario")):
            continue
        # Check it's actually a rosbag2 bag
        if not (cdir / "metadata.yaml").exists() and not list(cdir.glob("*.db3")):
            continue

        is_processed = cdir.name.endswith("_processed")
        if args.raw and is_processed:
            skipped += 1
            continue
        if not args.raw and not args.both and not is_processed:
            skipped += 1
            continue

        dst = chunks_dir / (cdir.name + "_mcap")
        if convert_bag(cdir, dst, force=args.force):
            ok += 1
        else:
            errors += 1

    if skipped:
        hint = "pass --both to also convert them" if not args.raw else "pass --both to also convert processed"
        print(f"\n  (skipped {skipped} chunks — {hint})")
    print(f"\nConverted: {ok}  Errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
