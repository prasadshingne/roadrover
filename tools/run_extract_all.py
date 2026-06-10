#!/usr/bin/env python3
"""
Run make_scenario.py on every processed chunk in a chunks directory.

Used by studio_server.py for the "Extract scenario" job, but can also be
run directly:
    python3 run_extract_all.py <chunks_dir> --map-graph <path/to/map_graph.pkl>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "src" / "roadrover_perception" / "scripts"


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch scenario extraction for all chunks")
    ap.add_argument("chunks_dir", help="Path to the session chunks directory")
    ap.add_argument("--map-graph", required=True, help="Path to map_graph.pkl")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    map_graph  = Path(args.map_graph)

    if not chunks_dir.exists():
        print(f"Chunks directory not found: {chunks_dir}", file=sys.stderr)
        sys.exit(1)
    if not map_graph.exists():
        print(f"Map graph not found: {map_graph}", file=sys.stderr)
        sys.exit(1)

    processed_dirs = sorted(chunks_dir.glob("chunk_*_processed"))
    if not processed_dirs:
        print("No processed chunks found. Run pipeline_session.py first.")
        sys.exit(1)

    errors = 0
    for proc_dir in processed_dirs:
        chunk_name   = proc_dir.name.replace("_processed", "")
        scenario_dir = chunks_dir / (chunk_name + "_scenario")
        scenario_dir.mkdir(exist_ok=True)

        print(f"\n[{chunk_name}] extracting → {scenario_dir.name}")
        result = subprocess.run([
            sys.executable,
            str(SCRIPTS_DIR / "make_scenario.py"),
            str(proc_dir),
            "--map-graph", str(map_graph),
            "--out-dir",   str(scenario_dir),
        ])
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})")
            errors += 1

    print(f"\nAll done. Errors: {errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
