#!/usr/bin/env python3
"""
End-to-end pipeline for a roadrover recording session.

Steps:
  1. Download OSM map once for the full session (make_map.py).
  2. Split the bag into fixed-duration chunks (default 30 s).
  3. Skip chunks where the vehicle never exceeded MIN_SPEED_MS.
  4. For each moving chunk:
       a. process_bag.py  — rotate 180° + lane detection + YOLO + map matching
       b. make_scenario.py — extract ego + actor trajectories → scenario.xosc

Output layout (alongside the original session directory):
  session_<ts>_map/         map_graph.pkl, map.geojson, lanes.geojson (downloaded once)
  session_<ts>_chunks/
    chunk_000/              raw 30-s bag (kept)
    chunk_000_processed/    processed bag
    chunk_000_scenario/     scenario.xosc + map.xodr
    chunk_001/  ...

Usage:
  python3 pipeline_session.py <bag_path>
  python3 pipeline_session.py <bag_path> --chunk-duration 30 --yolo-weights yolov8s.pt
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

VEL_TOPIC     = '/vel'
MIN_SPEED_MS  = 0.5    # m/s — chunks below this peak speed are skipped
SCRIPTS_DIR   = Path(__file__).parent


# ── Bag helpers ───────────────────────────────────────────────────────────────

def session_max_speed(bag_path: str) -> float:
    """Return max speed (m/s) seen across all /vel messages in the bag."""
    reader   = open_reader(bag_path)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if VEL_TOPIC not in type_map:
        return 0.0
    vel_type  = get_message(type_map[VEL_TOPIC])
    max_speed = 0.0
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != VEL_TOPIC:
            continue
        msg   = deserialize_message(data, vel_type)
        speed = math.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
        if speed > max_speed:
            max_speed = speed
    return max_speed


def open_reader(bag_path: str) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


# ── Split + velocity check in one pass ───────────────────────────────────────

def split_bag(bag_path: str, out_dir: Path,
              chunk_duration_s: int) -> list[dict]:
    """
    Split bag into chunks of chunk_duration_s seconds.
    Returns list of dicts: {path, duration_s, max_speed_ms}
    """
    chunk_ns = chunk_duration_s * 1_000_000_000

    reader      = open_reader(bag_path)
    topic_types = reader.get_all_topics_and_types()
    type_map    = {t.name: t.type for t in topic_types}
    vel_type    = get_message(type_map[VEL_TOPIC]) if VEL_TOPIC in type_map else None

    chunks: list[dict] = []
    writer              = None
    chunk_start         = None
    chunk_first_ts      = None
    chunk_last_ts       = None
    chunk_max_spd       = 0.0
    chunk_idx           = 0

    def _close():
        nonlocal writer
        if writer is not None:
            del writer
            writer = None
        if chunks:
            dur = (chunk_last_ts - chunk_first_ts) / 1e9 if chunk_first_ts and chunk_last_ts else 0.0
            chunks[-1]['duration_s']    = dur
            chunks[-1]['max_speed_ms']  = chunk_max_spd

    def _open(ts: int):
        nonlocal writer, chunk_idx, chunk_start, chunk_first_ts, chunk_last_ts, chunk_max_spd
        cdir = out_dir / f'chunk_{chunk_idx:03d}'
        if cdir.exists():
            shutil.rmtree(cdir)
        w = rosbag2_py.SequentialWriter()
        w.open(
            rosbag2_py.StorageOptions(uri=str(cdir), storage_id='sqlite3'),
            rosbag2_py.ConverterOptions('', ''),
        )
        for t in topic_types:
            w.create_topic(rosbag2_py.TopicMetadata(
                name=t.name, type=t.type, serialization_format='cdr',
            ))
        chunks.append({'path': cdir, 'duration_s': 0.0, 'max_speed_ms': 0.0})
        writer          = w
        chunk_idx      += 1
        chunk_start     = ts
        chunk_first_ts  = ts
        chunk_last_ts   = ts
        chunk_max_spd   = 0.0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if chunk_start is None:
            _open(timestamp)
        elif timestamp - chunk_start >= chunk_ns:
            _close()
            _open(timestamp)

        writer.write(topic, data, timestamp)
        chunk_last_ts = timestamp

        if vel_type and topic == VEL_TOPIC:
            msg   = deserialize_message(data, vel_type)
            speed = math.sqrt(msg.twist.linear.x ** 2 + msg.twist.linear.y ** 2)
            if speed > chunk_max_spd:
                chunk_max_spd = speed

    _close()
    return chunks


# ── Subprocess helper ─────────────────────────────────────────────────────────

def run(cmd: list, label: str) -> None:
    print(f'\n[{label}]', ' '.join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'[{label}] FAILED (exit {result.returncode})')
        sys.exit(result.returncode)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='End-to-end roadrover session pipeline')
    ap.add_argument('bag_path', help='Path to the raw rosbag2 session directory')
    ap.add_argument('--chunk-duration', type=int, default=30,
                    help='Chunk length in seconds (default: 30)')
    ap.add_argument('--yolo-weights', default='yolov8s.pt',
                    help='YOLOv8 weights file (default: yolov8s.pt)')
    ap.add_argument('--yes', action='store_true',
                    help='Skip confirmation prompt')
    ap.add_argument('--split-only', action='store_true',
                    help='Download map and split into chunks without running perception')
    args = ap.parse_args()

    session_dir = Path(args.bag_path).resolve()
    map_dir     = session_dir.parent / (session_dir.name + '_map')
    chunks_dir  = session_dir.parent / (session_dir.name + '_chunks')
    map_graph   = map_dir / 'map_graph.pkl'

    print(f'Session : {session_dir}')
    print(f'Map dir : {map_dir}')
    print(f'Chunks  : {chunks_dir}')

    # ── Pre-check: session-level velocity ────────────────────────────────────
    print('\nChecking session velocity ...')
    peak = session_max_speed(str(session_dir))
    print(f'  Peak speed: {peak:.2f} m/s')
    if peak < MIN_SPEED_MS:
        print('  Vehicle never moved in this session — nothing to process.')
        sys.exit(0)

    # ── Step 1: Download map ─────────────────────────────────────────────────
    if map_graph.exists():
        print(f'\nMap already exists at {map_graph} — skipping download.')
    else:
        print('\nStep 1: Downloading OSM map ...')
        map_dir.mkdir(parents=True, exist_ok=True)
        run([sys.executable, str(SCRIPTS_DIR / 'make_map.py'),
             str(session_dir), '--out-dir', str(map_dir)],
            label='make_map')

    # ── Step 2: Split bag ────────────────────────────────────────────────────
    print(f'\nStep 2: Splitting into {args.chunk_duration}s chunks ...')
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_bag(str(session_dir), chunks_dir, args.chunk_duration)
    print(f'  Created {len(chunks)} chunks.')

    # ── Dry-run preview ──────────────────────────────────────────────────────
    sep = '─' * 62
    print(f'\n{sep}')
    print(f'  {"Chunk":<12} {"Duration":>10} {"Peak speed":>12}  Action')
    print(sep)
    for c in chunks:
        skip   = c['max_speed_ms'] < MIN_SPEED_MS
        action = 'SKIP (stationary)' if skip else 'process'
        c['skip'] = skip
        print(f'  {c["path"].name:<12} {c["duration_s"]:>9.1f}s '
              f'{c["max_speed_ms"]:>10.2f} m/s  {action}')
    print(sep)

    n_process = sum(1 for c in chunks if not c['skip'])
    n_skip    = len(chunks) - n_process
    print(f'  {n_process} chunk(s) will be processed, {n_skip} skipped.\n')

    if args.split_only:
        print('Split-only mode — done. Use per-chunk processing to run perception.')
        sys.exit(0)

    if not args.yes:
        ans = input('Proceed? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            sys.exit(0)

    # ── Step 3: Process each moving chunk ────────────────────────────────────
    for i, c in enumerate(chunks):
        if c['skip']:
            print(f'\nSkipping {c["path"].name} (stationary).')
            continue

        cp = c['path']
        print(f'\n{"=" * 62}')
        print(f'  {cp.name}  ({i + 1}/{len(chunks)})')
        print(f'{"=" * 62}')

        processed    = cp.parent / (cp.name + '_processed')
        scenario_dir = cp.parent / (cp.name + '_scenario')
        scenario_dir.mkdir(exist_ok=True)

        run([sys.executable, str(SCRIPTS_DIR / 'process_bag.py'),
             str(cp),
             '--output', str(processed),
             '--map-graph', str(map_graph)],
            label=f'{cp.name}/process_bag')

        run([sys.executable, str(SCRIPTS_DIR / 'make_scenario.py'),
             str(processed),
             '--map-graph', str(map_graph),
             '--yolo-weights', args.yolo_weights,
             '--out-dir', str(scenario_dir)],
            label=f'{cp.name}/make_scenario')

    print('\nAll done.')
    print(f'Results under: {chunks_dir}')


if __name__ == '__main__':
    main()
