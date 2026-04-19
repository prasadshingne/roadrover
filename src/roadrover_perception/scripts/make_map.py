#!/usr/bin/env python3
"""
Downloads the OSM road network for the area covered by a recorded bag.

Outputs (default: same directory as the bag):
  map.geojson   — road edges as GeoJSON; drag onto Foxglove Map panel
  map_graph.pkl — pickled osmnx graph used by process_bag.py for map matching

Usage:
  python3 make_map.py <bag_path> [--out-dir /path/to/output]
"""

import argparse
import pickle
from pathlib import Path

import osmnx as ox
from shapely.geometry import box as shapely_box

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

FIX_TOPIC  = '/fix'
MARGIN_DEG = 0.003   # ~300 m padding around the GPS bounding box


def extract_gps_bbox(bag_path: str):
    """Read /fix messages from the bag; return (min_lat, max_lat, min_lon, max_lon)."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''),
    )
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if FIX_TOPIC not in type_map:
        raise RuntimeError(f'{FIX_TOPIC} not found in bag — is GPS data recorded?')

    lats, lons = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != FIX_TOPIC:
            continue
        msg = deserialize_message(data, get_message(type_map[topic]))
        if msg.status.status < 0:   # STATUS_NO_FIX
            continue
        lats.append(msg.latitude)
        lons.append(msg.longitude)

    if not lats:
        raise RuntimeError('No valid GPS fixes found in bag.')

    print(f'  GPS fixes : {len(lats)}')
    print(f'  Lat range : {min(lats):.6f} → {max(lats):.6f}')
    print(f'  Lon range : {min(lons):.6f} → {max(lons):.6f}')
    return min(lats), max(lats), min(lons), max(lons)


def clean_edges(edges):
    """Drop non-serializable columns and flatten list-valued cells."""
    edges = edges.reset_index()
    keep = ['geometry', 'name', 'highway', 'lanes', 'oneway', 'maxspeed', 'length']
    edges = edges[[c for c in keep if c in edges.columns]].copy()
    for col in edges.columns:
        if col == 'geometry':
            continue
        edges[col] = edges[col].apply(
            lambda v: v[0] if isinstance(v, list) else v
        )
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_path')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: same directory as the bag)')
    args = ap.parse_args()

    bag_path = str(Path(args.bag_path).resolve())
    out_dir  = Path(args.out_dir) if args.out_dir else Path(bag_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Reading GPS track from bag...')
    min_lat, max_lat, min_lon, max_lon = extract_gps_bbox(bag_path)

    north = max_lat + MARGIN_DEG
    south = min_lat - MARGIN_DEG
    east  = max_lon + MARGIN_DEG
    west  = min_lon - MARGIN_DEG

    print(f'\nDownloading OSM road network...')
    # graph_from_polygon is stable across osmnx versions
    bbox_poly = shapely_box(west, south, east, north)
    G = ox.graph_from_polygon(bbox_poly, network_type='drive', simplify=True)
    print(f'  Nodes : {len(G.nodes)}  Edges : {len(G.edges)}')

    # ── Save GeoJSON ─────────────────────────────────────────────────────────
    geojson_path = out_dir / 'map.geojson'
    _, edges = ox.graph_to_gdfs(G)
    clean_edges(edges).to_file(str(geojson_path), driver='GeoJSON')
    print(f'\nSaved : {geojson_path}')
    print('  → Drag onto the Foxglove Map panel to overlay the road network.')

    # ── Save pickled graph ────────────────────────────────────────────────────
    graph_path = out_dir / 'map_graph.pkl'
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f)
    print(f'Saved : {graph_path}')
    print('  → Pass --map-graph to process_bag.py for map-matched ego position.')

    # ── Quick sanity stats ────────────────────────────────────────────────────
    road_types = {}
    for _, _, data in G.edges(data=True):
        hw = data.get('highway', 'unknown')
        hw = hw[0] if isinstance(hw, list) else hw
        road_types[hw] = road_types.get(hw, 0) + 1
    print('\nRoad type breakdown:')
    for k, v in sorted(road_types.items(), key=lambda x: -x[1]):
        print(f'  {k:<25} {v}')


if __name__ == '__main__':
    main()
