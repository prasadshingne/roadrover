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
import json
import math
import pickle
from pathlib import Path

import numpy as np
import osmnx as ox
from shapely.geometry import LineString, box as shapely_box

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

FIX_TOPIC  = '/fix'
MARGIN_DEG = 0.003   # ~300 m padding around the GPS bounding box
LANE_WIDTH = 3.5     # metres


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


def _ll_to_enu(lat, lon, lat0, lon0):
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0
    y = (lat - lat0) * 111_320.0
    return x, y


def _enu_to_lonlat(x, y, lat0, lon0):
    lat = y / 111_320.0 + lat0
    lon = x / (111_320.0 * math.cos(math.radians(lat0))) + lon0
    return lon, lat


def _line_feature(coords_lonlat, road_name, highway, boundary_type):
    return {
        'type': 'Feature',
        'geometry': {'type': 'LineString', 'coordinates': list(coords_lonlat)},
        'properties': {'name': str(road_name), 'highway': str(highway),
                       'boundary': boundary_type},
    }


def generate_lane_geojson(G, lat0, lon0, out_path):
    """
    Build lane boundary GeoJSON from the osmnx graph.
    For each road edge, offsets the full polyline geometry by multiples of
    LANE_WIDTH in ENU space (metres) to produce individual lane boundaries.
    """
    all_directed = {(u, v, k) for u, v, k, _ in G.edges(keys=True, data=True)}
    features = []
    seen = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        canon = (min(u, v), max(u, v), key)
        if canon in seen:
            continue
        seen.add(canon)

        # Polyline in ENU (metres)
        geom = data.get('geometry')
        if geom is not None:
            enu_pts = [_ll_to_enu(lat, lon, lat0, lon0) for lon, lat in geom.coords]
        else:
            nu, nv = G.nodes[u], G.nodes[v]
            enu_pts = [_ll_to_enu(nu['y'], nu['x'], lat0, lon0),
                       _ll_to_enu(nv['y'], nv['x'], lat0, lon0)]

        enu_line = LineString(enu_pts)
        if enu_line.length < 1.0:
            continue

        # Lane counts
        try:
            raw = data.get('lanes', 2)
            n_lanes = max(1, int(raw[0] if isinstance(raw, list) else raw))
        except (ValueError, TypeError):
            n_lanes = 2

        oneway  = data.get('oneway', False)
        is_bidir = (v, u, key) in all_directed
        if oneway or not is_bidir:
            r_lanes, l_lanes = n_lanes, 0
        else:
            r_lanes = max(1, n_lanes // 2)
            l_lanes = max(1, n_lanes - r_lanes)

        name = data.get('name', '') or ''
        if isinstance(name, list): name = name[0]
        hw = data.get('highway', '') or ''
        if isinstance(hw, list): hw = hw[0]

        def offset_to_lonlat(line, dist):
            try:
                off = line.offset_curve(dist)
                if off is None or off.is_empty:
                    return []
                parts = off.geoms if hasattr(off, 'geoms') else [off]
                return [[_enu_to_lonlat(x, y, lat0, lon0) for x, y in p.coords]
                        for p in parts]
            except Exception:
                return []

        # Always draw the OSM centreline for reference
        coords_ll = [_enu_to_lonlat(x, y, lat0, lon0) for x, y in enu_line.coords]
        features.append(_line_feature(coords_ll, name, hw, 'center'))

        # For one-way roads the OSM edge runs through the physical road centre,
        # so lanes must be drawn symmetrically around it (half-width to each side).
        # For bidirectional roads the OSM edge is the centre divider; right-side
        # lanes go right (-) and left-side lanes go left (+).
        if l_lanes == 0:
            # One-way: centre the n lanes on the OSM edge
            r_base = r_lanes / 2.0 * LANE_WIDTH   # left road edge offset
            for i in range(r_lanes + 1):
                offset = r_base - i * LANE_WIDTH
                btype  = 'edge' if (i == 0 or i == r_lanes) else 'lane'
                for coords_ll in offset_to_lonlat(enu_line, offset):
                    features.append(_line_feature(coords_ll, name, hw, btype))
        else:
            # Bidirectional: right lanes go right, left lanes go left
            for i in range(1, r_lanes + 1):
                btype = 'edge' if i == r_lanes else 'lane'
                for coords_ll in offset_to_lonlat(enu_line, -i * LANE_WIDTH):
                    features.append(_line_feature(coords_ll, name, hw, btype))
            for i in range(1, l_lanes + 1):
                btype = 'edge' if i == l_lanes else 'lane'
                for coords_ll in offset_to_lonlat(enu_line, i * LANE_WIDTH):
                    features.append(_line_feature(coords_ll, name, hw, btype))

    geojson = {'type': 'FeatureCollection', 'features': features}
    with open(out_path, 'w') as f:
        json.dump(geojson, f)
    print(f'Saved : {out_path}  ({len(features)} lane boundary segments)')


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

    # ── ENU origin for lane offset calculations ──────────────────────────────
    all_lats = [d['y'] for _, d in G.nodes(data=True)]
    all_lons = [d['x'] for _, d in G.nodes(data=True)]
    lat0 = float(np.mean(all_lats))
    lon0 = float(np.mean(all_lons))

    # ── Save road centerline GeoJSON (lightweight reference) ─────────────────
    geojson_path = out_dir / 'map.geojson'
    _, edges = ox.graph_to_gdfs(G)
    clean_edges(edges).to_file(str(geojson_path), driver='GeoJSON')
    print(f'\nSaved : {geojson_path}  (road centerlines)')

    # ── Save lane boundary GeoJSON (used by process_bag.py for Foxglove) ─────
    lanes_path = out_dir / 'lanes.geojson'
    generate_lane_geojson(G, lat0, lon0, lanes_path)
    print('  → process_bag.py will publish this as /map/lanes in Foxglove.')

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
