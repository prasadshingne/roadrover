#!/usr/bin/env python3
"""
Generates an OpenDRIVE (.xodr) road network from the OSM graph created by
make_map.py, suitable for use as the road reference in OpenSCENARIO (xosc).

Design choices:
  - Each unique OSM edge → one OpenDRIVE Road with a single straight-line
    geometry from start node to end node (correct topology and lane counts;
    curve fidelity can be improved later).
  - Bidirectional OSM edges (two directed edges u→v and v→u for the same
    physical road) are merged into a single Road with left + right lanes.
  - A geoReference proj string is embedded in the header so simulators
    (esmini, CARLA, etc.) can georeference the map in WGS84.

Output:
  map.xodr  — OpenDRIVE file

Usage:
  python3 make_xodr.py <map_graph.pkl> [--out map.xodr]
"""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
from scenariogeneration import xodr

LANE_WIDTH = 3.5   # metres, standard lane width


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float):
    """Approximate lat/lon → local ENU (East, North) in metres."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0
    y = (lat - lat0) * 111_320.0
    return x, y


def edge_length_and_heading(G, u, v, key, lat0, lon0):
    """
    Return (x_start, y_start, heading_rad, length_m) for an OSM edge,
    using the full polyline length but straight-line heading.
    """
    data = G[u][v][key]
    geom = data.get('geometry')

    if geom is not None:
        coords = list(geom.coords)   # [(lon, lat), …]
        pts = [ll_to_enu(lat, lon, lat0, lon0) for lon, lat in coords]
    else:
        nu, nv = G.nodes[u], G.nodes[v]
        pts = [
            ll_to_enu(nu['y'], nu['x'], lat0, lon0),
            ll_to_enu(nv['y'], nv['x'], lat0, lon0),
        ]

    # Total polyline length (better than straight-line distance for curves)
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    # Heading: tangent at the start of the polyline
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    heading = math.atan2(dy, dx)

    return pts[0][0], pts[0][1], heading, length


def parse_lanes(raw, default=2):
    try:
        return max(1, int(raw[0] if isinstance(raw, list) else raw))
    except (ValueError, TypeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('graph_pkl', help='Path to map_graph.pkl from make_map.py')
    ap.add_argument('--out', default=None, help='Output .xodr path (default: map.xodr next to pkl)')
    args = ap.parse_args()

    graph_path = Path(args.graph_pkl)
    out_path   = Path(args.out) if args.out else graph_path.parent / 'map.xodr'

    print(f'Loading graph : {graph_path}')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f'  {len(G.nodes)} nodes, {len(G.edges)} directed edges')

    # ── ENU reference origin ─────────────────────────────────────────────────
    lats = [d['y'] for _, d in G.nodes(data=True)]
    lons = [d['x'] for _, d in G.nodes(data=True)]
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    print(f'  ENU origin   : lat={lat0:.6f}  lon={lon0:.6f}')

    # ── Build OpenDRIVE ──────────────────────────────────────────────────────
    odr = xodr.OpenDrive('roadrover_map')

    # Embed geoReference so simulators can georeference to WGS84
    geo_ref = (f'+proj=tmerc +lat_0={lat0:.8f} +lon_0={lon0:.8f} '
               f'+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')
    odr._header.geo_reference = geo_ref

    # ── De-duplicate bidirectional edges ─────────────────────────────────────
    # osmnx creates two directed edges (u→v, v→u) for two-way roads.
    # We merge them into one Road with left + right lanes; one-way roads
    # get right lanes only.
    seen   = {}   # canonical_key → (u, v, key, data)
    for u, v, key, data in G.edges(keys=True, data=True):
        canon = (min(u, v), max(u, v), key)
        if canon not in seen:
            seen[canon] = (u, v, key, data)
        # If we already saw (v,u) as canonical, note that both directions exist
        # so the road is bidirectional → handled by left+right lanes below.

    # Collect all canonical keys and determine directionality
    bidirectional = set()
    all_directed  = {(u, v, k) for u, v, k, _ in G.edges(keys=True, data=True)}
    for (a, b, k), (u, v, _, _) in seen.items():
        if (v, u, k) in all_directed:   # reverse edge also exists
            bidirectional.add((a, b, k))

    road_id    = 0
    road_count = 0
    skipped    = 0

    for canon, (u, v, key, data) in seen.items():
        x0, y0, heading, length = edge_length_and_heading(G, u, v, key, lat0, lon0)

        if length < 1.0:
            skipped += 1
            continue

        n_lanes = parse_lanes(data.get('lanes', 2))
        oneway  = data.get('oneway', False)
        is_bidir = canon in bidirectional

        if oneway or not is_bidir:
            right_lanes = n_lanes
            left_lanes  = 0
        else:
            # Two-way road: split lanes between directions
            right_lanes = max(1, n_lanes // 2)
            left_lanes  = max(1, n_lanes - right_lanes)

        road = xodr.create_road(
            geometry=[xodr.Line(length)],
            id=road_id,
            left_lanes=left_lanes,
            right_lanes=right_lanes,
            lane_width=LANE_WIDTH,
        )
        road.planview.set_start_point(x0, y0, heading)

        # Attach OSM metadata as road name for simulator display
        name = data.get('name', '')
        if isinstance(name, list):
            name = name[0]
        road.name = str(name or data.get('highway', ''))

        odr.add_road(road)
        road_id    += 1
        road_count += 1

    print(f'  Roads created : {road_count}  (skipped {skipped} degenerate)')

    print('Adjusting geometry...')
    odr.adjust_roads_and_lanes()

    print(f'Writing : {out_path}')
    odr.write_xml(str(out_path))

    print('\nDone.')
    print(f'  ENU origin for xosc georeferencing:')
    print(f'    lat0={lat0:.8f}  lon0={lon0:.8f}')
    print(f'  Verify in esmini:')
    print(f'    esmini --odr {out_path.name} --window 60 60 1200 800')


if __name__ == '__main__':
    main()
