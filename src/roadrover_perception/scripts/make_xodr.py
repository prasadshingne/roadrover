#!/usr/bin/env python3
"""
Generates an OpenDRIVE (.xodr) road network from the OSM graph created by
make_map.py, suitable for use as the road reference in OpenSCENARIO (xosc).

Design choices:
  - Each OSM edge polyline segment → one OpenDRIVE Road (one Road per
    consecutive pair of polyline points).  This preserves the actual curved
    shape of roads instead of a single chord from start node to end node.
  - Bidirectional OSM edges get left + right lanes split symmetrically around
    the OSM centreline.  One-way edges get all lanes on the right, with the
    reference line shifted left by n_lanes/2 * LANE_WIDTH so that the
    outermost (rightmost) lane centre aligns with process_bag.py's lane-1
    position.  Lane counts are inherited from the parent OSM edge.
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
MIN_SEG_LEN = 0.5  # metres — skip degenerate sub-segments shorter than this


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float):
    """Approximate lat/lon → local ENU (East, North) in metres."""
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0
    y = (lat - lat0) * 111_320.0
    return x, y


def edge_segments(G, u, v, key, lat0, lon0):
    """
    Return a list of (x0, y0, heading_rad, length_m) tuples — one per
    sub-segment of the OSM edge's polyline geometry.

    For edges without an intermediate geometry (just two nodes), this returns
    a single-element list with the straight-line chord.
    """
    data = G[u][v][key]
    geom = data.get('geometry')

    if geom is not None:
        coords = list(geom.coords)              # [(lon, lat), …]
        pts = [ll_to_enu(lat, lon, lat0, lon0) for lon, lat in coords]
    else:
        nu, nv = G.nodes[u], G.nodes[v]
        pts = [
            ll_to_enu(nu['y'], nu['x'], lat0, lon0),
            ll_to_enu(nv['y'], nv['x'], lat0, lon0),
        ]

    segments = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length  = math.sqrt(dx * dx + dy * dy)
        if length < MIN_SEG_LEN:
            continue
        segments.append((x0, y0, math.atan2(dy, dx), length))

    return segments


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
    seen = {}   # canonical_key → (u, v, key, data)
    for u, v, key, data in G.edges(keys=True, data=True):
        canon = (min(u, v), max(u, v), key)
        if canon not in seen:
            seen[canon] = (u, v, key, data)

    all_directed = {(u, v, k) for u, v, k, _ in G.edges(keys=True, data=True)}
    bidirectional = set()
    for (a, b, k), (u, v, _, _) in seen.items():
        if (v, u, k) in all_directed:
            bidirectional.add((a, b, k))

    road_id    = 0
    road_count = 0
    seg_skipped = 0

    for canon, (u, v, key, data) in seen.items():
        segs = edge_segments(G, u, v, key, lat0, lon0)

        n_lanes  = parse_lanes(data.get('lanes', 2))
        oneway   = data.get('oneway', False)
        is_bidir = canon in bidirectional

        if oneway or not is_bidir:
            # One-way: all lanes on the right of the reference line.
            # process_bag.py places lane centres at multiples of LANE_WIDTH from
            # the OSM snap (treating it as the road centreline), so the rightmost
            # lane centre sits at -(n_lanes/2 - 0.5)*LANE_WIDTH from the snap.
            # xodr lane -k centre is at (k-0.5)*LANE_WIDTH right of the reference.
            # Shifting the reference LEFT by n_lanes/2*LANE_WIDTH makes
            # lane -n_lanes centre coincide with process_bag's lane-1 position.
            right_lanes  = n_lanes
            left_lanes   = 0
            shift_left_m = n_lanes / 2.0 * LANE_WIDTH
        else:
            # Bidirectional: lanes split symmetrically; OSM snap IS the centre.
            right_lanes  = max(1, n_lanes // 2)
            left_lanes   = max(1, n_lanes - right_lanes)
            shift_left_m = 0.0

        name = data.get('name', '')
        if isinstance(name, list):
            name = name[0]
        road_name = str(name or data.get('highway', ''))

        for x0, y0, heading, length in segs:
            if length < MIN_SEG_LEN:
                seg_skipped += 1
                continue

            # Perpendicular-left unit vector (rotate heading 90° CCW)
            perp_x = -math.sin(heading)
            perp_y =  math.cos(heading)
            xs = x0 + shift_left_m * perp_x
            ys = y0 + shift_left_m * perp_y

            road = xodr.create_road(
                geometry=[xodr.Line(length)],
                id=road_id,
                left_lanes=left_lanes,
                right_lanes=right_lanes,
                lane_width=LANE_WIDTH,
            )
            road.planview.set_start_point(xs, ys, heading)
            road.name = road_name

            odr.add_road(road)
            road_id    += 1
            road_count += 1

    print(f'  Roads created : {road_count}  (skipped {seg_skipped} degenerate segments)')

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
