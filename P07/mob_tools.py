"""The toolbox of Precept 07, Traces and Networks.

Every algorithm the Lecture 07 figures and the P07 notebooks use lives here once, documented, so that a code
cell in a notebook is a few lines that call a named tool: the street network of a town as a graph with shortest
paths, isochrones and centralities on it; the two accessibility measures, cumulative opportunities and Hansen's
gravity-based score; the stay rule that turns a sequence of GPS fixes into stays and trips and its scoring
against a coded diary; origin-destination matrices on a grid, the gravity model fitted and the radiation model
in closed form; the uniqueness of a few spatio-temporal points, aggregation with a minimum count and Laplace
noise; and the plots the deck and the notebooks share. The course pipeline that computes the reference numbers
imports the same file, so the notebooks and their numbers cannot drift apart.

osmnx is imported only inside the two functions that need it (loading a GraphML and snapping fixes to edges),
so the module loads without a network and without osmnx; everything else is numpy, pandas, geopandas, shapely,
networkx, scipy and scikit-learn.

The graphs section also writes the shortest path out in depth: Dijkstra's method round by round and on a binary
heap, A* with the straight line as its guide, the search from both ends, the path of fewest edges, the k shortest
paths and a dissimilar alternative, a contraction hierarchy on a small graph and Brandes' dependencies. The flows
section fits one column of the matrix, the commuters into one destination, by the gravity and the radiation model.
The privacy section turns the mechanisms of Lecture 07's fourth chapter into numbers: one person's unicity test,
anonymity sets, the distinct people behind a heatmap's cells, a small key space hashed and timed, the uniqueness
lost per halving of the resolution, and the recording day as the individual behind a trace that names nobody.

Sections: graphs, accessibility, traces, flows, privacy, plotting.
"""

import hashlib
import heapq
import itertools
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    # graphs
    "SEED", "CRS", "WALK_KPH", "BIKE_KPH", "load_graph", "simple_graph", "set_speed", "node_xy", "nearest_nodes", "route",
    "straight_line", "circuity", "times_from", "isochrone", "isochrone_polygon", "circle", "basic_stats", "orientation",
    "centrality", "edge_betweenness", "degree_histogram", "dual_graph", "graph_from_edges",
    # shortest paths in depth: Dijkstra round by round and on a heap, A*, both ends, fewest edges, alternatives, shortcuts, Brandes
    "search", "dijkstra_steps", "straight_line_heuristic", "bidirectional_search", "fewest_edges_path",
    "k_shortest_paths", "dissimilar_alternative", "contraction_hierarchy", "brandes_dependencies", "all_pairs_cost",
    # accessibility
    "travel_time_matrix", "nearest_times", "cumulative", "hansen", "share_within", "lorenz", "gini",
    # traces
    "load_fixes", "haversine_m", "step_lengths", "stays", "trips", "match_stays", "radius_of_gyration", "snap_to_edges",
    # flows
    "make_grid", "cell_of", "od_matrix", "desire_lines", "length_histogram", "fit_power_law", "gonzalez_pdf", "gravity_fit",
    "radiation", "cpc",
    # flows into one destination
    "gravity_fit_destination", "radiation_to", "rate_by_band",
    # privacy
    "uniqueness", "aggregate", "laplace_noise", "key_space",
    # the privacy chapter's mechanisms as numbers
    "example_uniqueness", "anonymity_sets", "distinct_per_cell", "hash_keys", "halving_factors", "recording_days",
    # plotting
    "show_graph", "show_isochrones", "show_orientation", "show_access", "show_flows", "show_traces", "show_od",
    "show_scatter_fit", "show_loglog",
    # your own trace (the P07 kit): reading, local time, projection, trimming, the diary, the aggregate release
    "NO_TRACE", "LOCAL_TZ", "first_trace", "read_gpx", "read_trace", "local_seconds", "project", "fixes_from_trace", "trim_ends",
    "clean_fixes", "parse_clock", "next_departure",
    "read_diary", "coded_stops", "threshold_surface", "cell_hours", "grid_cells", "published_counts", "clock",
    "edge_names", "top_streets", "block_group_means", "show_edges",
    "show_lorenz",
]

SEED = 20261102                 # the lecture's seed, the date of Lecture 07
CRS = "EPSG:26918"              # NAD83 / UTM 18N, the course's CRS for every vector product
WALK_KPH = 4.8                  # 80 m a minute
BIKE_KPH = 15.0
EDGE_ATTRS = ("length", "speed_kph", "travel_time")
LOCAL_TZ = "America/New_York"           # the hunt's clock; the specimen's times are seconds since local midnight too


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #

def load_graph(path):
    """A street network saved by osmnx as GraphML, with `length` (m), `speed_kph` and `travel_time` (s) asserted on
    every edge, projected (its CRS is in `G.graph["crs"]`)."""
    import osmnx as ox
    G = ox.io.load_graphml(path)
    for u, v, k, d in G.edges(keys=True, data=True):
        for a in EDGE_ATTRS:
            assert a in d, f"edge {u} {v} {k} lacks {a}; run data/scripts/32_fetch_street_networks.py"
    return G


def simple_graph(G, weight="length"):
    """The undirected simple graph of a street network: one edge per pair of intersections, carrying the smallest
    `weight` of the parallel edges and the node coordinates. Centralities and paths that ignore one-way streets run
    on it; it is what networkx's centrality functions expect."""
    import networkx as nx
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(n, x=d["x"], y=d["y"])
    for u, v, d in G.edges(data=True):
        w = float(d[weight])
        if H.has_edge(u, v):
            if w < H[u][v][weight]:
                H[u][v][weight] = w
        else:
            H.add_edge(u, v, **{weight: w})
    return H


def set_speed(G, kph):
    """One speed on every edge and the travel time in seconds that follows from its length."""
    for u, v, k, d in G.edges(keys=True, data=True):
        d["speed_kph"] = float(kph)
        d["travel_time"] = float(d["length"]) / (kph / 3.6)
    return G


def node_xy(G):
    """The nodes as a frame of id, x, y in the graph's CRS."""
    return pd.DataFrame([(n, d["x"], d["y"]) for n, d in G.nodes(data=True)], columns=["node", "x", "y"]).set_index("node")


def nearest_nodes(G, x, y):
    """The node nearest to each (x, y) in the graph's CRS; a scalar pair gives one node, arrays give a list."""
    from scipy.spatial import cKDTree
    xy = node_xy(G)
    tree = cKDTree(xy[["x", "y"]].to_numpy())
    scalar = np.isscalar(x)
    _, idx = tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
    nodes = [xy.index[i] for i in idx]
    return nodes[0] if scalar else nodes


def route(G, origin, destination, weight="length"):
    """The shortest path between two nodes as (nodes, total weight); Dijkstra's method on the graph as given."""
    import networkx as nx
    nodes = nx.shortest_path(G, origin, destination, weight=weight)
    total = nx.path_weight(G, nodes, weight=weight)
    return nodes, float(total)


def straight_line(G, origin, destination):
    """The straight-line distance between two nodes in the graph's CRS."""
    a, b = G.nodes[origin], G.nodes[destination]
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def circuity(G, origin, destination, weight="length"):
    """Network length over straight-line length for one pair, at least 1; the inverse is the trip's directness."""
    _, length = route(G, origin, destination, weight)
    return length / max(straight_line(G, origin, destination), 1e-9)


def times_from(G, sources, weight="travel_time", cutoff=None):
    """The shortest `weight` from the nearest of the `sources` to every node that can be reached (a multi-source
    Dijkstra), as a dict node to value; `cutoff` stops the search beyond that value."""
    import networkx as nx
    return nx.multi_source_dijkstra_path_length(G, set(sources), cutoff=cutoff, weight=weight)


def isochrone(G, node, minutes, weight="travel_time"):
    """Every node within `minutes` of `node` along the network with its time in seconds, as a dict."""
    import networkx as nx
    return nx.single_source_dijkstra_path_length(G, node, cutoff=minutes * 60.0, weight=weight)


def isochrone_polygon(G, reached, buffer_m=25.0):
    """The reached edges (both ends in `reached`) buffered and dissolved into one polygon, the usual way to draw the
    area an isochrone covers; `reached` is the dict `isochrone()` returns or any set of nodes."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    keep = set(reached)
    lines = []
    for u, v, d in G.edges(data=True):
        if u in keep and v in keep:
            geom = d.get("geometry")
            if geom is None:
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
            lines.append(geom)
    if not lines:
        return None
    return unary_union([g.buffer(buffer_m) for g in lines])


def circle(xy, radius_m):
    """The straight-line disc of the same budget, the contrast every isochrone is drawn against."""
    from shapely.geometry import Point
    return Point(xy).buffer(radius_m)


def basic_stats(G, area_m2=None):
    """osmnx's basic statistics of a street network as a flat dict (counts, densities per square kilometre, mean
    segment length, circuity, streets per node)."""
    import osmnx as ox
    s = ox.stats.basic_stats(G, area=area_m2)
    out = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in s.items() if not isinstance(v, dict)}
    for k in ("streets_per_node_counts", "streets_per_node_proportions"):
        if isinstance(s.get(k), dict):
            out[k] = {int(a): float(b) for a, b in s[k].items()}
    return out


def orientation(G, bins=36, attr="bearing"):
    """The compass orientation of a street network (Boeing 2019): the bearings of every segment and their reciprocals
    in `bins` equal bins with 0 degrees at the centre of the first bin, Shannon's entropy of the bin shares in nats,
    and the order index 1 minus the square of (H minus ln 4) over (ln bins minus ln 4), which puts a perfect grid at 1
    and a uniform spread at 0 (the paper squares the ratio to linearise it; Chicago reads 0.899 and Charlotte 0.002).
    Edges must carry a `bearing` added before projection."""
    seen = set()
    bearings = []
    for u, v, k, d in G.edges(keys=True, data=True):
        if (v, u, k) in seen or attr not in d:
            continue
        seen.add((u, v, k))
        b = float(d[attr])
        bearings.extend([b % 360.0, (b + 180.0) % 360.0])
    width = 360.0 / bins
    idx = np.floor(((np.asarray(bearings) + width / 2) % 360.0) / width).astype(int)
    counts = np.bincount(idx, minlength=bins).astype(float)
    p = counts / counts.sum()
    H = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    Hmax, Hgrid = math.log(bins), math.log(4)
    order = 1.0 - min(max((H - Hgrid) / (Hmax - Hgrid), 0.0), 1.0) ** 2
    return {"counts": counts.tolist(), "entropy": H, "order": order, "n_segments": len(seen), "bins": bins}


def centrality(G, weight="length", k=None, seed=SEED):
    """Betweenness and closeness of every intersection and betweenness of every street segment on the undirected
    simple graph (Freeman 1977; Brandes 2001 as networkx implements it), normalised; as a node frame (x, y, degree,
    betweenness, closeness) and an edge frame (u, v, betweenness). With `k`, the betweenness is estimated from k
    source nodes drawn with the seed (networkx's sampling, rescaled), and exact when k is None or at least the node
    count; closeness is always exact."""
    import networkx as nx
    H = simple_graph(G, weight)
    kk = None if k is None or int(k) >= H.number_of_nodes() else int(k)
    bc = nx.betweenness_centrality(H, k=kk, weight=weight, normalized=True, seed=seed)
    cc = nx.closeness_centrality(H, distance=weight)
    nodes = pd.DataFrame({"x": [H.nodes[n]["x"] for n in H], "y": [H.nodes[n]["y"] for n in H],
                          "degree": [H.degree(n) for n in H], "betweenness": [bc[n] for n in H],
                          "closeness": [cc[n] for n in H]}, index=pd.Index(list(H), name="node"))
    return nodes, _edge_frame(H, weight, kk, seed)


def _edge_frame(H, weight, kk, seed):
    import networkx as nx
    ebc = nx.edge_betweenness_centrality(H, k=kk, weight=weight, normalized=True, seed=seed)
    return pd.DataFrame([(u, v, ebc[(u, v)] if (u, v) in ebc else ebc[(v, u)], H[u][v][weight]) for u, v in H.edges()],
                        columns=["u", "v", "betweenness", weight])


def edge_betweenness(G, weight="length", k=None, seed=SEED):
    """The betweenness of every street segment alone, the edge frame of `centrality` (u, v, betweenness and the weight)
    without the node betweenness and the closeness, which cost most of its time on a large graph. Exact when k is None
    or at least the node count, else estimated from k sources drawn with the seed."""
    H = simple_graph(G, weight)
    kk = None if k is None or int(k) >= H.number_of_nodes() else int(k)
    return _edge_frame(H, weight, kk, seed)


def degree_histogram(G):
    """The number of intersections by their number of streets, on the simple graph (1 is a dead end)."""
    H = simple_graph(G)
    return dict(sorted(Counter(dict(H.degree()).values()).items()))


def graph_from_edges(edges, layout=None, unit=1.0):
    """A small undirected graph from a list of (u, v, w) triples, the edge length `unit` times w (the lecture's toy
    states its lengths in hundreds of metres, so unit 100 gives metres), with x and y on every node from `layout`, a
    mapping of node to drawing coordinates in metres, when one is given."""
    import networkx as nx
    H = nx.Graph()
    for u, v, w in edges:
        H.add_edge(u, v, length=float(unit) * float(w))
    for n, (x, y) in (layout or {}).items():
        H.nodes[n]["x"] = float(x)
        H.nodes[n]["y"] = float(y)
    return H


def dual_graph(G):
    """The line graph of the simple graph: street segments become nodes, joined when they meet at an intersection."""
    import networkx as nx
    return nx.line_graph(simple_graph(G))


# shortest paths in depth

def _edges_out(G, u, weight):
    """The edges leaving `u` as (neighbour, weight) pairs, a multigraph's parallel edges at their smallest weight."""
    if G.is_multigraph():
        return [(v, min(float(d[weight]) for d in keys.values())) for v, keys in G[u].items()]
    return [(v, float(d[weight])) for v, d in G[u].items()]


def _edges_in(G, u, weight):
    """The edges arriving at `u` as (neighbour, weight) pairs; on an undirected graph the same as the edges leaving it."""
    if not G.is_directed():
        return _edges_out(G, u, weight)
    if G.is_multigraph():
        return [(v, min(float(d[weight]) for d in keys.values())) for v, keys in G.pred[u].items()]
    return [(v, float(d[weight])) for v, d in G.pred[u].items()]


def _path_to(pred, node):
    """The path from the root of a predecessor dict to `node`."""
    path = [node]
    while pred[path[-1]] is not None:
        path.append(pred[path[-1]])
    return path[::-1]


def _simple(G, weight):
    """A simple graph with the same direction as `G` and one edge per pair at the smallest weight of its parallel edges,
    keeping the node attributes; what networkx's simple-path functions expect."""
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = float(d[weight])
        if not H.has_edge(u, v) or w < H[u][v][weight]:
            H.add_edge(u, v, **{weight: w})
    return H


def _length(G, path, weight):
    """The length of a path as the sum of its edges' weights, a multigraph's parallel edges at their smallest weight."""
    return sum(min(w for x, w in _edges_out(G, u, weight) if x == v) for u, v in zip(path, path[1:]))


def _shared(G, path, first, weight):
    """The share of the length of `path` that runs on edges of `first`, the direction of travel ignored."""
    edges = {frozenset(e) for e in zip(first, first[1:])}
    common = sum(min(w for x, w in _edges_out(G, u, weight) if x == v) for u, v in zip(path, path[1:]) if frozenset((u, v)) in edges)
    total = _length(G, path, weight)
    return common / total if total > 0 else 1.0


# --------------------------------------------------------------------------- #
# one pair, one source
# --------------------------------------------------------------------------- #

def search(G, source, target=None, weight="length", heuristic=None):
    """Dijkstra's method from `source` on a binary heap, or A* towards `target` when a `heuristic` is given.

    Every node keeps a tentative distance g, infinite until the node is reached, and a predecessor. The heap holds
    entries (key, count, node) with the key g for Dijkstra and g + h for A*, so the entry popped next belongs to the
    unsettled node of least key; an entry whose node is already settled is skipped (a lazy heap: an improvement pushes
    a new entry instead of lowering the old one). Popping a node settles it, and relaxing its edges lowers a
    neighbour's g when the route through the settled node is shorter, d(v) <- min(d(v), d(u) + l_uv). With a target
    the search stops when the target is popped; without one it settles every node it can reach, and the predecessors
    form the shortest-path tree. On a binary heap the whole run costs O((n + m) log n).

    A* needs a heuristic that never overestimates the remaining distance to the target (admissible). One that also
    obeys h(u) <= l_uv + h(v) (consistent, as the straight line is when no edge is shorter than its chord) settles every
    node once; if a shorter route to a settled node turns up all the same, the node is reopened and settled again, so
    the answer stays the shortest for any admissible heuristic, and `reopened` counts how often that happened.

    Returns a dict: `distance` and `path` to the target (None without a target or when it cannot be reached),
    `settled` (the nodes in the order they were settled, a reopened node again), `pushes` (heap insertions, the
    source's included), `reopened`, and `dist` and `pred` for every node reached (final for the settled nodes).
    """
    if heuristic is not None and target is None:
        raise ValueError("A* steers towards a target; give one with the heuristic")
    h = heuristic if heuristic is not None else (lambda v: 0.0)
    tick = itertools.count()
    dist, pred = {source: 0.0}, {source: None}
    heap = [(h(source), next(tick), source)]
    pushes, reopened = 1, 0
    done, settled = set(), []
    while heap:
        _, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        settled.append(u)
        if u == target:
            break
        du = dist[u]
        for v, w in _edges_out(G, u, weight):
            nd = du + w
            if nd < dist.get(v, math.inf):
                dist[v], pred[v] = nd, u
                if v in done:              # only a heuristic that is not consistent can lower a settled node
                    done.discard(v)
                    reopened += 1
                heapq.heappush(heap, (nd + h(v), next(tick), v))
                pushes += 1
    reached = target is not None and target in done
    return {"distance": dist[target] if reached else None, "path": _path_to(pred, target) if reached else None,
            "settled": settled, "pushes": pushes, "reopened": reopened, "dist": dist, "pred": pred}


def dijkstra_steps(G, source, weight="length", target=None):
    """Dijkstra's method round by round, for a table on a slide or a notebook cell on a small graph.

    Each round settles the unsettled node of least tentative distance (equal distances go to the node that sorts
    first, so the rounds are the same every time) and then looks at every edge of that node. An edge to a settled node
    is a check that changes nothing; an edge to an unsettled node is relaxed and improves the neighbour when the route
    through the settled node is shorter. On an undirected graph every edge is looked at from both of its ends, so a
    run over the whole graph makes 2m checks. Finding the least tentative distance by scanning the unsettled nodes makes
    this the O(n^2) version of the method; `search` is the same method on a heap.

    Returns a dict with `rounds`, one dict per round: the node `settled` and its `distance`, the edges `relaxed` this
    round as (u, v, old, new) with old None for infinity, the edges `improved` among them, and after the round the
    `tentative` distances and the predecessors `pred` of every node (None for infinity and for none); and the totals
    `checks` and `improvements`, the settling `order` and the final `dist` and `pred`.
    """
    try:
        rank = {n: i for i, n in enumerate(sorted(G.nodes()))}
    except TypeError:
        rank = {n: i for i, n in enumerate(sorted(G.nodes(), key=str))}
    tent = {n: math.inf for n in G.nodes()}
    pred = {n: None for n in G.nodes()}
    tent[source] = 0.0
    done, order, rounds = set(), [], []
    checks = improvements = 0
    while True:
        open_ = [n for n in G.nodes() if n not in done and not math.isinf(tent[n])]
        if not open_:
            break
        u = min(open_, key=lambda n: (tent[n], rank[n]))
        done.add(u)
        order.append(u)
        relaxed, improved = [], []
        for v, w in _edges_out(G, u, weight):
            checks += 1
            if v in done:
                continue
            old, new = tent[v], tent[u] + w
            relaxed.append((u, v, None if math.isinf(old) else old, new))
            if new < old:
                tent[v], pred[v] = new, u
                improved.append((u, v, None if math.isinf(old) else old, new))
                improvements += 1
        rounds.append({"settled": u, "distance": tent[u], "relaxed": relaxed, "improved": improved,
                       "tentative": {n: (None if math.isinf(d) else d) for n, d in tent.items()}, "pred": dict(pred)})
        if u == target:
            break
    return {"rounds": rounds, "checks": checks, "improvements": improvements, "order": order,
            "dist": {n: d for n, d in tent.items() if not math.isinf(d)}, "pred": pred}


def straight_line_heuristic(G, target, speed_kph=None, tolerance_m=0.01):
    """The straight line from a node to the `target`, h(v) = ||x_v - x_t||, as A*'s guide: in metres, or in seconds at
    `speed_kph` for edges weighted by travel time (admissible only at a speed that no edge exceeds).

    A street is never shorter than the straight line between its ends, so the straight line never overestimates what
    is left to walk, and the triangle inequality of the plane makes it consistent. Stored lengths are measured on a
    sphere, though, while the map's coordinates come from the ellipsoid, so an edge running east to west can come out
    up to about 0.2 percent shorter than its chord:
    the returned function carries `short_edges`, the number of edges shorter than their chord by more than
    `tolerance_m`, and `short_by_m`, the largest shortfall, which bounds how far A* could stray from the shortest.
    """
    xt, yt = float(G.nodes[target]["x"]), float(G.nodes[target]["y"])
    scale = 1.0 if speed_kph is None else 3.6 / float(speed_kph)

    def h(v):
        d = G.nodes[v]
        return math.hypot(float(d["x"]) - xt, float(d["y"]) - yt) * scale

    short = []
    for u, v, d in G.edges(data=True):
        if u == v or "length" not in d:
            continue
        chord = math.hypot(float(G.nodes[u]["x"]) - float(G.nodes[v]["x"]), float(G.nodes[u]["y"]) - float(G.nodes[v]["y"]))
        if chord - float(d["length"]) > tolerance_m:
            short.append(chord - float(d["length"]))
    h.short_edges = len(short)
    h.short_by_m = max(short) if short else 0.0
    return h


def bidirectional_search(G, source, target, weight="length"):
    """Dijkstra from both ends at once: one node settled forward from the source, then one backward from the target
    against the directions of the edges, in turn, until the two searches meet.

    Every edge that joins a node the forward search reached to a node the backward search reached offers a route, and
    the shortest of those seen so far is mu. The searches stop the first time a node is settled by both (Newman's
    two-source search, carried over to lengths), and the answer is mu: the route through that node is not always the
    shortest, but no route shorter than mu can remain unseen once a node is settled from both ends. A sharper test,
    stopping as soon as the least keys on the two heaps add up to at least mu, stops a little earlier.

    Returns a dict: `distance`, `path`, `met` (the node settled by both), `meeting` (the edge where the best route
    passes from the forward to the backward search), the `forward` and `backward` settled lists and `pushes`.
    """
    if source == target:
        return {"distance": 0.0, "path": [source], "met": source, "meeting": None, "forward": [source], "backward": [],
                "pushes": 1}
    tick = itertools.count()
    dist = ({source: 0.0}, {target: 0.0})
    pred = ({source: None}, {target: None})
    heaps = ([(0.0, next(tick), source)], [(0.0, next(tick), target)])
    done = (set(), set())
    settled = ([], [])
    pushes, mu, meeting, met = 2, math.inf, None, None
    side = 0
    while True:
        while heaps[side] and heaps[side][0][2] in done[side]:     # an entry of a settled node is stale
            heapq.heappop(heaps[side])
        if not heaps[side]:
            break
        du, _, u = heapq.heappop(heaps[side])
        done[side].add(u)
        settled[side].append(u)
        other = 1 - side
        if u in done[other]:
            met = u
            break
        for v, w in (_edges_out(G, u, weight) if side == 0 else _edges_in(G, u, weight)):
            nd = du + w
            if nd < dist[side].get(v, math.inf):
                dist[side][v], pred[side][v] = nd, u
                heapq.heappush(heaps[side], (nd, next(tick), v))
                pushes += 1
            if v in dist[other] and du + w + dist[other][v] < mu:
                mu = du + w + dist[other][v]
                meeting = (u, v) if side == 0 else (v, u)
        side = other
    if meeting is None:
        return {"distance": None, "path": None, "met": met, "meeting": None, "forward": settled[0], "backward": settled[1],
                "pushes": pushes}
    a, b = meeting
    back = _path_to(pred[1], b)[::-1]          # from b to the target along the backward search's tree
    return {"distance": mu, "path": _path_to(pred[0], a) + back, "met": met, "meeting": meeting, "forward": settled[0],
            "backward": settled[1], "pushes": pushes}


def fewest_edges_path(G, source, target, weight="length"):
    """The path with the fewest edges, which breadth-first search finds in O(n + m) by visiting the nodes in rings of
    one, two, three edges from the source; of several paths with the fewest edges the shortest in `weight`.

    Returns a dict: `path`, `edges` (its number of edges), `length` (its weight) and `hops`, the number of edges from
    the source to every node it can reach.
    """
    hops = {source: 0}
    ring = [source]
    order = [source]
    while ring:
        nxt = []
        for u in ring:
            for v, _ in _edges_out(G, u, weight):
                if v not in hops:
                    hops[v] = hops[u] + 1
                    nxt.append(v)
        order.extend(nxt)
        ring = nxt
    if target not in hops:
        return {"path": None, "edges": None, "length": None, "hops": hops}
    best, pred = {source: 0.0}, {source: None}
    for u in order:                         # ring by ring: every node's predecessors one ring in are done before it
        for v, w in _edges_out(G, u, weight):
            if hops.get(v) == hops[u] + 1 and best[u] + w < best.get(v, math.inf):
                best[v], pred[v] = best[u] + w, u
    path = _path_to(pred, target)
    return {"path": path, "edges": len(path) - 1, "length": best[target], "hops": hops}


# --------------------------------------------------------------------------- #
# more than one path
# --------------------------------------------------------------------------- #

def k_shortest_paths(G, source, target, k=50, weight="length"):
    """The k shortest simple paths from `source` to `target` in order of length (Yen 1971, as networkx's
    shortest_simple_paths implements it), each as a dict with its `path`, its `length` and `shared`, the share of its
    length that runs on edges of the first path. On a street graph the next shortest paths are mostly the first one
    with a corner cut differently, which is why routers look for dissimilar alternatives instead."""
    H = _simple(G, weight)
    out = []
    for path in itertools.islice(nx.shortest_simple_paths(H, source, target, weight=weight), k):
        out.append({"path": list(path), "length": _length(H, path, weight)})
    first = out[0]["path"] if out else []
    for p in out:
        p["shared"] = _shared(H, p["path"], first, weight)
    return out


def dissimilar_alternative(G, path, weight="length"):
    """Link elimination: forbid every edge of `path` and search again between its two ends, which gives a route that
    shares no edge with the first, the simplest way a router offers a real alternative. Returns a dict with `path`,
    `length` and `shared` (zero by construction), or None values when the two ends are no longer connected."""
    H = _simple(G, weight)
    for u, v in zip(path, path[1:]):
        for a, b in ((u, v), (v, u)):
            if H.has_edge(a, b):
                H.remove_edge(a, b)
    res = search(H, path[0], path[-1], weight)
    if res["path"] is None:
        return {"path": None, "length": None, "shared": None}
    return {"path": res["path"], "length": res["distance"], "shared": _shared(H, res["path"], path, weight)}


# --------------------------------------------------------------------------- #
# routing at scale and all pairs
# --------------------------------------------------------------------------- #

def contraction_hierarchy(G, order, weight="length", source=None, target=None):
    """A contraction hierarchy on a small undirected graph, the idea behind the routers that answer for a continent in
    microseconds (Geisberger et al. 2008).

    The nodes are contracted one at a time in `order`, least important first, and a node's position in the order is
    its rank. Contracting v takes it out of the graph and, for every pair u, w of its neighbours still in the graph,
    adds a shortcut u-w of length l_uv + l_vw unless a witness search finds a path from u to w that avoids v and is no
    longer. A query searches upward only, from the source and from the target, along the edges and shortcuts that lead
    to a node of higher rank, and the distance is the least sum over the nodes both upward searches reach.

    Returns a dict: `rank`; `shortcuts` as (u, w, length, via, witness) with witness None when no other path exists;
    `checks`, every pair tried while contracting as (via node, u, w, length through via, witness length, added); and,
    given a source and a target, `up_forward` and `up_backward` (the distances the two upward searches settle),
    `meetings` as (node, sum) in order of the sum, and the `distance`.
    """
    H = _simple(G, weight)
    if H.is_directed():
        raise ValueError("contraction_hierarchy is written for undirected graphs")
    rank = {v: i for i, v in enumerate(order)}
    if set(rank) != set(H.nodes()):
        raise ValueError("the order must name every node once")
    rest = H.copy()                         # the nodes not yet contracted, with the shortcuts added so far
    up = H.copy()                           # every edge and every shortcut, for the queries
    shortcuts, checks = [], []
    for v in order:
        nbrs = sorted(rest[v], key=rank.get)
        without_v = nx.subgraph_view(rest, filter_node=lambda n, v=v: n != v)
        for i, u in enumerate(nbrs):
            for w in nbrs[i + 1:]:
                via = rest[u][v][weight] + rest[v][w][weight]
                witness = search(without_v, u, w, weight)["distance"]
                added = witness is None or witness > via
                checks.append((v, u, w, via, witness, added))
                if added:
                    rest.add_edge(u, w, **{weight: via})
                    up.add_edge(u, w, **{weight: via})
                    shortcuts.append((u, w, via, v, witness))
        rest.remove_node(v)
    out = {"rank": rank, "shortcuts": shortcuts, "checks": checks}
    if source is not None and target is not None:
        upward = nx.DiGraph()
        upward.add_nodes_from(up.nodes())
        for a, b, d in up.edges(data=True):
            lo, hi = (a, b) if rank[a] < rank[b] else (b, a)
            upward.add_edge(lo, hi, **{weight: d[weight]})
        fwd = search(upward, source, weight=weight)
        bwd = search(upward, target, weight=weight)
        f = {n: fwd["dist"][n] for n in fwd["settled"]}
        b = {n: bwd["dist"][n] for n in bwd["settled"]}
        meetings = sorted(((n, f[n] + b[n]) for n in f if n in b), key=lambda nm: (nm[1], rank[nm[0]]))
        out.update({"up_forward": f, "up_backward": b, "meetings": meetings,
                    "distance": meetings[0][1] if meetings else None})
    return out


def brandes_dependencies(G, source, weight="length"):
    """One source of Brandes' algorithm (2001), the way betweenness is computed for a whole network.

    A Dijkstra from `source` that also counts the shortest paths to every node, sigma_sv = sum of sigma_su over the
    predecessors u of v on shortest paths (two routes of equal length both count), followed by a pass over the nodes in
    the reverse order of settling that accumulates the dependency of the source on every node,
    delta_s(v) = sum over the nodes w with v among their predecessors of sigma_sv / sigma_sw (1 + delta_s(w)).
    The betweenness of v is the sum of delta_s(v) over every source s other than v, halved on an undirected graph where
    every pair is counted from both of its ends; n such passes cost O(nm + n^2 log n). Equal lengths are compared
    exactly, which suits lengths in whole metres.

    Returns a dict: `dist`, `sigma`, `pred` (the list of predecessors on shortest paths), `delta` (for every node
    reached other than the source) and the settling `order`.
    """
    tick = itertools.count()
    dist, sigma, preds = {source: 0.0}, {source: 1}, {source: []}
    heap = [(0.0, next(tick), source)]
    done, order = set(), []
    while heap:
        d, _, v = heapq.heappop(heap)
        if v in done:
            continue
        done.add(v)
        order.append(v)
        for w, l in _edges_out(G, v, weight):
            if w in done:
                continue
            nd = d + l
            old = dist.get(w, math.inf)
            if nd < old:
                dist[w], sigma[w], preds[w] = nd, sigma[v], [v]
                heapq.heappush(heap, (nd, next(tick), w))
            elif nd == old:
                sigma[w] += sigma[v]
                preds[w].append(v)
    delta = {v: 0.0 for v in order}
    for w in reversed(order):
        for v in preds[w]:
            delta[v] += sigma[v] / sigma[w] * (1.0 + delta[w])
    delta.pop(source)
    return {"dist": dist, "sigma": sigma, "pred": preds, "delta": delta, "order": order}


def all_pairs_cost(n, m):
    """The number of steps the two ways to every distance take, with every constant set to one: n runs of Dijkstra on a
    binary heap, n (n + m) log2 n, against the triple loop of Floyd-Warshall, n^3, which updates
    d_k(i, j) = min(d_(k-1)(i, j), d_(k-1)(i, k) + d_(k-1)(k, j)) for every k, i and j. Orders of magnitude to compare,
    not timings. Returns a dict with `dijkstra` and `floyd_warshall`."""
    n, m = float(n), float(m)
    return {"dijkstra": n * (n + m) * math.log2(n), "floyd_warshall": n ** 3}


# --------------------------------------------------------------------------- #
# accessibility
# --------------------------------------------------------------------------- #

def travel_time_matrix(G, origin_nodes, destination_nodes, weight="travel_time", cutoff_min=60.0):
    """Minutes from every origin node to every destination node along the network, one Dijkstra per origin with a
    cutoff, infinity beyond it; (origins, destinations) as an array."""
    import networkx as nx
    dest = list(destination_nodes)
    pos = {n: j for j, n in enumerate(dest)}
    out = np.full((len(origin_nodes), len(dest)), np.inf)
    for i, o in enumerate(origin_nodes):
        d = nx.single_source_dijkstra_path_length(G, o, cutoff=cutoff_min * 60.0, weight=weight)
        for n, t in d.items():
            j = pos.get(n)
            if j is not None:
                out[i, j] = min(out[i, j], t / 60.0)
    return out


def nearest_times(G, origin_nodes, destinations_by_category, weight="travel_time"):
    """Minutes from every origin node to the nearest destination of each category (one multi-source Dijkstra per
    category), as a frame indexed by origin node with one column per category; infinity when none is reached."""
    cols = {}
    for cat, nodes in destinations_by_category.items():
        d = times_from(G, nodes, weight=weight)
        cols[cat] = [d.get(o, np.inf) / 60.0 for o in origin_nodes]
    return pd.DataFrame(cols, index=pd.Index(list(origin_nodes), name="node"))


def cumulative(minutes, weights=None, threshold=15.0):
    """Cumulative opportunities: the (weighted) count of destinations within the threshold, per origin, from a
    (origins, destinations) matrix of minutes."""
    m = np.asarray(minutes)
    w = np.ones(m.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    return ((m <= threshold) * w).sum(axis=1)


def hansen(minutes, weights=None, beta=0.1):
    """Hansen's gravity-based accessibility A_i = sum_j w_j exp(minus beta t_ij) with t in minutes, per origin, from a
    (origins, destinations) matrix of minutes; unreachable pairs contribute nothing."""
    m = np.asarray(minutes, dtype=float)
    w = np.ones(m.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    f = np.where(np.isfinite(m), np.exp(-beta * np.where(np.isfinite(m), m, 0.0)), 0.0)
    return (f * w).sum(axis=1)


def share_within(minutes, population, threshold=15.0):
    """The share of the population whose nearest destination is within the threshold."""
    m, p = np.asarray(minutes, dtype=float), np.asarray(population, dtype=float)
    return float(p[m <= threshold].sum() / max(p.sum(), 1e-9))


def lorenz(values, population):
    """The Lorenz curve of a quantity across people: cumulative population share against cumulative share of the
    quantity, both sorted by the quantity per person; (x, y) arrays starting at 0."""
    v, p = np.asarray(values, dtype=float), np.asarray(population, dtype=float)
    order = np.argsort(v)
    cp = np.concatenate([[0.0], np.cumsum(p[order])]) / p.sum()
    cv = np.concatenate([[0.0], np.cumsum((v * p)[order])]) / max((v * p).sum(), 1e-9)
    return cp, cv


def gini(values, population):
    """The Gini coefficient of the Lorenz curve, 0 when everyone has the same, approaching 1 when one has it all."""
    x, y = lorenz(values, population)
    return float(1.0 - 2.0 * np.trapezoid(y, x))


# --------------------------------------------------------------------------- #
# traces
# --------------------------------------------------------------------------- #

def load_fixes(path, participant=None, t="t_s"):
    """A table of fixes (csv or csv.gz with t_s, x, y and whatever else), one participant if asked, sorted by time."""
    df = pd.read_csv(path)
    if participant is not None and "participant_id" in df.columns:
        df = df[df["participant_id"] == participant]
    return df.sort_values(t).reset_index(drop=True)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres between points given in degrees; arrays or scalars."""
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def step_lengths(x, y):
    """The distance between consecutive fixes in the units of x and y."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.hypot(np.diff(x), np.diff(y))


def stays(fixes, dist_m=50.0, min_s=600.0, x="x", y="y", t="t_s"):
    """The stay rule of Li and others 2008. From an anchor fix, the run extends while every later fix lies within
    `dist_m` of the anchor; when the run breaks, it is a stay if it spans at least `min_s`, with its centroid and its
    interval, and the search continues after it; otherwise the anchor moves one fix on. Returns a frame of x, y,
    start, end, duration_s, n, first, last (the fix indices)."""
    X, Y, T = (np.asarray(fixes[c], dtype=float) for c in (x, y, t))
    n = len(X)
    out = []
    i = 0
    while i < n - 1:
        j = i + 1
        while j < n and math.hypot(X[j] - X[i], Y[j] - Y[i]) <= dist_m:
            j += 1
        if T[j - 1] - T[i] >= min_s:
            out.append((X[i:j].mean(), Y[i:j].mean(), T[i], T[j - 1], T[j - 1] - T[i], j - i, i, j - 1))
            i = j
        else:
            i += 1
    return pd.DataFrame(out, columns=["x", "y", "start", "end", "duration_s", "n", "first", "last"])


def trips(fixes, stays_df, x="x", y="y", t="t_s"):
    """The segments between consecutive stays: walked length along the fixes, the straight line between the two
    stays' centroids, the duration, and directness as straight line over walked length (1 is a bee line)."""
    X, Y, T = (np.asarray(fixes[c], dtype=float) for c in (x, y, t))
    rows = []
    for k in range(len(stays_df) - 1):
        a, b = stays_df.iloc[k], stays_df.iloc[k + 1]
        i0, i1 = int(a["last"]), int(b["first"])
        walked = float(np.hypot(np.diff(X[i0:i1 + 1]), np.diff(Y[i0:i1 + 1])).sum())
        straight = float(math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
        rows.append((k, k + 1, T[i0], T[i1], T[i1] - T[i0], walked, straight, straight / walked if walked > 0 else np.nan, i0, i1))
    return pd.DataFrame(rows, columns=["from_stay", "to_stay", "start", "end", "duration_s", "walked_m", "straight_m", "directness", "first", "last"])


def match_stays(detected, coded, min_overlap_s=60.0, pois=None, dist_m=60.0):
    """Score detected stays against a coded diary of stops (start, end in the same seconds). A detected stay matches a
    coded stop when their intervals overlap by at least `min_overlap_s`, and, if the coded stop has a point in `pois`
    (a frame indexed by the stop's name with x and y), when the stay's centroid lies within `dist_m` of it. Matching is
    one to one, greedy by overlap. Returns precision, recall, the matched pairs and the unmatched of each side."""
    det = detected.reset_index(drop=True)
    cod = coded.dropna(subset=["start", "end"]).reset_index(drop=True)
    pairs = []
    for i, d in det.iterrows():
        for j, c in cod.iterrows():
            ov = min(d["end"], c["end"]) - max(d["start"], c["start"])
            if ov < min_overlap_s:
                continue
            if pois is not None and c.get("poi_name") in pois.index:
                p = pois.loc[c["poi_name"]]
                if math.hypot(d["x"] - p["x"], d["y"] - p["y"]) > dist_m:
                    continue
            pairs.append((ov, i, j))
    pairs.sort(reverse=True)
    used_d, used_c, matched = set(), set(), []
    for ov, i, j in pairs:
        if i in used_d or j in used_c:
            continue
        used_d.add(i); used_c.add(j); matched.append((i, j, ov))
    precision = len(matched) / len(det) if len(det) else float("nan")
    recall = len(matched) / len(cod) if len(cod) else float("nan")
    return {"precision": precision, "recall": recall, "matched": matched, "n_detected": len(det), "n_coded": len(cod),
            "unmatched_detected": sorted(set(range(len(det))) - used_d), "unmatched_coded": sorted(set(range(len(cod))) - used_c)}


def radius_of_gyration(x, y, weights=None):
    """The root mean square distance of the positions from their centre of mass (Gonzalez 2008)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    cx, cy = np.average(x, weights=w), np.average(y, weights=w)
    return float(math.sqrt(np.average((x - cx) ** 2 + (y - cy) ** 2, weights=w)))


def snap_to_edges(G, x, y, tolerance_m=10.0):
    """The nearest street segment of every fix and its distance, and the share of fixes within the tolerance; the
    simplest map matching, a nearest-edge snap (the hidden Markov method is the next step)."""
    import osmnx as ox
    edges, dist = ox.distance.nearest_edges(G, np.asarray(x, dtype=float), np.asarray(y, dtype=float), return_dist=True)
    dist = np.asarray(dist, dtype=float)
    return edges, dist, float((dist <= tolerance_m).mean())


# --------------------------------------------------------------------------- #
# flows
# --------------------------------------------------------------------------- #

def make_grid(bounds, size_m, crs=CRS):
    """Square cells of `size_m` over (minx, miny, maxx, maxy), as a GeoDataFrame with row, col and a flat id equal to
    the index; the origin is the lower-left corner of the bounds."""
    import geopandas as gpd
    from shapely.geometry import box
    minx, miny, maxx, maxy = bounds
    ncol, nrow = int(math.ceil((maxx - minx) / size_m)), int(math.ceil((maxy - miny) / size_m))
    rows, cols, geoms = [], [], []
    for r in range(nrow):
        for c in range(ncol):
            rows.append(r); cols.append(c)
            geoms.append(box(minx + c * size_m, miny + r * size_m, minx + (c + 1) * size_m, miny + (r + 1) * size_m))
    g = gpd.GeoDataFrame({"row": rows, "col": cols}, geometry=geoms, crs=crs)
    g.attrs.update({"bounds": tuple(map(float, bounds)), "size_m": float(size_m), "ncol": ncol, "nrow": nrow})
    return g


def cell_of(grid, x, y):
    """The flat cell id of every point in a grid made by `make_grid`, minus 1 outside it."""
    minx, miny, maxx, maxy = grid.attrs["bounds"]
    s, ncol, nrow = grid.attrs["size_m"], grid.attrs["ncol"], grid.attrs["nrow"]
    c = np.floor((np.asarray(x, dtype=float) - minx) / s).astype(int)
    r = np.floor((np.asarray(y, dtype=float) - miny) / s).astype(int)
    inside = (c >= 0) & (c < ncol) & (r >= 0) & (r < nrow)
    return np.where(inside, r * ncol + c, -1)


def od_matrix(origin_cells, destination_cells, n_cells):
    """The origin-destination counts T_ij as a dense (n_cells, n_cells) array; pairs with a cell of minus 1 are dropped."""
    o, d = np.asarray(origin_cells), np.asarray(destination_cells)
    keep = (o >= 0) & (d >= 0)
    return np.bincount(o[keep] * n_cells + d[keep], minlength=n_cells * n_cells).reshape(n_cells, n_cells)


def desire_lines(od, grid, min_n=1, include_internal=False):
    """Straight lines between cell centroids for every T_ij of at least `min_n`, with the count, for a flow map."""
    import geopandas as gpd
    from shapely.geometry import LineString
    cent = grid.geometry.centroid
    ii, jj = np.nonzero(od >= min_n)
    rows = []
    for i, j in zip(ii, jj):
        if i == j and not include_internal:
            continue
        rows.append((int(i), int(j), int(od[i, j]), LineString([(cent.iloc[i].x, cent.iloc[i].y), (cent.iloc[j].x, cent.iloc[j].y)])))
    out = gpd.GeoDataFrame(rows, columns=["o", "d", "n", "geometry"], crs=grid.crs)
    return out.sort_values("n").reset_index(drop=True)


def length_histogram(lengths, bins=30, xmin=None, xmax=None):
    """A histogram on logarithmic bins: the edges, the counts and the density per unit length."""
    v = np.asarray(lengths, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lo, hi = (xmin or v.min()), (xmax or v.max())
    edges = np.logspace(math.log10(lo), math.log10(hi), bins + 1)
    counts, _ = np.histogram(v, bins=edges)
    density = counts / (len(v) * np.diff(edges))
    return edges, counts, density


def fit_power_law(x, xmin):
    """The maximum-likelihood exponent of a power-law tail P(x) proportional to x to the minus alpha above xmin
    (Clauset, Shalizi and Newman 2009), with its standard error and the tail's size."""
    v = np.asarray(x, dtype=float)
    v = v[v >= xmin]
    n = len(v)
    alpha = 1.0 + n / np.log(v / xmin).sum()
    return {"alpha": float(alpha), "se": float((alpha - 1.0) / math.sqrt(n)), "n": int(n), "xmin": float(xmin)}


def gonzalez_pdf(r, beta=1.75, r0=1.5, kappa=400.0):
    """The displacement distribution of Gonzalez, Hidalgo and Barabasi 2008, (r + r0) to the minus beta times exp
    of minus r over kappa (r, r0 and kappa in km), normalised numerically over the given r."""
    r = np.asarray(r, dtype=float)
    f = (r + r0) ** (-beta) * np.exp(-r / kappa)
    return f / np.trapezoid(f, r)


def gravity_fit(flows, pop_o, pop_d, dist_km, decay="power"):
    """The gravity model T_ij = K m_i^a n_j^b f(d_ij) fitted as a Poisson regression on the logs (or on d itself for an
    exponential decay); returns the coefficients and the fitted flows. The distance coefficient is the exponent gamma
    (power) or the rate (exponential); the intercept is log K."""
    from sklearn.linear_model import PoissonRegressor
    T = np.asarray(flows, dtype=float)
    m, n, d = (np.asarray(a, dtype=float) for a in (pop_o, pop_d, dist_km))
    dcol = np.log(np.maximum(d, 0.1)) if decay == "power" else d
    X = np.column_stack([np.log(np.maximum(m, 1)), np.log(np.maximum(n, 1)), dcol])
    model = PoissonRegressor(alpha=0.0, fit_intercept=True, max_iter=2000, tol=1e-8).fit(X, T)
    pred = model.predict(X)
    a, b, g = model.coef_
    return {"intercept": float(model.intercept_), "a": float(a), "b": float(b), "gamma": float(-g), "decay": decay,
            "deviance": _deviance(T, pred), "n": int(len(T))}, pred


def _deviance(T, pred):
    """The Poisson deviance, the zero flows contributing their fitted value only (no log of zero evaluated)."""
    ratio = np.divide(T, pred, out=np.ones_like(T, dtype=float), where=T > 0)
    return float(2 * np.sum(T * np.log(ratio) - (T - pred)))


def radiation(pop, xy, outflow, finite_size=True):
    """The radiation model of Simini and others 2012 for every ordered pair of zones: T_ij = T_i m_i n_j / ((m_i + s_ij)
    (m_i + n_j + s_ij)), where s_ij is the population in the circle around i of radius d_ij excluding i and j, and T_i
    is the outflow of i (its commuters who leave the zone). No parameter is fitted. With `finite_size`, T_i is divided
    by 1 minus m_i over the total population, Simini's correction for a finite region. Returns the (n, n) matrix."""
    m = np.asarray(pop, dtype=float)
    P = np.asarray(xy, dtype=float)
    Ti = np.asarray(outflow, dtype=float)
    n = len(m)
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    T = np.zeros((n, n))
    M = m.sum()
    for i in range(n):
        order = np.argsort(D[i])
        cum = np.cumsum(m[order])                      # population within the circle, including i and the zone itself
        s = np.empty(n)
        s[order] = cum - m[order] - m[i]               # exclude the destination and the origin
        s[i] = 0.0
        s = np.maximum(s, 0.0)
        ti = Ti[i] / (1.0 - m[i] / M) if finite_size and M > m[i] else Ti[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            T[i] = ti * m[i] * m / ((m[i] + s) * (m[i] + m + s))
        T[i, i] = 0.0
    return T


def cpc(observed, predicted):
    """The common part of commuters, 2 sum min(T, T') over (sum T + sum T'), 1 when the two flow sets agree."""
    o, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    return float(2 * np.minimum(o, p).sum() / max(o.sum() + p.sum(), 1e-9))


# flows into one destination

def gravity_fit_destination(flows, residents, dist_km, decay="power", free_exponent=False, min_km=0.1):
    """The gravity model for one destination, T_i = K m_i f(d_i), fitted to the flows T_i from the origins i.

    m_i is the population of origin i (its resident workers) and d_i its distance to the destination in km; f(d) is
    d to the minus gamma for decay "power" or exp(minus beta d) for decay "exp". The fit is a Poisson regression of
    the rate T_i / m_i with m_i as the weights, which is the same as a Poisson regression of T_i with log m_i as an
    offset, so K matches the observed total and an origin without residents carries no weight. Distances under
    `min_km` are raised to it before the logarithm. With `free_exponent` the residents enter with an exponent of
    their own, T_i = K m_i^a f(d_i), fitted without the offset: an a close to 1 says the offset form is right.

    Returns the fit as a dict (decay, K, log_K, gamma or beta, a, deviance, the number of origins, the observed and
    predicted totals, and for an exponential decay the distance over which the rate halves) and the fitted flows.
    """
    from sklearn.linear_model import PoissonRegressor
    T = np.asarray(flows, dtype=float)
    m = np.asarray(residents, dtype=float)
    d = np.maximum(np.asarray(dist_km, dtype=float), min_km)
    if decay not in ("power", "exp"):
        raise ValueError(f"decay is 'power' or 'exp', not {decay!r}")
    x = np.log(d) if decay == "power" else d
    keep = m > 0
    model = PoissonRegressor(alpha=0.0, fit_intercept=True, max_iter=5000, tol=1e-10)
    if free_exponent:
        X = np.column_stack([np.log(m[keep]), x[keep]])
        model.fit(X, T[keep])
        a, g = (float(c) for c in model.coef_)
        pred = np.zeros_like(T)
        pred[keep] = model.predict(X)
    else:
        X = x[keep, None]
        model.fit(X, T[keep] / m[keep], sample_weight=m[keep])
        a, g = 1.0, float(model.coef_[0])
        pred = np.zeros_like(T)
        pred[keep] = m[keep] * model.predict(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(T > 0, T * np.log(np.where(T > 0, T, 1) / np.where(pred > 0, pred, 1)), 0.0) - (T - pred))
    fit = {"decay": decay, "K": float(math.exp(model.intercept_)), "log_K": float(model.intercept_), "a": a,
           "deviance": float(dev), "n": int(keep.sum()), "observed": float(T.sum()), "predicted": float(pred.sum())}
    if decay == "power":
        fit["gamma"] = -g
    else:
        fit["beta"] = -g
        fit["half_km"] = math.log(2) / -g if g < 0 else math.inf
    return fit, pred


def radiation_to(residents, xy, dest_xy, n_dest, outflow, finite_size=True, total=None, chunk=512):
    """The radiation model for one destination: the flow from every origin i into it, with nothing fitted,

        T_i = T_i^out m_i n / ((m_i + s_i)(m_i + n + s_i)),

    where m_i is the population of origin i (its resident workers), n the destination's (its jobs, or its resident
    workers), T_i^out the commuters who leave i, and s_i the population of the other origins that lie closer to i
    than the destination does, the circle around i that just reaches the destination, with i itself and the
    destination left out. The origins should cover the region around the circles, since a circle that reaches
    beyond them counts too little. With `finite_size`, T_i^out is divided by 1 minus m_i over the region's
    population `total` (by default the origins' population plus n), the correction of the model's authors for a
    finite region. Returns the flows and s, both one value per origin.
    """
    m = np.asarray(residents, dtype=float)
    P = np.asarray(xy, dtype=float)
    out = np.asarray(outflow, dtype=float)
    dx, dy = float(dest_xy[0]), float(dest_xy[1])
    n = float(n_dest)
    M = float(m.sum() + n) if total is None else float(total)
    r = np.hypot(P[:, 0] - dx, P[:, 1] - dy)                 # each origin's distance to the destination, its circle's radius
    s = np.empty(len(m))
    for a in range(0, len(m), chunk):
        b = min(a + chunk, len(m))
        D = np.hypot(P[a:b, None, 0] - P[None, :, 0], P[a:b, None, 1] - P[None, :, 1])
        inside = D < r[a:b, None]
        inside[np.arange(b - a), np.arange(a, b)] = False      # the origin itself is not in its own circle
        s[a:b] = np.dot(inside, m)       # np.dot rather than the matrix operator, which the kit's address check forbids
    ti = out / (1.0 - m / M) if finite_size else out
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(m > 0, ti * m * n / ((m + s) * (m + n + s)), 0.0)
    return T, s


def rate_by_band(dist_km, flows, residents, edges, per=1000.0):
    """A column's flow per `per` residents in distance bands.

    For every band from edges[k] up to edges[k + 1] (the last edge may be infinity) the number of origins, their
    flow and residents, the rate flow over residents times `per`, and the residents' mean distance, which is where
    a band sits on a distance axis. Returns a DataFrame with columns lo, hi, origins, flow, residents, rate, mean_km.
    """
    d = np.asarray(dist_km, dtype=float)
    T = np.asarray(flows, dtype=float)
    m = np.asarray(residents, dtype=float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (d >= lo) & (d < hi)
        res = float(m[k].sum())
        rows.append({"lo": float(lo), "hi": float(hi), "origins": int(k.sum()), "flow": float(T[k].sum()), "residents": res,
                     "rate": float(per * T[k].sum() / res) if res > 0 else float("nan"),
                     "mean_km": float(np.average(d[k], weights=m[k])) if res > 0 else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# privacy
# --------------------------------------------------------------------------- #

def uniqueness(user, cell, hour, p=4, draws=1, seed=SEED, min_points=None):
    """The share of users identified uniquely by `p` of their spatio-temporal points (de Montjoye and others 2013):
    for every user with at least `min_points` (default p) distinct (cell, hour) points, `p` are drawn at random, and the
    user is unique if no other user visited all of them. Repeated `draws` times; returns the mean share, the shares per
    draw and the number of users scored."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"u": np.asarray(user), "c": np.asarray(cell), "h": np.asarray(hour)}).drop_duplicates()
    df = df[df["c"] >= 0]
    pts = df.groupby("u").apply(lambda g: list(zip(g["c"], g["h"])), include_groups=False)
    need = p if min_points is None else min_points
    pts = pts[pts.map(len) >= need]
    index = {}
    for u, lst in pts.items():
        for key in lst:
            index.setdefault(key, set()).add(u)
    shares = []
    for _ in range(draws):
        unique = 0
        for u, lst in pts.items():
            pick = [lst[i] for i in rng.choice(len(lst), size=p, replace=False)]
            others = set.intersection(*(index[k] for k in pick))
            unique += int(others == {u})
        shares.append(unique / len(pts) if len(pts) else float("nan"))
    return {"share": float(np.mean(shares)), "shares": shares, "n_users": int(len(pts)), "p": p}


def aggregate(cell, hour, k=10, n_cells=None, n_hours=24):
    """Counts of points per (cell, hour) with cells under `k` suppressed, the way an aggregate release is published:
    returns the (n_cells, n_hours) count array, the suppressed mask, the share of non-empty cells suppressed and the
    share of points they held."""
    c, h = np.asarray(cell), np.asarray(hour)
    keep = c >= 0
    n_cells = int(c.max()) + 1 if n_cells is None else n_cells
    counts = np.bincount(c[keep] * n_hours + h[keep], minlength=n_cells * n_hours).reshape(n_cells, n_hours)
    nonempty = counts > 0
    suppressed = nonempty & (counts < k)
    return counts, suppressed, float(suppressed.sum() / max(nonempty.sum(), 1)), float(counts[suppressed].sum() / max(counts.sum(), 1))


def laplace_noise(count, epsilon=1.0, sensitivity=1.0, draws=100, seed=SEED):
    """The count released under the Laplace mechanism of differential privacy, `draws` times: the count plus noise of
    scale sensitivity over epsilon, so that one person's presence changes the answer's distribution by at most a
    factor exp(epsilon)."""
    rng = np.random.default_rng(seed)
    return count + rng.laplace(0.0, sensitivity / epsilon, size=draws)


def key_space(pattern):
    """The number of keys a pattern allows, with 9 a digit and A a letter (the 2014 New York taxi medallions were
    9A99, AA999 or AAA999, about eighteen million keys in all, enumerated and hashed in minutes)."""
    n = 1
    for ch in pattern:
        n *= 10 if ch == "9" else 26 if ch == "A" else 1
    return n


# the privacy chapter's mechanisms

_ALPHABET = {"9": "0123456789", "A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


def _points_by_user(user, cell, hour):
    """The distinct (cell, hour) points of every user, the cells below zero (outside every grid) dropped."""
    df = pd.DataFrame({"u": np.asarray(user), "c": np.asarray(cell), "h": np.asarray(hour)})
    return df[df["c"] >= 0].drop_duplicates()


def example_uniqueness(user, cell, hour, who=None, p=4, seed=SEED):
    """One person's unicity test written out point by point.

    Each fix of each user is a point, the cell and the hour of the day. For the user `who` (by default one of the users
    with the median number of distinct points, picked with the seed), `p` of their distinct points are drawn at random.
    The function counts the other users who visited each drawn point, orders the points from the most shared to the
    least, and counts the other users who visited the first one, the first two and so on up to all `p` together. The
    person is unique when nobody else visited all `p`, which is the test de Montjoye and others ran on a million and a
    half phones in 2013.

    Returns a dict with who, the number of their distinct points, the drawn points as (cell, hour) pairs, `sharers`
    (others per point), `together` (others sharing the first 1, 2, ..., p points) and `unique`.
    """
    df = _points_by_user(user, cell, hour)
    counts = df.groupby("u").size()
    rng = np.random.default_rng(seed)
    if who is None:
        median = int(np.median(counts.to_numpy()))
        candidates = np.sort(counts.index[counts == median].to_numpy())
        if len(candidates) == 0:                       # an even count whose median falls between two sizes
            candidates = np.sort(counts.index[counts >= median].to_numpy())
        who = candidates[rng.integers(len(candidates))]
    mine = df[df["u"] == who].sort_values(["c", "h"])
    assert len(mine) >= p, f"user {who} has {len(mine)} distinct points, fewer than p = {p}"
    pick = mine.iloc[np.sort(rng.choice(len(mine), size=p, replace=False))]
    cells, hours, users = df["c"].to_numpy(), df["h"].to_numpy(), df["u"].to_numpy()
    crowd = {}
    for c, h in zip(pick["c"], pick["h"]):
        crowd[(int(c), int(h))] = set(users[(cells == c) & (hours == h)]) - {who}
    points = sorted(crowd, key=lambda key: (-len(crowd[key]), key))      # the most shared point first
    sharers, together, still = [], [], None
    for key in points:
        sharers.append(len(crowd[key]))
        still = crowd[key] if still is None else still & crowd[key]
        together.append(len(still))
    return {"who": who.item() if hasattr(who, "item") else who, "n_points": int(len(mine)), "p": p, "points": points,
            "sharers": sharers, "together": together, "unique": together[-1] == 0, "users": int(counts.size)}


def anonymity_sets(keys, weights=None):
    """How many share each record's key: the anonymity set of a home and work pair, a night cell, any quasi-identifier.

    `keys` holds one key per record (a tuple, a string or a number); `weights` the people a record stands for (the
    workers of a block pair in a commuting table), one each by default. A key's set size is the number of people with
    that key, and a person is alone when their set size is one, which is when the key names them.

    Returns a dict: `sizes` (the set size of every record), `keys` (distinct keys), `people`, `alone_keys` (the share of
    keys held by one person), `alone_people` (the share of people alone in their key) and `median` (the median set size
    over people, the number Golle and Partridge report for home and work pairs).
    """
    k = pd.Series(list(keys)) if not isinstance(keys, (pd.Series, pd.Index, np.ndarray)) else pd.Series(np.asarray(keys))
    w = pd.Series(np.ones(len(k)) if weights is None else np.asarray(weights, dtype=float))
    per_key = w.groupby(k.to_numpy()).sum()
    sizes = k.map(per_key).to_numpy(dtype=float)
    order = np.argsort(sizes, kind="stable")
    cum = np.cumsum(w.to_numpy()[order])
    median = float(sizes[order][np.searchsorted(cum, cum[-1] / 2.0)]) if len(cum) else float("nan")
    return {"sizes": sizes, "keys": int(per_key.size), "people": float(w.sum()),
            "alone_keys": float((per_key == 1).mean()) if per_key.size else float("nan"),
            "alone_people": float(w[sizes == 1].sum() / w.sum()) if w.sum() else float("nan"), "median": median}


def distinct_per_cell(ids, cell, n_cells=None):
    """The number of distinct individuals that visited every cell, the count a heatmap of traces publishes.

    `ids` names the individual of every fix (a recording day, a taxi-day, a tracker), `cell` its cell (below zero
    outside the grid). A cell visited by one individual draws that individual's path, so the shares of cells with
    one and with fewer than k individuals say how much of a heatmap is a trace. Returns an integer array of length
    `n_cells` (the largest cell plus one by default) with zeros for cells nobody visited.
    """
    df = pd.DataFrame({"i": np.asarray(ids), "c": np.asarray(cell)})
    df = df[df["c"] >= 0].drop_duplicates()
    n = int(df["c"].max()) + 1 if n_cells is None else int(n_cells)
    return np.bincount(df["c"].to_numpy(dtype=np.int64), minlength=n)[:n] if len(df) else np.zeros(n, dtype=np.int64)


def hash_keys(patterns, algo="md5", example=None, released=None):
    """Enumerate every key the patterns allow, hash each one, and time the table.

    A pattern spells a key's format with 9 for a digit and A for a letter, as the 2014 New York taxi medallions were
    9A99, AA999 or AAA999. A pseudonym made by hashing such a key protects nothing, because the whole key space can be
    hashed once and every released digest looked up. The function does exactly that with hashlib: it hashes every key
    of every pattern, compares each digest with the released ones (by default the digest of `example`), and measures
    the seconds the table took on this machine.

    Returns a dict: `keys` per pattern and in all, `seconds`, `per_second`, `algo`, `example` with its digest, and
    `found`, the released digests mapped back to their keys.
    """
    targets = set()
    digest_of_example = None
    if example is not None:
        digest_of_example = hashlib.new(algo, example.encode()).hexdigest()
        targets.add(bytes.fromhex(digest_of_example))
    for d in released or ():
        targets.add(bytes.fromhex(d))
    sizes, found = {}, {}
    new = hashlib.new
    t0 = time.perf_counter()
    for pattern in patterns:
        n = 0
        for chars in itertools.product(*(_ALPHABET.get(ch, ch) for ch in pattern)):
            key = "".join(chars).encode()
            d = new(algo, key).digest()
            n += 1
            if d in targets:
                found[d.hex()] = key.decode()
        sizes[pattern] = n
    seconds = time.perf_counter() - t0
    total = int(sum(sizes.values()))
    return {"patterns": sizes, "keys": total, "seconds": seconds, "per_second": total / seconds if seconds else float("inf"),
            "algo": algo, "example": example, "example_digest": digest_of_example, "found": found}


def halving_factors(sizes, shares, dims=2):
    """How much uniqueness falls each time the resolution halves, between measurements at growing sizes.

    `sizes` are cell sides in metres (`dims` 2, so doubling a side halves the resolution twice, as the cell's area
    grows fourfold) or time windows in hours (`dims` 1); `shares` the uniqueness measured at each. Between two sizes the
    number of halvings is dims times log2 of their ratio, and the factor per halving is the ratio of the shares to the
    power one over that number. de Montjoye and others found the factor nearly constant, about two to the minus beta
    with beta = 0.157 - 0.007 p, a uniqueness 9.3 percent lower per halving at four points and 6.2 percent at ten.

    Returns a list of dicts with the two sizes, the halvings between them, the factor per halving and the percent drop.
    """
    s = np.asarray(sizes, dtype=float)
    e = np.asarray(shares, dtype=float)
    out = []
    for (s0, e0), (s1, e1) in zip(zip(s, e), zip(s[1:], e[1:])):
        halvings = dims * np.log2(s1 / s0)
        factor = float((e1 / e0) ** (1.0 / halvings)) if e0 > 0 and e1 > 0 else 0.0
        out.append({"from": float(s0), "to": float(s1), "halvings": float(halvings), "factor": factor, "drop_pct": 100.0 * (1.0 - factor)})
    return out


def recording_days(times, tz=LOCAL_TZ, ids=None):
    """The individual behind a public trace that names nobody: the local calendar day of every fix.

    A public trace is cut into tracks by the service, and a track is no person. The honest unit is the recording day, the
    fixes of one local date (the taxi-day of a taxi file), or with `ids`, an uploader code per fix, the uploader-day;
    fixes whose id is below zero (no uploader known) fall back to their date alone.

    Returns three things: the individual of every fix as an integer code, the local hour of the day of every fix, and
    the label of every code (the date, or uploader and date).
    """
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True, format="ISO8601")).tz_convert(tz)
    date = np.asarray(t.strftime("%Y-%m-%d"))
    if ids is None:
        label = pd.Series(date)
    else:
        u = np.asarray(ids)
        label = pd.Series(np.where(u >= 0, pd.Series(u).astype(str).to_numpy(), "-1")) + ":" + pd.Series(date)
    codes, labels = pd.factorize(label, sort=True)
    return codes.astype(np.int64), np.asarray(t.hour, dtype=np.int64), list(labels)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _edge_lines(G):
    from shapely.geometry import LineString
    for u, v, d in G.edges(data=True):
        geom = d.get("geometry")
        yield (geom if geom is not None else LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])), d


def show_graph(G, ax, colour="#666666", lw=0.3, width_by=None, colour_by=None, cmap="viridis", vmax=None, alpha=1.0, zorder=1):
    """Every edge of a street network as a line; `width_by` or `colour_by` name an edge attribute (or a dict keyed by
    (u, v)) that sets the width or the colour."""
    from matplotlib.collections import LineCollection
    import matplotlib as mpl
    segs, vals = [], []
    for geom, d in _edge_lines(G):
        segs.append(np.asarray(geom.coords))
        key = colour_by or width_by
        if key is not None:
            vals.append(float(d.get(key, 0.0)) if isinstance(key, str) else 0.0)
    kw = dict(linewidths=lw, colors=colour, alpha=alpha, zorder=zorder, capstyle="round")
    if colour_by is not None:
        vals = np.asarray(vals)
        norm = mpl.colors.Normalize(0, vmax or vals.max())
        kw.update(colors=None, array=vals, cmap=cmap, norm=norm)
    if width_by is not None:
        v = np.asarray(vals)
        kw["linewidths"] = lw + 2.0 * lw * v / max(v.max(), 1e-9) * 6
    lc = LineCollection(segs, **kw)
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return lc


def show_isochrones(polys, ax, colours, labels=None, alpha=0.75, zorder=2):
    """Isochrone polygons drawn largest first so the nearest budget sits on top."""
    import geopandas as gpd
    items = list(zip(polys, colours, labels or [None] * len(polys)))
    for poly, col, lab in reversed(items):
        if poly is None:
            continue
        gpd.GeoSeries([poly]).plot(ax=ax, color=col, alpha=alpha, edgecolor="none", zorder=zorder, label=lab)
    ax.set_aspect("equal")


def show_orientation(counts, ax, colour="#444444", title=None, fontsize=6):
    """A polar histogram of street bearings (the rose of Boeing 2019) on a polar axes, north up, clockwise."""
    counts = np.asarray(counts, dtype=float)
    bins = len(counts)
    width = 2 * np.pi / bins
    theta = np.arange(bins) * width
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(theta, counts / counts.max(), width=width, color=colour, edgecolor="white", linewidth=0.3, align="center")
    ax.set_yticks([]); ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=fontsize)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, pad=4)
    return ax


def show_access(grid, values, ax, cmap="YlGnBu_r", vmin=None, vmax=None, label=None, edge="none", legend=True):
    """A choropleth of a value per grid cell (minutes, counts, a score)."""
    g = grid.copy()
    g["v"] = np.asarray(values, dtype=float)
    g = g[np.isfinite(g["v"])]
    g.plot(column="v", ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor=edge, linewidth=0.1, legend=legend,
           legend_kwds={"label": label, "shrink": 0.6} if legend else None)
    ax.set_aspect("equal"); ax.set_axis_off()
    return ax


def show_flows(lines, ax, cmap="Oranges", lw_max=2.5, alpha=0.8, zorder=3):
    """Desire lines with width and colour by their count, the largest drawn last."""
    import matplotlib as mpl
    n = lines["n"].to_numpy(dtype=float)
    norm = mpl.colors.Normalize(n.min(), n.max())
    lines.plot(ax=ax, column="n", cmap=cmap, norm=norm, linewidth=0.2 + lw_max * n / n.max(), alpha=alpha, zorder=zorder)
    ax.set_aspect("equal")
    return ax


def show_traces(fixes, ax, colour="#999999", lw=0.4, stays_df=None, stay_colour="#e77500", x="x", y="y", zorder=2):
    """A trajectory as a thin line through its fixes, with the stays as circles scaled by their duration."""
    ax.plot(fixes[x], fixes[y], color=colour, lw=lw, zorder=zorder)
    if stays_df is not None and len(stays_df):
        s = 6 + 40 * stays_df["duration_s"] / stays_df["duration_s"].max()
        ax.scatter(stays_df["x"], stays_df["y"], s=s, facecolor=stay_colour, edgecolor="white", linewidth=0.4, zorder=zorder + 1)
    ax.set_aspect("equal")
    return ax


def show_od(matrix, ax, cmap="Oranges", log=True, labels=None, fontsize=6):
    """An origin-destination matrix as a heat map, origins down, destinations across."""
    import matplotlib as mpl
    m = np.asarray(matrix, dtype=float)
    im = ax.imshow(m, cmap=cmap, norm=mpl.colors.LogNorm(1, max(m.max(), 2)) if log else None, interpolation="nearest")
    if labels is not None:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=fontsize); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlabel("destination", fontsize=fontsize); ax.set_ylabel("origin", fontsize=fontsize)
    ax.grid(False)
    return im


def show_scatter_fit(observed, predicted, ax, colour="#DB5461", label=None, s=4, alpha=0.5):
    """Predicted against observed flows on logarithmic axes with the identity line."""
    o, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    keep = (o > 0) & (p > 0)
    ax.scatter(o[keep], p[keep], s=s, color=colour, alpha=alpha, linewidths=0, label=label)
    lo, hi = min(o[keep].min(), p[keep].min()), max(o[keep].max(), p[keep].max())
    ax.plot([lo, hi], [lo, hi], color="#000000", lw=0.5, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    return ax


def show_loglog(edges, density, ax, colour="#000000", label=None, marker="o", s=6):
    """A histogram on logarithmic bins drawn as points at the bins' geometric centres, on logarithmic axes."""
    centres = np.sqrt(edges[:-1] * edges[1:])
    d = np.asarray(density, dtype=float)
    keep = d > 0
    ax.scatter(centres[keep], d[keep], s=s, color=colour, marker=marker, linewidths=0, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    return ax


# --------------------------------------------------------------------------- #
# your own trace: reading, local time, projection, trimming, the diary, the aggregate release (the P07 kit)
# --------------------------------------------------------------------------- #

NO_TRACE = "no trace of your own in my_traces yet, so the specimen stands in; drop a GPX or CSV file there and run again"


def first_trace(folder, suffixes=(".gpx", ".csv")):
    """The first trace file in a folder by name (GPX or CSV, the suffix in any case), or None when the folder holds none;
    diary.csv is skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    found = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes and p.name.lower() != "diary.csv")
    return found[0] if found else None


def read_gpx(path):
    """The track points of a GPX file (1.0 or 1.1) as a frame of lon, lat, ele_m and time (UTC, NaT when absent), in
    file order; nothing else of the file is kept, so no name, link or author element travels with the points."""
    import xml.etree.ElementTree as ET
    rows = []
    for _, el in ET.iterparse(str(path)):
        if el.tag.rsplit("}", 1)[-1] == "trkpt":
            ele = time = None
            for child in el:
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "ele":
                    ele = child.text
                elif tag == "time":
                    time = child.text
            rows.append((float(el.get("lon")), float(el.get("lat")), None if ele is None else float(ele), time))
            el.clear()
    df = pd.DataFrame(rows, columns=["lon", "lat", "ele_m", "time"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="ISO8601")
    return df


def _column(cols, *names):
    """The first column whose lower-case name equals one of `names` or starts with one followed by a space or bracket."""
    for name in names:
        if name in cols:
            return cols[name]
    for name in names:
        for low, orig in cols.items():
            if low.startswith(name + " ") or low.startswith(name + "("):
                return orig
    return None


def _degrees(values, hemisphere=None):
    """Decimal degrees from numbers or strings such as 40.3487N or 74.6593 W, the sign taken from a trailing letter or
    from a separate hemisphere column (S and W negative); unreadable values become NaN."""
    text = pd.Series(values).astype(str).str.strip().str.upper()
    letter = text.str.extract(r"([NSEW])\s*$", expand=False)
    number = pd.to_numeric(text.str.replace(r"[NSEW]\s*$", "", regex=True).str.strip(), errors="coerce")
    if hemisphere is not None:
        letter = letter.fillna(pd.Series(hemisphere).astype(str).str.strip().str.upper().str[:1])
    sign = np.where(letter.isin(["S", "W"]).to_numpy(), -1.0, 1.0)
    return (number.abs().where(letter.notna(), number) * np.where(letter.notna(), sign, 1.0)).to_numpy(dtype=float)


def _times(values, column_name="", naive_tz=LOCAL_TZ):
    """Timestamps as UTC: values that carry a zone or a Z are read as they are; naive values are read as `naive_tz`
    (the hunt's local clock, as a diary is written) unless the column's name says UTC or GMT."""
    t = pd.to_datetime(pd.Series(values).astype(str).str.strip(), errors="coerce", format="mixed", utc=False)
    if getattr(t.dt, "tz", None) is not None:
        return t.dt.tz_convert("UTC")
    zone = "UTC" if any(w in column_name.lower() for w in ("utc", "gmt")) else naive_tz
    return t.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def read_trace(path, naive_tz=LOCAL_TZ):
    """A trace from a GPX file or from a CSV, as lon, lat, ele_m and time (UTC). A CSV may name its columns lat or
    latitude, lon, lng or longitude, and time, timestamp or datetime, or carry the date and the time in two columns; a
    coordinate may end in N, S, E or W or take its sign from an N/S or E/W column; ele, elevation or altitude are
    optional. Times without a zone are read as the local clock (`naive_tz`) unless their column says UTC. Fixes with
    an unreadable time or position stay in the frame; fixes_from_trace drops them."""
    path = Path(path)
    if path.suffix.lower() == ".gpx":
        return read_gpx(path)
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}
    lat = _column(cols, "lat", "latitude")
    lon = _column(cols, "lon", "lng", "long", "longitude")
    ns, ew = _column(cols, "n/s", "ns", "lat_ref"), _column(cols, "e/w", "ew", "lon_ref")
    t = _column(cols, "time", "timestamp", "datetime", "date time", "utc time", "local time")
    d = _column(cols, "date", "utc date", "local date")
    ele = _column(cols, "ele", "ele_m", "elevation", "altitude", "height")
    assert lat and lon and (t or d), f"{path.name} needs columns for latitude, longitude and time; it has {list(df.columns)}"
    holds_date = t is not None and df[t].astype(str).str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True).any()
    if t and d and t != d and not holds_date:
        stamp, name = df[d].astype(str) + " " + df[t].astype(str), f"{d} {t}"
    else:
        stamp, name = df[t or d], (t or d)
    return pd.DataFrame({"lon": _degrees(df[lon], df[ew] if ew else None), "lat": _degrees(df[lat], df[ns] if ns else None),
                         "ele_m": pd.to_numeric(df[ele], errors="coerce").to_numpy() if ele else np.nan,
                         "time": _times(stamp, name, naive_tz).to_numpy()})


def local_seconds(times, tz=LOCAL_TZ):
    """Clock times as seconds since local midnight in the given zone, the specimen's convention, which drops the date."""
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).tz_convert(tz)
    return (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6).to_numpy(dtype=float)


def project(lon, lat, crs=CRS):
    """Longitude and latitude to x and y in a projected CRS (metres), through pyproj."""
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = t.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x), np.asarray(y)


def clean_fixes(trace, max_km=50.0):
    """The fixes of a read trace that can be used: a time, a latitude within 90 and a longitude within 180 degrees, not
    the receiver's (0, 0) before it has a fix, and within `max_km` of the trace's median position (a jump to the
    other side of the world is a fault, not a walk); sorted by time, with the number of fixes dropped."""
    df = trace.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    ok = df["time"].notna() & df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
    ok &= ~((df["lat"].abs() < 1e-6) & (df["lon"].abs() < 1e-6))
    if ok.any():
        lat0, lon0 = float(df.loc[ok, "lat"].median()), float(df.loc[ok, "lon"].median())
        ok &= haversine_m(df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float), lon0, lat0) <= 1000.0 * max_km
    kept = df[ok].sort_values("time").reset_index(drop=True)
    return kept, int((~ok).sum())


def fixes_from_trace(trace, crs=CRS, tz=LOCAL_TZ):
    """A read trace (lon, lat, ele_m, time) as the table the stay rule reads: t_s in seconds since local midnight, x and
    y in the projected CRS, lon, lat and ele_m, sorted by time, the unusable fixes dropped (clean_fixes)."""
    df, _ = clean_fixes(trace)
    x, y = project(df["lon"], df["lat"], crs)
    return pd.DataFrame({"t_s": local_seconds(df["time"], tz), "lon": df["lon"].to_numpy(), "lat": df["lat"].to_numpy(),
                         "x": x, "y": y, "ele_m": df["ele_m"].to_numpy()})


def trim_ends(fixes, metres=200.0, x="x", y="y"):
    """The fixes with the first and last `metres` of walked path removed, the course's rule for a trace that starts and
    ends at a door; returns the trimmed frame and the number of fixes cut at the start and at the end."""
    X, Y = np.asarray(fixes[x], dtype=float), np.asarray(fixes[y], dtype=float)
    if len(X) < 2:
        return fixes.iloc[0:0].reset_index(drop=True), len(fixes), 0
    walked = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
    keep = (walked >= metres) & (walked <= walked[-1] - metres)
    if not keep.any():
        return fixes.iloc[0:0].reset_index(drop=True), len(fixes), 0
    first = int(np.argmax(keep))
    last = int(len(keep) - np.argmax(keep[::-1]))
    return fixes.iloc[first:last].reset_index(drop=True), first, len(keep) - last


def read_diary(path):
    """A diary of stops as a CSV with the columns stop, start and end (clock times such as 14:05 or 14:05:30), as the
    frame match_stays reads: poi_name, start and end in seconds since midnight, duration_min, sorted by start."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    need = [cols.get(k) for k in ("stop", "start", "end")]
    assert all(need), f"{Path(path).name} needs the columns stop, start and end; it has {list(df.columns)}"

    out = pd.DataFrame({"poi_name": df[need[0]].astype(str), "start": df[need[1]].map(parse_clock), "end": df[need[2]].map(parse_clock)})
    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] > out["start"]]
    out["duration_min"] = (out["end"] - out["start"]) / 60.0
    return out.sort_values("start").reset_index(drop=True)


def parse_clock(value):
    """A clock time as seconds since midnight: 14:05, 14:05:30, 14.05, 2:05 pm or 2pm; NaN when blank or unreadable."""
    text = str(value).strip().lower().replace(".", ":")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?(?::(\d{1,2}))?\s*([ap])?\.?m?\.?", text)
    if not m:
        return float("nan")
    h, mnt, sec, half = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0), m.group(4)
    if half == "p" and h < 12:
        h += 12
    elif half == "a" and h == 12:
        h = 0
    if h > 23 or mnt > 59 or sec > 59:
        return float("nan")
    return float(h * 3600 + mnt * 60 + sec)


def coded_stops(stops, participant):
    """One visitor's coded stops with both clock times, as the frame match_stays reads: start and end in seconds since
    midnight (from start_s and end_s), the untimed rows dropped, sorted by start with a fresh index."""
    c = stops[stops["participant_id"] == participant].rename(columns={"start_s": "start", "end_s": "end"})
    return c.dropna(subset=["start", "end"]).sort_values("start").reset_index(drop=True)


def threshold_surface(fixes, coded, dists_m, mins, min_overlap_s=60.0):
    """Precision, recall and the number of stays detected at every pair of a distance threshold (rows) and a time
    threshold in minutes (columns), the stay rule scored against a coded diary by time overlap; three arrays."""
    P = np.full((len(dists_m), len(mins)), np.nan)
    R = np.full((len(dists_m), len(mins)), np.nan)
    N = np.zeros((len(dists_m), len(mins)), dtype=int)
    for i, d in enumerate(dists_m):
        for j, m in enumerate(mins):
            found = stays(fixes, float(d), 60.0 * float(m))
            r = match_stays(found, coded, min_overlap_s=min_overlap_s)
            P[i, j], R[i, j], N[i, j] = r["precision"], r["recall"], len(found)
    return P, R, N


def cell_hours(grid, x, y, t_s):
    """The grid cell of every fix (make_grid's flat id, minus 1 outside the grid) and its hour of the day."""
    return cell_of(grid, x, y), (np.asarray(t_s, dtype=float) // 3600).astype(int) % 24


def grid_cells(cells, x, y):
    """The row position in a GeoDataFrame of cells (the P02 grid, which is not a full rectangle) that holds every point,
    minus 1 outside every cell, by a spatial join; the slower sibling of cell_of for a grid with holes."""
    import geopandas as gpd
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(x, dtype=float), np.asarray(y, dtype=float)), crs=cells.crs)
    cells_ = gpd.GeoDataFrame({"_pos": np.arange(len(cells))}, geometry=cells.geometry.to_numpy(), crs=cells.crs)
    joined = gpd.sjoin(pts, cells_, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["_pos"].fillna(-1).to_numpy(dtype=int)


def published_counts(cell, hour, k=5, n_cells=None, ids=None):
    """What an aggregate release holds: a table of cell, hour and count for every (cell, hour) with at least k points,
    the full count array, the share of non-empty cells suppressed and the share of points they held. With `ids`, the
    count is of distinct ids (trackers) per cell and hour rather than of points."""
    c, h = np.asarray(cell), np.asarray(hour)
    if ids is not None:
        df = pd.DataFrame({"i": np.asarray(ids), "c": c, "h": h}).drop_duplicates()
        c, h = df["c"].to_numpy(), df["h"].to_numpy()
    counts, suppressed, share_cells, share_points = aggregate(c, h, k=k, n_cells=n_cells)
    rows, cols = np.nonzero(counts >= k)
    table = pd.DataFrame({"cell": rows, "hour": cols, "count": counts[rows, cols]}).sort_values(["hour", "cell"]).reset_index(drop=True)
    return table, counts, share_cells, share_points


def clock(seconds):
    """Seconds since midnight as hh:mm, or --:-- when there is no time."""
    if seconds is None or not np.isfinite(float(seconds)):
        return "--:--"
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def next_departure(departures, leave_min, access_min, ride_min):
    """The next train for a traveller who leaves the door at `leave_min` (minutes after midnight, one value or many)
    and can board `access_min` later (the walk to the platform and the router's minute on it): the departure taken,
    the wait on the platform beyond the access, the arrival `ride_min` after the departure and the door to door
    minutes. A schedule is a list of departures in minutes after midnight; NaN where no train leaves later."""
    deps = np.sort(np.asarray(departures, dtype=float))
    t = np.atleast_1d(np.asarray(leave_min, dtype=float))
    i = np.searchsorted(deps, t + float(access_min) - 1e-9)
    train = np.where(i < len(deps), deps[np.minimum(i, len(deps) - 1)], np.nan)
    arrive = train + float(ride_min)
    return pd.DataFrame({"leave": t, "train": train, "wait": train - t - float(access_min), "arrive": arrive, "total": arrive - t})


def edge_names(G, edges):
    """The edge frame of centrality() joined to the street names of the graph and the coordinates of both ends, so that
    a street can be ranked by its most central segment and the segments drawn without the graph."""
    names = {}
    for u, v, d in G.edges(data=True):
        n = d.get("name")
        n = n[0] if isinstance(n, list) else n
        if isinstance(n, str) and n and frozenset((u, v)) not in names:
            names[frozenset((u, v))] = n
    out = edges.copy()
    out["name"] = [names.get(frozenset((u, v))) for u, v in zip(out["u"], out["v"])]
    for c in ("u", "v"):
        out[f"{c}x"] = [G.nodes[n]["x"] for n in out[c]]
        out[f"{c}y"] = [G.nodes[n]["y"] for n in out[c]]
    return out


def top_streets(edges, n=8):
    """The n streets with the most central segment, from an edge frame with a name column, as (name, betweenness) pairs."""
    named = edges.dropna(subset=["name"]).groupby("name")["betweenness"].max().sort_values(ascending=False)
    return [(str(k), float(v)) for k, v in named.head(n).items()]


def show_edges(edges, ax, colour_by="betweenness", cmap="viridis", lw=0.4, vmax=None, width=True, zorder=2):
    """Straight segments from an edge table with ux, uy, vx and vy, coloured and optionally widened by a column."""
    from matplotlib.collections import LineCollection
    import matplotlib as mpl
    segs = [[(a, b), (c, d)] for a, b, c, d in zip(edges["ux"], edges["uy"], edges["vx"], edges["vy"])]
    v = np.asarray(edges[colour_by], dtype=float)
    norm = mpl.colors.Normalize(0, vmax or v.max())
    lws = lw + (6 * lw * v / max(v.max(), 1e-9) if width else 0.0)
    lc = LineCollection(segs, array=v, cmap=cmap, norm=norm, linewidths=lws, zorder=zorder, capstyle="round")
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    return lc


def block_group_means(geoid, population, values):
    """A cell quantity summed and averaged to the block groups: a frame indexed by GEOID with the population and the
    population-weighted mean of the values over the cells of each block group (cells outside every block group dropped)."""
    df = pd.DataFrame({"GEOID": np.asarray(geoid, dtype=object), "population": np.asarray(population, dtype=float),
                       "weighted": np.asarray(values, dtype=float) * np.asarray(population, dtype=float)}).dropna(subset=["GEOID"])
    out = df.groupby("GEOID")[["population", "weighted"]].sum()
    out["mean"] = out["weighted"] / out["population"].replace(0, np.nan)
    return out.drop(columns="weighted")


def show_lorenz(values, population, ax, colour="#e77500", label=None):
    """The Lorenz curve of a quantity across people beside the line of equality, with the Gini in the legend."""
    x, y = lorenz(values, population)
    g = gini(values, population)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8)
    ax.plot(x, y, color=colour, lw=1.5, label=f"{label + ', ' if label else ''}Gini {g:.2f}")
    ax.set_xlabel("share of residents, least accessible first")
    ax.set_ylabel("share of accessibility")
    ax.set_aspect("equal")
    ax.legend(frameon=False)
    return g
