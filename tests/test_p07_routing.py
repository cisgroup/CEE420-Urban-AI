"""The routing tools of P07 find what the textbook algorithms find, and the counts Lecture 07 prints come back from the
kit's own street networks.

Every search is checked against networkx on random graphs (directed and undirected, with ties), the toy of seven places
gives the numbers worked by hand in the lecture, and the walk and drive networks the kit ships reproduce the deck's
counts for Nassau Hall to the Dinky station and to Princeton Junction. The tools live in the graphs section of
`P07/mob_tools.py`.
"""

import itertools
import math
from importlib.util import module_from_spec, spec_from_file_location

import networkx as nx
import numpy as np
import pytest

from conftest import P07_DATA, P07_KIT

TOY_EDGES = [("A", "B", 2), ("A", "C", 3), ("B", "C", 2), ("B", "D", 3), ("C", "E", 2), ("D", "E", 2), ("D", "F", 3),
             ("E", "G", 4), ("F", "G", 2)]
NASSAU_HALL = (528936.0, 4466524.0)


@pytest.fixture(scope="module")
def rt(mob):
    """The routing tools, which live in the toolbox's graphs section."""
    return mob


@pytest.fixture(scope="module")
def toy():
    H = nx.Graph()
    for u, v, w in TOY_EDGES:
        H.add_edge(u, v, length=100.0 * w)
    return H


def _random_graph(seed, n=30, m=70, directed=False, integer=False):
    """A connected-ish random graph with lengths; integer lengths make ties between paths common."""
    rng = np.random.default_rng(seed)
    G = nx.gnm_random_graph(n, m, seed=int(seed), directed=directed)
    for u, v in G.edges():
        G[u][v]["length"] = float(rng.integers(1, 6)) if integer else float(rng.uniform(1.0, 10.0))
    return G


def _geometric_graph(seed, n=60, radius=0.25):
    """Points in the unit square joined within a radius, each edge at least as long as its chord, as streets are."""
    rng = np.random.default_rng(seed)
    G = nx.random_geometric_graph(n, radius, seed=int(seed))
    for v, d in G.nodes(data=True):
        d["x"], d["y"] = float(d["pos"][0]), float(d["pos"][1])
    for u, v in G.edges():
        chord = math.hypot(G.nodes[u]["x"] - G.nodes[v]["x"], G.nodes[u]["y"] - G.nodes[v]["y"])
        G[u][v]["length"] = chord * float(rng.uniform(1.0, 1.6))
    return G


# --------------------------------------------------------------------------- #
# the toy of seven places, worked by hand in the lecture
# --------------------------------------------------------------------------- #

def test_toy_rounds_match_the_hand_count(rt, toy):
    steps = rt.dijkstra_steps(toy, "A")
    assert steps["order"] == list("ABCDEFG")
    assert [steps["dist"][n] for n in "ABCDEFG"] == [0, 200, 300, 500, 500, 800, 900]
    assert [steps["pred"][n] for n in "BCDEFG"] == list("AABCDE")
    assert steps["checks"] == 18 and steps["improvements"] == 6          # every edge from both ends; six improvements
    assert [len(r["improved"]) for r in steps["rounds"]] == [2, 1, 1, 1, 1, 0, 0]
    assert steps["rounds"][5]["relaxed"] == [("F", "G", 900.0, 1000.0)]  # through F, G would be 1,000


def test_toy_paths_hops_and_alternative(rt, toy):
    paths = rt.k_shortest_paths(toy, "A", "G", k=20)
    assert [p["length"] for p in paths] == [900, 1000, 1000, 1100, 1200, 1300, 1300, 1400]   # the eight simple paths
    assert paths[0]["path"] == list("ACEG") and paths[0]["shared"] == 1.0
    few = rt.fewest_edges_path(toy, "A", "G")
    assert few["edges"] == 3 and few["length"] == 900
    assert few["hops"] == {"A": 0, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3, "G": 3}
    alt = rt.dissimilar_alternative(toy, list("ACEG"))
    assert alt["path"] == list("ABDFG") and alt["length"] == 1000 and alt["shared"] == 0.0


def test_toy_contraction_adds_one_shortcut(rt, toy):
    ch = rt.contraction_hierarchy(toy, list("AFGBCDE"), source="A", target="G")
    assert [(frozenset((u, w)), length, via, witness) for u, w, length, via, witness in ch["shortcuts"]] == \
        [(frozenset("DG"), 500.0, "F", 600.0)]
    assert ch["meetings"] == [("E", 900.0), ("D", 1000.0)] and ch["distance"] == 900.0
    assert ch["up_forward"] == {"A": 0, "B": 200, "C": 300, "D": 500, "E": 500}
    assert ch["up_backward"] == {"G": 0, "E": 400, "D": 500}


def test_toy_dependencies_on_D(rt, toy):
    deps = {s: rt.brandes_dependencies(toy, s)["delta"].get("D", 0.0) for s in toy if s != "D"}
    assert deps == {"A": 1.0, "B": 1.5, "C": 1.0, "E": 1.0, "F": 4.0, "G": 0.5}
    assert sum(deps.values()) / 2 == 4.5
    tree = rt.brandes_dependencies(toy, "B")
    assert tree["sigma"]["G"] == 2 and sorted(tree["pred"]["G"]) == ["E", "F"]


def test_toy_bidirectional_meets(rt, toy):
    res = rt.bidirectional_search(toy, "A", "G")
    assert res["distance"] == 900 and res["path"] == list("ACEG")
    assert res["met"] in toy and res["forward"][0] == "A" and res["backward"][0] == "G"


# --------------------------------------------------------------------------- #
# against networkx on random graphs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("directed", [False, True])
def test_search_equals_networkx(rt, seed, directed):
    G = _random_graph(seed, directed=directed)
    truth = nx.single_source_dijkstra_path_length(G, 0, weight="length")
    tree = rt.search(G, 0)
    assert set(tree["settled"]) == set(truth)
    for v, d in truth.items():
        assert tree["dist"][v] == pytest.approx(d)
    for t in list(truth)[1:8]:
        one = rt.search(G, 0, t)
        assert one["distance"] == pytest.approx(truth[t])
        assert nx.path_weight(G, one["path"], "length") == pytest.approx(truth[t])
        assert one["settled"][-1] == t and one["pushes"] >= len(one["settled"])
    missing = [v for v in G if v not in truth]
    if missing:
        assert rt.search(G, 0, missing[0])["distance"] is None


@pytest.mark.parametrize("seed", range(6))
def test_dijkstra_steps_equals_the_heap(rt, seed):
    G = _random_graph(seed, n=15, m=30, integer=True)
    steps = rt.dijkstra_steps(G, 0)
    heap = rt.search(G, 0)
    assert steps["dist"] == pytest.approx({v: heap["dist"][v] for v in heap["settled"]})
    if nx.is_connected(G):
        assert steps["checks"] == 2 * G.number_of_edges()


@pytest.mark.parametrize("seed", range(6))
def test_astar_is_exact_and_searches_less(rt, seed):
    G = _geometric_graph(seed)
    comp = max(nx.connected_components(G), key=len)
    nodes = sorted(comp)
    for s, t in [(nodes[0], nodes[-1]), (nodes[1], nodes[len(nodes) // 2])]:
        h = rt.straight_line_heuristic(G, t)
        assert h.short_edges == 0
        a = rt.search(G, s, t, heuristic=h)
        d = rt.search(G, s, t)
        assert a["distance"] == pytest.approx(d["distance"])
        assert a["reopened"] == 0 and len(a["settled"]) <= len(d["settled"])


def test_astar_reopens_under_an_inconsistent_heuristic(rt):
    G = nx.DiGraph()
    G.add_weighted_edges_from([("S", "A", 1), ("S", "B", 4), ("A", "B", 1), ("B", "T", 5)], weight="length")
    h = {"S": 0.0, "A": 5.0, "B": 0.0, "T": 0.0}           # admissible (A is 6 from T) but not consistent
    res = rt.search(G, "S", "T", heuristic=h.get)
    assert res["distance"] == 7.0 and res["path"] == ["S", "A", "B", "T"] and res["reopened"] == 1


def test_heuristic_counts_short_edges_and_speed(rt):
    G = nx.Graph()
    G.add_node(0, x=0.0, y=0.0); G.add_node(1, x=3.0, y=4.0); G.add_node(2, x=6.0, y=8.0)
    G.add_edge(0, 1, length=5.0); G.add_edge(1, 2, length=4.5)        # the second edge is 0.5 m shorter than its chord
    h = rt.straight_line_heuristic(G, 2)
    assert h(0) == pytest.approx(10.0) and h.short_edges == 1 and h.short_by_m == pytest.approx(0.5)
    hs = rt.straight_line_heuristic(G, 2, speed_kph=36.0)                 # 10 m at 10 m/s is one second
    assert hs(0) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("directed", [False, True])
def test_bidirectional_equals_dijkstra(rt, seed, directed):
    G = _random_graph(seed + 10, directed=directed, integer=bool(seed % 2))
    truth = nx.single_source_dijkstra_path_length(G, 0, weight="length")
    for t in G:
        if t == 0:
            continue
        res = rt.bidirectional_search(G, 0, t)
        if t in truth:
            assert res["distance"] == pytest.approx(truth[t])
            assert res["path"][0] == 0 and res["path"][-1] == t
            assert nx.path_weight(G, res["path"], "length") == pytest.approx(truth[t])
        else:
            assert res["distance"] is None


@pytest.mark.parametrize("seed", range(6))
def test_fewest_edges_is_the_shortest_of_the_fewest(rt, seed):
    G = _random_graph(seed + 20, n=20, m=40, integer=True)
    hops = nx.single_source_shortest_path_length(G, 0)
    for t in list(hops)[1:10]:
        res = rt.fewest_edges_path(G, 0, t)
        assert res["edges"] == hops[t] and res["hops"] == hops
        best = min(nx.path_weight(G, p, "length") for p in nx.all_shortest_paths(G, 0, t))
        assert res["length"] == pytest.approx(best)


@pytest.mark.parametrize("seed", range(4))
def test_k_shortest_paths_in_order(rt, seed):
    G = _random_graph(seed + 30, n=20, m=45)
    t = max(nx.single_source_shortest_path_length(G, 0), key=lambda v: (nx.dijkstra_path_length(G, 0, v, weight="length"), v))
    paths = rt.k_shortest_paths(G, 0, t, k=10)
    lengths = [p["length"] for p in paths]
    assert lengths == sorted(lengths) and lengths[0] == pytest.approx(nx.dijkstra_path_length(G, 0, t, weight="length"))
    assert len({tuple(p["path"]) for p in paths}) == len(paths)
    assert all(0.0 <= p["shared"] <= 1.0 for p in paths) and paths[0]["shared"] == 1.0
    alt = rt.dissimilar_alternative(G, paths[0]["path"])
    if alt["path"] is not None:
        first = {frozenset(e) for e in zip(paths[0]["path"], paths[0]["path"][1:])}
        assert not first & {frozenset(e) for e in zip(alt["path"], alt["path"][1:])}
        assert alt["length"] >= lengths[0]


@pytest.mark.parametrize("seed", range(6))
def test_contraction_queries_are_exact(rt, seed):
    G = _random_graph(seed + 40, n=12, m=22, integer=True)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    order = list(np.random.default_rng(seed).permutation(list(G.nodes())))
    truth = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    for s, t in itertools.combinations(sorted(G), 2):
        ch = rt.contraction_hierarchy(G, order, source=s, target=t)
        assert ch["distance"] == pytest.approx(truth[s][t])
    for u, w, length, via, witness in ch["shortcuts"]:
        assert witness is None or witness > length


@pytest.mark.parametrize("seed", range(6))
def test_brandes_sums_to_betweenness(rt, seed):
    G = _random_graph(seed + 50, n=18, m=35, integer=True)
    truth = nx.betweenness_centrality(G, weight="length", normalized=False)
    ours = {v: 0.0 for v in G}
    for s in G:
        for v, d in rt.brandes_dependencies(G, s)["delta"].items():
            ours[v] += d
    for v in G:
        assert ours[v] / 2 == pytest.approx(truth[v])


def test_all_pairs_cost_formula(rt):
    c = rt.all_pairs_cost(4, 3)
    assert c["dijkstra"] == pytest.approx(4 * 7 * 2) and c["floyd_warshall"] == 64


# --------------------------------------------------------------------------- #
# the deck's counts on the kit's own networks
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def stations():
    import geopandas as gpd
    path = P07_DATA / "layers.gpkg"
    if not path.exists():
        pytest.skip("the kit's layers are not published yet")
    d = gpd.read_file(path, layer="destinations")
    st = d[d["category"] == "station"]
    return {name: (float(g.x), float(g.y)) for name, g in zip(st["name"], st.geometry)}


@pytest.fixture(scope="module")
def walk(mob):
    path = P07_DATA / "princeton_walk.graphml"
    if not path.exists():
        pytest.skip("the kit's walk network is not published yet")
    return mob.load_graph(path)


def test_nassau_hall_to_the_dinky(rt, mob, walk, stations):
    s = mob.nearest_nodes(walk, *NASSAU_HALL)
    t = mob.nearest_nodes(walk, *stations["Princeton"])
    dij = rt.search(walk, s, t)
    assert round(dij["distance"]) == 854 and len(dij["path"]) - 1 == 32
    assert (len(dij["settled"]), dij["pushes"]) == (1625, 2051)
    h = rt.straight_line_heuristic(walk, t)
    ast = rt.search(walk, s, t, heuristic=h)
    assert ast["distance"] == pytest.approx(dij["distance"]) and (len(ast["settled"]), ast["pushes"]) == (290, 420)
    assert (h.short_edges, round(h.short_by_m, 2)) == (4640, 0.98)
    bi = rt.bidirectional_search(walk, s, t)
    assert bi["distance"] == pytest.approx(dij["distance"]) and (len(bi["forward"]), len(bi["backward"])) == (404, 404)
    few = rt.fewest_edges_path(walk, s, t)
    assert (few["edges"], round(few["length"])) == (25, 1013)
    yen = rt.k_shortest_paths(walk, s, t, k=50)
    assert yen[-1]["length"] - yen[0]["length"] < 10 and sum(p["shared"] <= 0.8 for p in yen) == 8
    assert round(rt.dissimilar_alternative(walk, dij["path"])["length"]) == 945


def test_nassau_hall_to_the_junction_by_car(rt, mob, stations):
    path = P07_DATA / "princeton_drive.graphml"
    if not path.exists():
        pytest.skip("the kit's drive network is not published yet")
    drive = mob.load_graph(path)
    s = mob.nearest_nodes(drive, *NASSAU_HALL)
    t = mob.nearest_nodes(drive, *stations["Princeton Junction"])
    dij = rt.search(drive, s, t)
    ast = rt.search(drive, s, t, heuristic=rt.straight_line_heuristic(drive, t))
    assert ast["distance"] == pytest.approx(dij["distance"])
    assert (len(dij["settled"]), len(ast["settled"]), drive.number_of_nodes()) == (642, 95, 932)
