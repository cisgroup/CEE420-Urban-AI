"""Every published P08 file is what the notebooks assume it is, and the lessons they teach still hold.

The kit ships the seven places of Lecture 07 as Lecture 08 reads them, the 1,219 street segments of Princeton with the
street graph and the proximity graph, their lines for the maps, the lecture's one run in full and its repeated runs,
and METR-LA's speeds with the sensors' distances. These tests hold the pipeline to its contract: every file carries its
provenance and licence, nothing that identifies a person ships, the published ground truth carries no answers, and the
published numbers come back from the published files with the kit's own toolbox. The toy and the forecasts are NumPy
and must come back exactly; a training run may keep another epoch on another machine, so its accuracy must come back
within three points and the models must keep their order.
"""

import json
import re

import numpy as np
import pytest

from conftest import P08_DATA, P08_KIT

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator|contributor|username|uploader", re.I)
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\||api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9]{8,}|CENSUS_API_KEY=\S", re.I)
FILES = ["scene.json", "ground_truth.json", "toy.json", "streets.npz", "streets.gpkg", "lecture_main.npz", "lecture_runs.json", "metr_speeds.npz",
         "metr_graph.npz"]
TOLERANCE = 0.03            # a trained accuracy on another machine, which may keep another epoch
GRAPHS = ("none", "proximity", "streets")


@pytest.fixture(scope="module")
def streets(gt08):
    return gt08.load_arrays(P08_DATA / "streets.npz")


@pytest.fixture(scope="module")
def metr(gt08, scene08):
    """METR-LA as the forecaster sees it: the readings, the gap-filled readings, the neighbours' mean, the clock, the origins."""
    mr = scene08["rules"]["metr"]
    v = gt08.load_arrays(P08_DATA / "metr_speeds.npz")
    g = gt08.load_arrays(P08_DATA / "metr_graph.npz")
    speeds = v["speeds"].astype(float)
    (train, _, _), origins = gt08.forecast_origins(len(speeds), mr["lags"], max(mr["horizons"].values()), mr["train"], mr["val"])
    start = np.array([speeds[train, i][speeds[train, i] > 0].mean() for i in range(speeds.shape[1])])
    Xf = gt08.fill_missing(speeds, start=start)
    W = gt08.kernel_adjacency(g["road_dist"].astype(float), mr["threshold"])
    slots = gt08.time_of_week(v["seconds"])
    HA = gt08.historical_average(speeds, slots, train)
    return dict(speeds=speeds, Xf=Xf, W=W, NB=gt08.neighbour_average(Xf, gt08.symmetric(W)), slots=slots, HA=HA, clock=HA[slots], origins=origins,
                line_W=g["line_W"].astype(float))


@pytest.mark.parametrize("name", FILES)
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries every file it opens."""
    assert (P08_DATA / name).exists()


def test_the_kit_stays_within_its_budget():
    total = sum(p.stat().st_size for p in P08_DATA.iterdir() if p.is_file())
    assert total < 40_000_000, f"the kit weighs {total / 1e6:.1f} MB; thirty laptops pull it"
    largest = max(P08_DATA.iterdir(), key=lambda p: p.stat().st_size)
    assert largest.stat().st_size < 25_000_000, f"{largest.name} is larger than the kit needs"


def test_every_file_carries_its_provenance(scene08):
    import pyogrio

    for path in sorted(P08_DATA.glob("*.npz")):
        keys = set(np.load(path).files)
        assert {"source", "licence", "retrieved"} <= keys, f"{path.name} lacks provenance arrays: {sorted(keys)}"
    for path in sorted(P08_DATA.glob("*.json")):
        if path.name == "ground_truth.json":
            continue
        meta = json.loads(path.read_text())
        assert {"source", "licence", "retrieved"} <= set(meta), f"{path.name} lacks provenance keys"
    layers = {name for name, _ in pyogrio.list_layers(P08_DATA / "streets.gpkg")}
    assert layers == {"segments"}, f"streets.gpkg holds {sorted(layers)}"
    meta = pyogrio.read_info(P08_DATA / "streets.gpkg", layer="segments").get("layer_metadata") or {}
    assert {"source", "licence", "retrieved"} <= set(meta) and "OpenStreetMap" in meta["licence"], f"streets.gpkg lacks provenance: {sorted(meta)}"
    for name in FILES:
        if name != "ground_truth.json":
            assert name in scene08["files"], f"{name} has no provenance row in scene.json"
    assert "ODbL" in scene08["files"]["streets.npz"]["licence"] and "Google" in scene08["files"]["streets.npz"]["licence"]
    assert "Li" in scene08["files"]["metr_speeds.npz"]["licence"] and "PeMS" in scene08["files"]["metr_speeds.npz"]["licence"]


def test_nothing_that_identifies_a_person_or_a_key_ships():
    import geopandas as gpd

    seg = gpd.read_file(P08_DATA / "streets.gpkg")
    bad = [c for c in seg.columns if PII.search(c)]
    assert not bad, f"streets.gpkg carries {bad}"
    assert "@" not in seg.drop(columns="geometry").to_csv(index=False), "streets.gpkg carries an address"
    for path in sorted(P08_DATA.glob("*.npz")):
        z = np.load(path)
        bad = [k for k in z.files if PII.search(k)]
        assert not bad, f"{path.name} carries {bad}"
        for k in z.files:
            if z[k].dtype.kind == "U":
                assert not any("@" in str(v) for v in np.atleast_1d(z[k])), f"{path.name}:{k} carries an address"
    for path in list(P08_DATA.glob("*.json")) + [P08_KIT / "graph_tools.py"] + sorted(P08_KIT.glob("*.ipynb")):
        text = path.read_text()
        assert not SECRETS.search(text), f"{path.name} carries something that looks like a token or a key"
        assert "/Users" not in text, f"{path.name} mentions a home folder"
        assert "@" not in text, f"{path.name} carries an @, which the kit never prints"


def test_published_p08_ground_truth_carries_no_answers():
    text = (P08_DATA / "ground_truth.json").read_text()
    published = json.loads(text)
    assert set(published) == {"generated", "p08"}, f"the published P08 ground truth carries {sorted(published)}"
    for key in ("p08_answers", "from_rounds", "from_layers", "from_split", "from_horizon", "lecture_depth"):
        assert key not in text, f"the published ground truth leaks the Friday answer {key}"


def test_the_toy_comes_back_exactly(gt08, p08):
    toy = gt08.load_toy(P08_DATA / "toy.json")
    t = p08["toy"]
    A, x = toy["A"], toy["x"]
    assert np.dot(A, x).tolist() == t["Ax"]
    assert [v.tolist() for v in gt08.message_passing(A, x, 2, "sum", False)] == t["sum"]
    assert [v.tolist() for v in gt08.message_passing(A, x, 2, "sum", True)] == t["sum_loops"]
    assert np.allclose(gt08.message_passing(A, x, 1, "mean", True)[1], t["mean_loops_1"], rtol=0, atol=1e-12)
    assert np.allclose(gt08.message_passing(A, x, 1, "sym", True)[1], t["sym_loops_1"], rtol=0, atol=1e-12)
    n = toy["nodes"]
    assert abs(gt08.normalise(A, "sym", True)[n.index("D"), n.index("F")] - t["weight_D_F"]) < 1e-12
    L = gt08.laplacian(A)
    assert np.dot(L, x).tolist() == t["Lx"] and float(np.dot(x, np.dot(L, x))) == t["xLx"] == 25
    vals = np.linalg.eigvalsh(L)
    assert abs(vals[1] - t["lambda2"]) < 1e-9 and abs(vals[-1] - t["lambda_max"]) < 1e-9
    lg = gt08.line_graph(toy["edges"])
    assert (len(lg), int(lg.sum() // 2)) == (t["line_graph"]["nodes"], t["line_graph"]["edges"])
    assert t["sum"] == t["lecture"]["sum"] and t["xLx"] == t["lecture"]["xLx"]


def test_the_trained_toy_names_the_zones(gt08, p08):
    """Two layers on one labelled place per zone: the GCN names all four other places right, the model without a graph one."""
    if not gt08.torch_ok:
        pytest.skip("PyTorch is not installed")
    toy = gt08.load_toy(P08_DATA / "toy.json")
    lab = np.array([toy["nodes"].index(v) for v in toy["labelled"]])
    rest = np.setdiff1d(np.arange(len(toy["nodes"])), lab)
    feats = gt08.one_hot(len(toy["nodes"]))
    h = gt08.fit(gt08.GCN(7, 4, 3, layers=2, dropout=0.0), feats, gt08.normalise(toy["A"], "sym", True), toy["y"], lab, lab, epochs=200, lr=0.05, weight_decay=0.0)
    hm = gt08.fit(gt08.MLP(7, 4, 3, layers=2, dropout=0.0), feats, None, toy["y"], lab, lab, epochs=200, lr=0.05, weight_decay=0.0)
    assert int((h["pred"][rest] == toy["y"][rest]).sum()) == p08["toy"]["trained"]["one_hot"]["gcn_correct"] == 4
    assert int((hm["pred"][rest] == toy["y"][rest]).sum()) < 4


def test_the_street_graphs_are_the_lectures(gt08, streets, p08, scene08):
    s, c = streets, p08["streets"]["counts"]
    As, An = s["A_street"].astype(float), s["A_near"].astype(float)
    assert (len(s["y"]), int(s["tagged"].sum()), int(As.sum() // 2), int(An.sum() // 2)) == (c["segments"], c["tagged"], c["edges_street"], c["edges_near"])
    assert int((np.triu(As, 1) * np.triu(An, 1)).sum()) == c["shared"] == scene08["lecture"]["streets"]["shared_edges"]
    assert np.array_equal(gt08.proximity_graph(s["mid"], c["edges_street"]), s["A_near"]), "the proximity graph is no longer the closest pairs"
    assert np.array_equal(As, As.T) and np.trace(As) == 0
    assert round(gt08.homophily(As, s["y"], s["tagged"]), 4) == p08["streets"]["homophily"]["street"]
    assert round(gt08.homophily(An, s["y"], s["tagged"]), 4) == p08["streets"]["homophily"]["near"]
    rc = p08["streets"]["reach"]
    hs = gt08.hops(As, scene08["anchors"]["nassau"])
    assert [int((hs <= k).sum()) for k in range(9)] == rc["street"] == scene08["lecture"]["reach"]["street"]
    assert str(s["name"][rc["centre"]]) == rc["name"] == scene08["anchors"]["nassau_name"]
    seg = gt08.load_segments(P08_DATA / "streets.gpkg")
    assert len(seg) == len(s["y"]) and str(seg.crs.to_epsg()) == scene08["crs"].split(":")[1]
    assert (seg["segment"].to_numpy() == np.arange(len(seg))).all()


def test_the_lectures_run_comes_back(gt08, streets, p08):
    """The lecture's one run, retrained with the kit's toolbox: each accuracy within three points, the three models in order."""
    if not gt08.torch_ok:
        pytest.skip("PyTorch is not installed")
    s, lr = streets, p08["streets"]["lecture_run"]
    main = gt08.load_arrays(P08_DATA / "lecture_main.npz")
    y, F, P = gt08.street_classes(s), gt08.street_features(s, "with the road class"), gt08.street_graphs(s)
    lab, val, test = gt08.split_streets(s, "random", np.random.default_rng(gt08.SEED + 1))
    assert np.array_equal(lab, main["labelled"]) and np.array_equal(test, main["test"]), "the split is no longer the lecture's"
    acc = {}
    for name, key in (("mlp", "none"), ("near", "proximity"), ("gcn", "streets")):
        h, model = gt08.train(F, P[key], y, lab, val, gt08.SEED)
        acc[name] = gt08.accuracy(h["pred"], y, test)
        assert abs(acc[name] - lr[name]["accuracy"]) <= TOLERANCE, f"{name}: {acc[name]:.4f} against the published {lr[name]['accuracy']:.4f}"
        assert gt08.count_parameters(model) == lr[name]["parameters"]
    assert acc["mlp"] < acc["near"] < acc["gcn"], f"the three models are out of order: {acc}"
    if gt08.pyg_available():
        ei = gt08.to_edge_index(s["A_street"])
        library = gt08.pyg_gcn(F.shape[1], 32, 4)
        gt08.copy_weights(model, library)
        diff = float(np.abs(gt08.scores(model, F, P["streets"]) - gt08.scores(library, F, ei)).max())
        assert diff < 1e-4, f"the library's layer and the layer by hand differ by {diff}"
        assert ei.shape[1] == lr["pyg"]["edge_index_columns"] == 2 * p08["streets"]["counts"]["edges_street"]


def test_the_new_draws_come_back(gt08, streets, p08):
    """The five draws nobody had seen, with the live feature set on the three graphs: each mean within three points, the street graph first."""
    if not gt08.torch_ok:
        pytest.skip("PyTorch is not installed")
    b = p08["streets"]["draws"]["live"]
    F, P = gt08.street_features(streets, b["features"]), gt08.street_graphs(streets)
    got = {}
    for g in GRAPHS:
        r = gt08.run_draws(streets, F, P[g], "random", b["draws"], first=b["first"])
        got[g] = float(np.mean(r["accuracy"]))
        assert abs(got[g] - b[g]["mean"]) <= TOLERANCE, f"{g}: {got[g]:.4f} against the published {b[g]['mean']:.4f}"
        assert abs(float(np.mean(r["majority"])) - b["majority"]) < 1e-12, "the splits are no longer the published ones"
    assert got["streets"] > max(got["none"], got["proximity"]), f"the street graph no longer leads: {got}"


def test_the_forecasts_come_back_exactly(gt08, metr, p08, scene08):
    """The six forecasts of METR-LA at three horizons with the lecture's penalties: NumPy, so to the ninth decimal."""
    mr, m = scene08["rules"]["metr"], p08["metr"]
    o_te = metr["origins"][2]
    assert int(((metr["W"] > 0) & ~np.eye(len(metr["W"]), dtype=bool)).sum()) == m["edges_directed"]
    assert abs(float((metr["speeds"] == 0).mean()) - m["missing_share"]) < 1e-12 and len(o_te) == m["split"]["origins_test"]
    for mins, h in mr["horizons"].items():
        truth = metr["speeds"][o_te + h]
        got = {"last": gt08.masked_mae(metr["Xf"][o_te], truth), "ha": gt08.masked_mae(metr["HA"][metr["slots"][o_te + h]], truth)}
        for name, nb, clk in (("ar", False, None), ("arnb", True, None), ("arclock", False, metr["clock"]), ("arboth", True, metr["clock"])):
            P, _, _ = gt08.ridge_forecast(metr["Xf"], metr["NB"], metr["speeds"], metr["origins"], h, nb, [mr["alphas"][f"{name} {mins}"]], clk, mr["lags"])
            got[name] = gt08.masked_mae(P, truth)
        for name, v in got.items():
            assert abs(v - m["mae"][name][mins]) < 1e-9, f"{name} at {mins} minutes: {v} against the published {m['mae'][name][mins]}"
            assert abs(v - m["lecture"]["mae"][name][mins]) < 1e-9, f"{name} at {mins} minutes no longer gives the lecture's error"


def test_the_lessons_still_hold(p08):
    t, s, m = p08["toy"], p08["streets"], p08["metr"]
    assert t["xLx"] == 25 and t["sum"][1] == [3, 4, 3, 3, 6, 5, 1], "the toy no longer reproduces the slides"
    assert t["trained"]["one_hot"]["gcn_correct"] > t["trained"]["one_hot"]["mlp_correct"], "the graph no longer names the toy's zones"
    assert s["counts"]["shared_share"] < 0.7, "the street graph and the proximity graph now join nearly the same pairs"
    assert s["tagged_by_road_class"]["residential"] < 0.3 < s["tagged_by_road_class"]["tertiary"], "the small streets are no longer the untagged ones"
    rc = s["reach"]
    assert rc["street"][8] < rc["circle"]["8"], "eight steps on the street graph no longer reach fewer segments than the circle"
    lr = s["lecture_run"]
    assert lr["reproduces_lecture"] in (True, False)
    assert lr["mlp"]["accuracy"] < lr["near"]["accuracy"] < lr["gcn"]["accuracy"], "the street graph no longer beats the proximity graph and no graph"
    assert lr["gcn"]["c30_right"] >= 5 * max(lr["mlp"]["c30_right"], 1), "the graph no longer finds the 30 mph segments"
    assert lr["fixed_by_graph"] > 2 * lr["broken_by_graph"]
    live, none = s["draws"]["live"], s["draws"]["none"]
    assert live["streets"]["mean"] > live["proximity"]["mean"] and live["streets"]["mean"] > live["none"]["mean"]
    assert none["streets"]["mean"] > none["proximity"]["mean"] > none["none"]["mean"] - 1e-9, "without features the topology no longer carries the score"
    assert abs(none["none"]["mean"] - none["majority"]) < 0.02, "with no features and no graph the model no longer falls to the majority"
    assert s["near_a_label"]["random"] > 0.8 > 0.2 > s["near_a_label"]["blocks"], "the held-out blocks no longer keep the labels away"
    assert sum(s["fill"]["gcn"]["untagged_per_class"]) == s["counts"]["untagged"]
    mae = m["mae"]
    for mins in ("15", "30", "60"):
        assert mae["arnb"][mins] < mae["ar"][mins], f"the neighbours no longer help at {mins} minutes"
        assert mae["arclock"][mins] < mae["arnb"][mins], f"the clock no longer beats the neighbours at {mins} minutes"
    assert mae["ar"]["60"] - mae["arnb"]["60"] > mae["ar"]["15"] - mae["arnb"]["15"], "the neighbours no longer help more an hour ahead"
    assert mae["arclock"]["60"] - mae["arboth"]["60"] < 0.01, "after the clock the neighbours now add something"
    assert mae["arnb"]["60"] < m["line"]["mae_60"] < mae["ar"]["60"], "the straight-line graph no longer falls between"
