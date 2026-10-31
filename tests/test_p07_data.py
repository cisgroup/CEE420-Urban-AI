"""Every published P07 file is what the notebooks assume it is, and the lessons they teach still hold.

The kit ships Princeton's drive, bike and walk networks with two comparison towns, the boundary, the 839 cells with their
residents, the destinations, the block groups and the lecture's isochrones, the lecture's cell table and walk betweenness,
one theme-park visitor's day with the fifteen visitors' fixes, the coded stops and the park, the lecture's threshold
surface, Porto's origin-destination counts and cell-hour points, the trip-length histogram and the commuting flows between
the tracts. These tests hold the pipeline to its contract: every file carries its provenance and licence, nothing that
identifies a person ships, the files agree with one another, the published numbers come back from the published files
with the kit's own toolbox, and the lessons the notebooks are built on still hold on the data as published.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from conftest import P07_DATA, P07_KIT

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator|contributor|username|uploader", re.I)
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\||api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9]{8,}|CENSUS_API_KEY=\S", re.I)
FILES = ["scene.json", "ground_truth.json", "princeton_drive.graphml", "princeton_bike.graphml", "princeton_walk.graphml", "trenton_drive.graphml",
         "hopewell_drive.graphml", "layers.gpkg", "access_cells.csv.gz", "centrality_walk.csv.gz", "themepark_kp2.csv.gz", "themepark_kbf_fixes.csv.gz",
         "themepark_coded_stops.csv.gz", "themepark_kbf.gpkg", "stay_surface.json", "porto_od.npz", "porto_points.npz", "porto_lengths.json",
         "lodes_tracts.csv.gz", "lodes_tract_totals.csv.gz", "princeton_inflow.csv.gz", "lodes_blocks_princeton.csv.gz", "junction_cells.csv.gz", "dinky.json",
         "themepark_kbf_visits.csv.gz"]
LAYERS = {"boundary", "grid", "destinations", "block_groups", "isochrones", "duel"}
NASSAU_HALL = (528936.0, 4466524.0)


@pytest.fixture(scope="module")
def graphs(mob):
    return {m: mob.load_graph(P07_DATA / f"princeton_{m}.graphml") for m in ("drive", "bike", "walk")}


@pytest.fixture(scope="module")
def kp2(mob):
    return mob.load_fixes(P07_DATA / "themepark_kp2.csv.gz")


@pytest.mark.parametrize("name", FILES)
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries every file it opens."""
    assert (P07_DATA / name).exists()


def test_the_kit_stays_within_its_budget():
    total = sum(p.stat().st_size for p in P07_DATA.iterdir() if p.is_file())
    assert total < 100_000_000, f"the kit weighs {total / 1e6:.1f} MB; thirty laptops pull it"
    largest = max(P07_DATA.iterdir(), key=lambda p: p.stat().st_size)
    assert largest.stat().st_size < 50_000_000, f"{largest.name} is over GitHub's warning size"


def test_every_file_carries_its_provenance(scene07):
    import pyogrio

    for path in sorted(P07_DATA.glob("*.graphml")):
        prov = scene07["files"][path.name]
        assert {"source", "licence"} <= set(prov) and "OpenStreetMap" in prov["source"] and "ODbL" in prov["licence"], path.name
    for path in sorted(P07_DATA.glob("*.npz")):
        keys = set(np.load(path).files)
        assert {"source", "licence", "retrieved"} <= keys, f"{path.name} lacks provenance arrays: {sorted(keys)}"
    for path in sorted(P07_DATA.glob("*.json")):
        if path.name == "ground_truth.json":
            continue
        meta = json.loads(path.read_text())
        assert {"source", "licence", "retrieved"} <= set(meta), f"{path.name} lacks provenance keys"
    for gpkg, expected in ((P07_DATA / "layers.gpkg", LAYERS), (P07_DATA / "themepark_kbf.gpkg", {"ParkOutline", "WalkingPaths", "PointOfInterests"})):
        layers = {name for name, _ in pyogrio.list_layers(gpkg)}
        assert expected <= layers, f"{gpkg.name} holds {sorted(layers)}"
        for name in layers:
            meta = pyogrio.read_info(gpkg, layer=name).get("layer_metadata") or {}
            assert {"source", "licence", "retrieved"} <= set(meta), f"{gpkg.name}:{name} lacks provenance: {sorted(meta)}"
    for name in ("access_cells.csv.gz", "centrality_walk.csv.gz", "themepark_kp2.csv.gz", "lodes_tracts.csv.gz"):
        assert name in scene07["files"], f"{name} has no provenance row in scene.json"


def test_nothing_that_identifies_a_person_or_a_key_ships():
    for csv in sorted(P07_DATA.glob("*.csv.gz")):
        frame = pd.read_csv(csv)
        bad = [c for c in frame.columns if PII.search(c)]
        assert not bad, f"{csv.name} carries {bad}"
        text = frame.to_csv(index=False)
        assert "@" not in text and "<name>" not in text and "http" not in text, f"{csv.name} carries an address, a name element or a link"
    for path in sorted(P07_DATA.glob("*.graphml")):
        text = path.read_text()
        assert "@" not in text and "<name>" not in text, f"{path.name} carries an address or a name element"
    for path in list(P07_DATA.glob("*.json")) + [P07_KIT / "mob_tools.py"] + sorted(P07_KIT.glob("*.ipynb")):
        text = path.read_text()
        assert not SECRETS.search(text), f"{path.name} carries something that looks like a token or a key"
        assert "/Users" not in text, f"{path.name} mentions a home folder"
        assert "@" not in text or path.suffix == ".ipynb", f"{path.name} carries an address"
    for nb in sorted(P07_KIT.glob("*.ipynb")):
        assert "@" not in nb.read_text().replace("@", "", 0), nb.name     # the notebooks print nothing with an address
    pts = np.load(P07_DATA / "porto_points.npz")
    assert pts["user"].dtype == np.int32 and pts["cell"].dtype == np.int16 and pts["hour"].dtype == np.int8, "the points carry more than a cell and an hour"
    stops = pd.read_csv(P07_DATA / "themepark_coded_stops.csv.gz")
    assert stops["participant_id"].str.match(r"^[A-Z]{2}\d+$").all(), "a participant is not a code"


def test_published_p07_ground_truth_carries_no_answers():
    text = (P07_DATA / "ground_truth.json").read_text()
    published = json.loads(text)
    assert set(published) == {"generated", "p07"}, f"the published P07 ground truth carries {sorted(published)}"
    for key in ("p07_answers", "from_mode", "newly_found", "biggest_drop", "drops_walk", "pairs_published", "from_suppressed_share", "from_cell_hours"):
        assert key not in text, f"the published ground truth leaks the Friday answer {key}"


def test_the_graphs_are_the_lectures(graphs, scene07, p07):
    for m, g in graphs.items():
        lec = scene07["lecture"]["graphs"][m]
        assert (g.number_of_nodes(), g.number_of_edges()) == (lec["nodes"], lec["edges"]) == (p07["graphs"][m]["nodes"], p07["graphs"][m]["edges"]), m
        assert str(g.graph.get("crs")) == scene07["crs"], f"{m} is not in the course CRS"
    assert graphs["walk"].number_of_nodes() > graphs["bike"].number_of_nodes() > graphs["drive"].number_of_nodes()


def test_the_layers_agree_with_one_another(scene07, p07):
    import geopandas as gpd

    grid = gpd.read_file(P07_DATA / "layers.gpkg", layer="grid")
    bg = gpd.read_file(P07_DATA / "layers.gpkg", layer="block_groups")
    dest = gpd.read_file(P07_DATA / "layers.gpkg", layer="destinations")
    assert len(grid) == 839 and len(bg) == 21 and len(dest) == 293
    assert abs(grid["population"].sum() - bg["population"].sum()) < 1, "the residents on the cells and in the block groups differ"
    assert set(grid["GEOID"].dropna()) <= set(bg["GEOID"])
    assert set(dest["category"]) >= {"food", "grocery", "pharmacy", "school", "park", "bus_stop", "station"}
    cells = pd.read_csv(P07_DATA / "access_cells.csv.gz")
    assert len(cells) == 839 and np.allclose(cells["population"], grid["population"], atol=1e-3)
    assert not [c for c in cells.columns if c.endswith("junction_min")], "the town-edge minutes to the Junction are back in the cell table"
    jc = pd.read_csv(P07_DATA / "junction_cells.csv.gz")
    assert len(jc) == 839 and jc["transit_p50"].notna().sum() > 800
    assert ((jc["transit_p10"] <= jc["transit_p50"]) | jc["transit_p50"].isna()).all() and ((jc["transit_p50"] <= jc["transit_p90"]) | jc["transit_p90"].isna()).all()
    for g in (grid, bg, dest):
        assert g.crs.to_string() == scene07["crs"]


def test_the_published_numbers_come_back_from_the_files(mob, graphs, kp2, p07, scene07):
    """What the notebooks print beside 'reference' is recomputed here from the shipped files, with the toolbox."""
    import geopandas as gpd

    _, edges = mob.centrality(graphs["drive"])
    top = mob.top_streets(mob.edge_names(graphs["drive"], edges))
    assert top[0][0] == p07["centrality"]["top_streets"][0][0] and round(top[0][1], 4) == p07["centrality"]["top_streets"][0][1]
    for place, path in (("trenton", "trenton_drive.graphml"), ("hopewell", "hopewell_drive.graphml")):
        o = mob.orientation(mob.load_graph(P07_DATA / path))
        assert round(o["order"], 4) == p07["towns"][place]["order"]
    walk = graphs["walk"]
    nh = mob.nearest_nodes(walk, *NASSAU_HALL)
    reached = mob.isochrone(walk, nh, 15)
    assert abs(mob.isochrone_polygon(walk, reached, 25.0).area / 1e4 - p07["access"]["isochrones"]["walk_15"]["area_ha"]) < 0.01
    stays = mob.stays(kp2, p07["rules"]["stay"]["dist_m"], p07["rules"]["stay"]["min_s"])
    stops = pd.read_csv(P07_DATA / "themepark_coded_stops.csv.gz")
    coded = stops[stops["participant_id"] == "KP2"].rename(columns={"start_s": "start", "end_s": "end"}).dropna(subset=["start", "end"])
    r = mob.match_stays(stays, coded.sort_values("start").reset_index(drop=True), min_overlap_s=p07["rules"]["stay"]["overlap_s"])
    assert len(stays) == p07["traces"]["kp2"]["n_detected"] and round(r["precision"], 4) == p07["traces"]["kp2"]["precision"]
    assert round(r["recall"], 4) == p07["traces"]["kp2"]["recall"]
    z = np.load(P07_DATA / "porto_od.npz")
    od = z["od_500m"]
    ne = od > 0
    assert round(float((ne & (od < p07["rules"]["agg_k"])).sum() / ne.sum()), 4) == p07["privacy"]["aggregation"]["grids"]["500m"]["suppressed_share"]
    assert int(od.sum()) == p07["flows"]["porto"]["inside"] == int(z["n_inside_500m"])
    pairs = pd.read_csv(P07_DATA / "lodes_tracts.csv.gz", dtype={"h_tract": str, "w_tract": str})
    totals = pd.read_csv(P07_DATA / "lodes_tract_totals.csv.gz", dtype={"tract": str})
    df = (pairs.merge(totals[["tract", "residents"]].rename(columns={"tract": "h_tract"}), on="h_tract")
               .merge(totals[["tract", "jobs"]].rename(columns={"tract": "w_tract"}), on="w_tract"))
    inter = df[(df["S000"] >= 1) & (df["h_tract"] != df["w_tract"]) & (df["dist_km"] > scene07["lecture"]["lodes"]["min_dist_km"])]
    fit, pred = mob.gravity_fit(inter["S000"].to_numpy(float), inter["residents"], inter["jobs"], inter["dist_km"])
    assert round(fit["gamma"], 4) == p07["flows"]["lodes"]["listed"]["gamma"] and round(mob.cpc(inter["S000"], pred), 4) == p07["flows"]["lodes"]["listed"]["cpc"]
    home = pd.read_csv(P07_DATA / "princeton_inflow.csv.gz", dtype={"tract": str})
    win = home[(home["dist_km"] <= scene07["lecture"]["princeton"]["window_km"]) & (home["residents"] > 0)]
    pfit, ppred = mob.gravity_fit_destination(win["inflow"].to_numpy(float), win["residents"], win["dist_km"], decay="power")
    assert round(pfit["gamma"], 4) == p07["flows"]["princeton"]["power"]["gamma"] and round(mob.cpc(win["inflow"], ppred), 4) == p07["flows"]["princeton"]["power"]["cpc"]
    pts = np.load(P07_DATA / "porto_points.npz")
    per_day = pd.Series(pts["user"]).value_counts()
    assert int((per_day >= p07["rules"]["unique_min_points"]).sum()) == p07["privacy"]["uniqueness"]["scored"]
    dk = json.loads((P07_DATA / "dinky.json").read_text())
    deps = [t[2] for t in dk["trips"] if t[1] == dk["stops"]["junction"]["name"]]
    nd = mob.next_departure(deps, np.asarray(dk["minutes"]["minute"], dtype=float), dk["access_min"], dk["ride_min"])
    assert int(np.isclose(nd["total"], dk["minutes"]["rail_min"]).sum()) == p07["access"]["dinky"]["matches"] == len(nd), "the next-train rule misses the router"
    jc = pd.read_csv(P07_DATA / "junction_cells.csv.gz").set_index("cell")
    row = jc.loc[p07["access"]["transit"]["nassau_cell"]]
    assert (row["walk_min"], row["transit_p50"], row["car_min"]) == tuple(p07["access"]["transit"][k] for k in ("walk_min", "transit_p50", "car_min"))
    park = pd.read_csv(P07_DATA / "themepark_kbf_fixes.csv.gz")
    box = scene07["lecture"]["park"]["box"]
    x, y = np.clip(park["x"], box[0], box[2] - 0.01), np.clip(park["y"], box[1], box[3] - 0.01)
    cell, hour = mob.cell_hours(mob.make_grid(tuple(box), 25.0, crs=scene07["lecture"]["park"]["crs"]), x, y, park["t_s"])
    lit = mob.published_counts(cell, hour, k=1, ids=park["participant_id"])[1]
    assert int((lit > 0).sum()) == p07["privacy"]["park"]["release"]["25"]["lit_cell_hours"]


def test_the_lessons_still_hold(p07):
    g, t, a, tr, fl, pr = p07["graphs"], p07["towns"], p07["access"], p07["traces"], p07["flows"], p07["privacy"]
    assert g["walk_drive_ratio"] > 5, "the walk graph no longer dwarfs the drive graph"
    assert p07["centrality"]["top_streets"][0][0] == "Nassau Street", "Nassau Street is no longer the most central street by car"
    assert t["hopewell"]["order"] > t["princeton"]["order"] > t["trenton"]["order"], "the three towns no longer tell the three-grids story"
    assert p07["circuity"]["circuity"] > 1.1, "the circle no longer lies by a measurable amount"
    assert a["duel"]["circle"] > 3 * a["duel"]["network"], "the circle no longer overcounts the duel"
    assert a["isochrones"]["circle_1200"]["ratio_to_walk_15"] > 1.5
    assert a["share_grocery_15"]["drive"] > a["share_grocery_15"]["bike"] > a["share_grocery_15"]["walk"], "the modes are out of order"
    assert a["all_six"]["bike"]["share"] > a["all_six"]["walk"]["share"] + 0.3, "the bicycle no longer buys the 15-minute city"
    assert a["fairness"]["pearson_income_hansen"] < 0, "income and accessibility are no longer negatively related"
    assert tr["kp2"]["recall"] < tr["kp2"]["precision"], "the stay rule no longer misses more than it invents"
    assert tr["kp2"]["coded_under_10_min"] >= 5, "the short stops that cap the recall are gone"
    assert tr["pooled"]["precision"] > 0.6 and tr["pooled"]["recall"] > 0.4
    assert tr["leg"]["directness"] < 0.6, "the leg no longer shows a detour"
    assert fl["lodes"]["complete"]["cpc"] > fl["lodes"]["complete"]["radiation_cpc"] + 0.3, "the radiation model no longer fails at the tract scale"
    assert 0.5 < fl["lodes"]["complete"]["gamma"] < 1.5, "the region's gravity exponent left the range the literature gives"
    assert fl["princeton"]["power"]["gamma"] > fl["lodes"]["complete"]["gamma"] + 0.5, "Princeton's pull no longer falls more steeply than the region's"
    assert fl["princeton"]["power"]["cpc"] > fl["princeton"]["radiation"]["cpc"] + 0.3, "the radiation model no longer fails for one destination"
    rt = p07["routing"]
    assert rt["astar"]["settled"] < rt["bidirectional"]["forward"] < rt["dijkstra"]["settled"] < rt["dijkstra"]["every_node"], "the searches no longer rank"
    assert rt["fewest_edges"]["edges"] < rt["shortest"]["edges"] and rt["fewest_edges"]["length_m"] > rt["shortest"]["length_m"]
    assert p07["toy"]["brandes_D"]["half"] == 4.5 and p07["toy"]["checks"] == 18, "the toy no longer reproduces the slides"
    assert a["transit"]["share30"]["car"] > a["transit"]["share30"]["transit"] > a["transit"]["share30"]["walk"], "the three modes are out of order"
    assert a["dinky"]["hour"]["max"] - a["dinky"]["hour"]["min"] >= a["dinky"]["headway_peak_min"], "the wait no longer spans a headway"
    assert tr["kp2"]["longest_gap_inside"] is not None and tr["stranger"]["longest"]["minutes"] > 30, "the longest gap no longer lies inside a coded stop"
    assert tr["on_paths"]["share"] > 0.6, "the fixes no longer follow the paths"
    park = pr["park"]
    assert park["release"]["25"]["k"]["10"]["share"] < 0.05 < park["release"]["25"]["k"]["3"]["share"], "a small crowd no longer hides nobody at k of 10"
    assert park["uniqueness"]["p4"] > 0.9, "four points no longer identify the park's visitors"
    assert p07["privacy"]["homework"]["block"]["median"] == 1, "a home and work pair no longer singles out the median worker"
    assert pr["uniqueness"]["p4"] > 0.9 > 0.5 > pr["uniqueness"]["p2"], "four points no longer identify nearly everybody"
    shares = [pr["aggregation"]["grids"][k]["suppressed_share"] for k in ("500m", "1km", "2km")]
    assert shares[0] > shares[1] > shares[2], "coarser cells no longer suppress fewer pairs"
    assert pr["aggregation"]["grids"]["500m"]["trips_in_suppressed_share"] < 0.1, "the suppressed pairs now hold many trips"
    assert pr["aggregation"]["key_space"]["total"] == 18_278_000
    assert p07["class_traces"]["published"] in (True, False)
