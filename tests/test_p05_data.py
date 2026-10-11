"""Every published P05 file is what the notebooks assume it is, and the lessons they teach still hold.

The kit ships the lecture's running example cut to one window at 2.4 m and one test block at 30 cm: the
imagery and the mask, the two models' probability maps, the chip table with the lecture's split, the
frozen-encoder U-Net's weights, the frozen and fine-tuned features of every chip, the lecture's curves,
one validation chip after every epoch, 24 training chips, the first batch, and one street image with its
label map. These tests hold the pipeline to its contract: every file carries its provenance and licence,
nothing that identifies a person ships, the files agree with one another (one grid, one split, one seed),
the published numbers come back from the published files, and the lessons the notebooks are built on
still hold on the data as published.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import P05_DATA

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator|contributor|username", re.I)
FILES = ["scene.json", "ground_truth.json", "layers.gpkg", "chips.csv", "window_2m4.tif", "mask_2m4.tif", "prob_2m4.tif",
         "block_7.tif", "mask_7.tif", "prob_7_finetuned.tif", "prob_7_frozen.tif", "chip_features.npz", "segmenter_frozen.npz",
         "train_chips.npz", "first_batch.npz", "curves.json", "evolution.npz", "street_nassau.jpg", "street_nassau.npz"]
BLOCK_PX, CHIP_PX, N_BLOCKS = 2048, 256, 5


@pytest.fixture(scope="module")
def chips():
    return pd.read_csv(P05_DATA / "chips.csv")


@pytest.fixture(scope="module")
def block_rasters():
    """The four rasters of block 7, read once: the image, the mask and the two probability maps."""
    out = {}
    for name in ("block_7", "mask_7", "prob_7_finetuned", "prob_7_frozen"):
        with rasterio.open(P05_DATA / f"{name}.tif") as src:
            out[name] = (src.read(), src.transform, src.crs)
    return out


@pytest.mark.parametrize("name", FILES)
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries every file it opens."""
    assert (P05_DATA / name).exists()


def test_the_kit_stays_small():
    total = sum(p.stat().st_size for p in P05_DATA.iterdir() if p.is_file())
    assert total < 60_000_000, f"the kit weighs {total / 1e6:.1f} MB; thirty laptops pull it"
    largest = max(P05_DATA.iterdir(), key=lambda p: p.stat().st_size)
    assert largest.stat().st_size < 30_000_000, f"{largest.name} is too large for the repository"


def test_every_raster_carries_its_provenance():
    for path in sorted(P05_DATA.glob("*.tif")):
        with rasterio.open(path) as src:
            tags = src.tags()
        assert {"source", "licence", "retrieved"} <= set(tags), f"{path.name} lacks provenance tags: {sorted(tags)}"


def test_every_array_file_carries_its_provenance():
    for path in sorted(P05_DATA.glob("*.npz")):
        keys = set(np.load(path).files)
        assert {"source", "licence", "retrieved"} <= keys, f"{path.name} lacks provenance arrays: {sorted(keys)}"


def test_every_layer_carries_its_provenance():
    import pyogrio

    layers = {name for name, _ in pyogrio.list_layers(P05_DATA / "layers.gpkg")}
    assert {"window", "blocks", "footprints_7"} <= layers, f"layers.gpkg holds {sorted(layers)}"
    for name in layers:
        meta = pyogrio.read_info(P05_DATA / "layers.gpkg", layer=name).get("layer_metadata") or {}
        assert {"source", "licence", "retrieved"} <= set(meta), f"layer {name} lacks provenance: {sorted(meta)}"
    footprints = pyogrio.read_info(P05_DATA / "layers.gpkg", layer="footprints_7")["layer_metadata"]
    assert "OpenStreetMap" in footprints["source"] and "ODbL" in footprints["licence"], "the footprints are OpenStreetMap's and say so"


def test_nothing_that_identifies_a_person_ships(scene05):
    from PIL import Image

    for csv in sorted(P05_DATA.glob("*.csv")):
        bad = [c for c in pd.read_csv(csv, nrows=1).columns if PII.search(c)]
        assert not bad, f"{csv.name} carries {bad}"
    street = scene05["choices"]["street"]
    assert set(street) <= {"street", "label", "image", "date", "credit", "model", "green_view_index_street", "sky_street", "images_street"}, (
        f"the street record carries {sorted(street)}; the contributor stays a Mapillary contributor")
    assert "Mapillary contributors" in street["credit"] and "CC BY-SA" in street["credit"]
    image = Image.open(P05_DATA / "street_nassau.jpg")
    assert not dict(image.getexif()), "the street image still carries EXIF, which may hold a position or a device"
    for forbidden in ("/Users", "data/scripts", "mapillary.env"):
        assert forbidden not in (P05_DATA / "scene.json").read_text(), f"scene.json mentions {forbidden}"


def test_published_p05_ground_truth_carries_no_answers():
    text = (P05_DATA / "ground_truth.json").read_text()
    published = json.loads(text)
    assert set(published) == {"generated", "p05"}, f"the published P05 ground truth carries {sorted(published)}"
    for key in ("p05_answers", "threshold_0_3", "blocks_307m", "answers"):
        assert key not in text, f"the published ground truth leaks the Friday answer {key}"


def test_the_rasters_share_one_grid(block_rasters, scene05):
    (image, tr, crs), (mask, tr_m, crs_m) = block_rasters["block_7"], block_rasters["mask_7"]
    assert image.shape == (3, BLOCK_PX, BLOCK_PX) and image.dtype == np.uint8 and mask.shape == (1, BLOCK_PX, BLOCK_PX)
    assert set(np.unique(mask).tolist()) <= {0, 1}, "the mask is 0 or 1"
    for name in ("mask_7", "prob_7_finetuned", "prob_7_frozen"):
        arr, tr_n, crs_n = block_rasters[name]
        assert tr_n == tr and crs_n == crs and arr.shape[1:] == (BLOCK_PX, BLOCK_PX), f"{name} is not on the block's grid"
        assert arr.dtype == np.uint8
    assert abs(tr.a - 0.3) < 1e-9 and str(crs) == "EPSG:26918"
    with rasterio.open(P05_DATA / "window_2m4.tif") as small, rasterio.open(P05_DATA / "mask_2m4.tif") as share, rasterio.open(P05_DATA / "prob_2m4.tif") as prob:
        assert small.shape == share.shape == prob.shape == (1280, 1280) and abs(small.res[0] - 2.4) < 1e-9
        assert small.transform == share.transform == prob.transform and small.crs == crs
        left, top = small.transform.c, small.transform.f
    # block 7 is the second row, third column of the five by five grid, and its bounds say so
    row, col = divmod(scene05["block"]["id"], N_BLOCKS)
    assert abs(tr.c - (left + col * BLOCK_PX * 0.3)) < 1e-6 and abs(tr.f - (top - row * BLOCK_PX * 0.3)) < 1e-6
    minx, miny, maxx, maxy = scene05["block"]["bounds"]
    assert abs(minx - tr.c) < 1e-6 and abs(maxy - tr.f) < 1e-6 and abs(maxx - minx - BLOCK_PX * 0.3) < 1e-6


def test_the_chip_table_is_the_lectures_split(chips, scene05, p05):
    tiled = p05["chips"]["tiled"]
    assert len(chips) == tiled["total"] == 1600
    counts = chips["split"].value_counts()
    assert (counts["train"], counts["val"], counts["test"]) == (tiled["train"], tiled["val"], tiled["test"])
    assert chips["row"].between(0, 10240 - CHIP_PX).all() and chips["col"].between(0, 10240 - CHIP_PX).all()
    assert ((chips["row"] % CHIP_PX == 0) & (chips["col"] % CHIP_PX == 0)).all(), "the tiled chips sit on the chip grid"
    assert (chips["block"] == (chips["row"] // BLOCK_PX) * N_BLOCKS + chips["col"] // BLOCK_PX).all(), "the block id is row-major"
    role = scene05["blocks"]["role"]
    assert (chips["split"] == chips["block"].map(lambda b: role[str(b)])).all(), "a chip's role is its block's role"
    assert (chips["label"].astype(bool) == (chips["share"] >= scene05["chips"]["label_share"])).all()
    assert int(chips["label"].sum()) == tiled["with_building"]
    assert (chips.loc[chips["block"] == scene05["block"]["id"], "split"] == "test").all() and (chips["block"] == scene05["block"]["id"]).sum() == 64


def test_the_features_match_the_table(chips):
    F = np.load(P05_DATA / "chip_features.npz")
    assert F["frozen"].shape == F["finetuned"].shape == (1600, 512) and F["frozen"].dtype == np.float32
    assert (F["row"] == chips["row"].to_numpy()).all() and (F["col"] == chips["col"].to_numpy()).all()
    assert (F["block"] == chips["block"].to_numpy()).all() and (F["labels"].astype(bool) == chips["label"].astype(bool).to_numpy()).all()
    code = chips["split"].map({"train": 0, "val": 1, "test": 2}).to_numpy()
    assert (F["split"] == code).all(), "the split is coded 0, 1, 2 for train, validation, test in the table's order"
    assert np.isfinite(F["frozen"]).all() and np.isfinite(F["finetuned"]).all()
    assert ((F["prob_frozen"] >= 0) & (F["prob_frozen"] <= 1)).all() and ((F["prob_finetuned"] >= 0) & (F["prob_finetuned"] <= 1)).all()


def test_the_published_numbers_come_back_from_the_files(seg, chips, block_rasters, scene05, p05):
    """What the notebooks print beside 'reference' is recomputed here from the shipped files, with the toolbox."""
    mask = block_rasters["mask_7"][0][0] > 0
    assert round(float(mask.mean()), 4) == p05["block"]["share"]
    for name, key in (("prob_7_finetuned", "finetuned"), ("prob_7_frozen", "frozen")):
        prob = block_rasters[name][0][0]
        assert round(seg.iou(prob >= 128, mask), 4) == p05["block"][key]["iou"], f"the {key} map no longer gives the published IoU"
        rows, pooled = seg.chip_scores(prob / 255, mask)
        defined = [r["iou"] for r in rows if r["iou"] is not None]
        assert len(defined) == p05["block"][key]["chips_defined"] and round(float(np.mean(defined)), 4) == p05["block"][key]["per_chip_mean"]
        assert round(pooled["accuracy"], 4) == p05["block"][key]["accuracy"]
    with rasterio.open(P05_DATA / "mask_2m4.tif") as src:
        share = src.read(1)
    assert abs(share.mean() / 100 - p05["window"]["mask_share"]) < 0.001
    F = np.load(P05_DATA / "chip_features.npz")
    labels, train, test = F["labels"].astype(int), F["split"] == 0, np.flatnonzero(F["split"] == 2)
    head = seg.fit_head(F["frozen"], labels, train)
    assert round(seg.head_accuracy(head, F["frozen"], labels, test), 4) == p05["head"]["block_split"]
    assert round(max(labels[test].mean(), 1 - labels[test].mean()), 4) == p05["head"]["baseline"]
    batch = np.load(P05_DATA / "first_batch.npz")
    assert (seg.first_batch(chips, seed=scene05["seed"]) == batch["index"]).all(), "the first batch no longer replays from the seed"
    assert int(batch["label"].sum()) == p05["chips"]["first_batch"]["with_building"]
    z = np.load(P05_DATA / "street_nassau.npz")
    names = [str(n) for n in z["names"]]
    assert round(seg.green_view_index(z["labels"], names), 4) == p05["street"]["green_view_index"]
    assert round(seg.class_shares(z["labels"], names)["sky"], 4) == p05["street"]["sky"]


def test_the_objects_match_the_published_counts(seg, block_rasters, p05):
    import geopandas as gpd

    image, transform, _ = block_rasters["block_7"]
    footprints = gpd.read_file(P05_DATA / "layers.gpkg", layer="footprints_7")
    assert len(footprints) == p05["block"]["footprints"]
    ids = seg.rasterise(footprints, transform, image.shape[1:])
    assert len(np.unique(ids)) - 1 == len(footprints), "every footprint rasterises to its own id, 0 is the background"
    prob = block_rasters["prob_7_finetuned"][0][0]
    matched = seg.match_objects(prob >= 128, ids, threshold=0.5)
    ref = p05["block"]["finetuned"]["objects"]
    assert (matched["reference"], matched["predicted"], matched["matched"]) == (ref["reference"], ref["predicted"], ref["matched"])
    assert round(matched["precision"], 4) == ref["precision"] and round(matched["recall"], 4) == ref["recall"]


def test_the_shipped_model_is_the_lectures_frozen_run(seg, block_rasters, p05):
    """The weights load into the toolbox's U-Net, count as published, and predict the block's first chip as the shipped map does."""
    pytest.importorskip("torch")
    model, provenance = seg.load_model(P05_DATA / "segmenter_frozen.npz")
    counts = seg.count_parameters(model)
    assert (counts["total"], counts["encoder"], counts["decoder"]) == (p05["model"]["parameters"], p05["model"]["encoder"], p05["model"]["decoder"])
    assert provenance["run"] == p05["model"]["run"] and int(provenance["best_epoch"]) == p05["model"]["best_epoch"]
    image = block_rasters["block_7"][0]
    own = seg.predict(model, image[None, :, :CHIP_PX, :CHIP_PX], dev="cpu")[0]
    shipped = block_rasters["prob_7_frozen"][0][0][:CHIP_PX, :CHIP_PX] / 255
    keep = slice(64, CHIP_PX - 64)                        # the chip's centre; the block's rim was predicted with the neighbours in view
    assert np.abs(own[keep, keep] - shipped[keep, keep]).mean() < 0.01, "the shipped weights no longer reproduce the shipped map"
    assert seg.iou(own[keep, keep] >= 0.5, shipped[keep, keep] >= 0.5) > 0.97


def test_the_training_chips_and_the_batch_come_from_the_table(chips, scene05, p05):
    T = np.load(P05_DATA / "train_chips.npz")
    choice = scene05["choices"]["train_chips"]
    assert T["images"].shape == (24, 3, CHIP_PX, CHIP_PX) and T["masks"].shape == (24, CHIP_PX, CHIP_PX) and T["images"].dtype == np.uint8
    assert set(np.unique(T["masks"]).tolist()) <= {0, 1}
    assert T["index"].tolist() == choice["indices"] and sorted(set(T["block"].tolist())) == choice["blocks"] == p05["train_chips"]["blocks"]
    assert (chips.loc[T["index"], "split"] == "train").all(), "a training chip never comes from a held-out block"
    assert (chips.loc[T["index"], "row"].to_numpy() == T["row"]).all() and (chips.loc[T["index"], "col"].to_numpy() == T["col"]).all()
    assert np.bincount(T["block"]).max() == 3, "three chips per block"
    assert round(float(T["share"].min()), 4) == p05["train_chips"]["share_min"] and round(float(T["share"].max()), 4) == p05["train_chips"]["share_max"]
    assert np.allclose(T["masks"].mean(axis=(1, 2)), T["share"], atol=1e-3), "a chip's share is its mask's mean"
    B = np.load(P05_DATA / "first_batch.npz")
    assert B["images"].shape == (32, 3, CHIP_PX, CHIP_PX) and (chips.loc[B["index"], "split"] == "train").all()
    assert (B["label"] == (B["share"] >= scene05["chips"]["label_share"])).all()


def test_the_evolution_file_is_one_validation_chip(scene05, p05):
    ev = np.load(P05_DATA / "evolution.npz")
    e = p05["evolution"]
    assert ev["prob"].shape == (e["epochs"], CHIP_PX, CHIP_PX) and ev["prob"].dtype == np.uint8 and len(ev["iou"]) == e["epochs"]
    assert int(ev["best_epoch"]) == e["best_epoch"] and int(ev["block"]) in scene05["blocks"]["val"]
    assert (int(ev["row"]), int(ev["col"])) == (e["row"], e["col"]) and round(float(ev["share"]), 4) == e["share"]
    for epoch, value in e["iou"].items():
        assert round(float(ev["iou"][int(epoch) - 1]), 4) == value
    mask = ev["mask"] > 0
    for epoch in e["shown"]:
        pred = ev["prob"][epoch - 1] >= 128
        again = (pred & mask).sum() / (pred | mask).sum()
        assert abs(again - ev["iou"][epoch - 1]) < 0.01, f"epoch {epoch}: the stored IoU is not the stored map's"


def test_the_curves_are_the_lectures_runs(p05):
    curves = json.loads((P05_DATA / "curves.json").read_text())
    for key in ("main", "frozen", "scratch", "noskip"):
        run = curves["unet"][key]
        assert run["best_epoch"] == p05["lecture"][key]["best_epoch"] and round(run["test_iou"], 4) == p05["lecture"][key]["test_iou"]
        assert run["trainable"] == p05["lecture"][key]["trainable"] and len(run["curve"]) == p05["lecture"][key]["epochs"]


def test_the_lessons_still_hold(p05):
    b, lec = p05["block"], p05["lecture"]
    assert b["finetuned"]["iou"] > b["frozen"]["iou"], "fine-tuning the encoder no longer helps on the block"
    for key in ("finetuned", "frozen"):
        assert b[key]["per_chip_mean"] < b[key]["iou"], "the chip-by-chip average no longer sits below the pooled IoU"
        assert b[key]["accuracy"] - b[key]["iou"] > 0.15 and b[key]["no_building_accuracy"] > 0.8, "pixels right no longer flatters"
    pair = b["pair"]
    assert abs(pair["low"]["accuracy"] - pair["high"]["accuracy"]) < 0.005 and pair["high"]["iou"] - pair["low"]["iou"] > 0.3
    obj = b["finetuned"]["objects"]
    assert obj["matched"] <= min(obj["reference"], obj["predicted"]) and round(obj["matched"] / obj["predicted"], 4) == obj["precision"]
    assert round(obj["matched"] / obj["reference"], 4) == obj["recall"]
    head = p05["head"]
    assert head["block_split"] >= head["baseline"] + 0.1 and head["block_split"] <= head["finetuned_features"]
    assert lec["classifier"]["finetuned"]["test_acc"] > lec["classifier"]["frozen"]["test_acc"]
    assert lec["main"]["test_iou"] > lec["frozen"]["test_iou"] and lec["random_split"]["test_iou"] > lec["main"]["test_iou"] + 0.05, (
        "the random split no longer flatters, the lesson of the blocks is gone")
    e = p05["evolution"]
    assert e["iou"]["1"] >= 0.78, "after one epoch the chip no longer overlaps by four fifths"
    best_of_chip = max(e["iou"], key=lambda k: e["iou"][k])
    assert int(best_of_chip) != e["best_epoch"], "the kept epoch is the chip's own best, the lesson of 05-02 is gone"
    s = p05["street"]
    assert s["green_view_index"] > s["green_view_index_street"] and s["sky"] < s["sky_street"]
    assert p05["train_chips"]["n"] == 24 and len(p05["train_chips"]["blocks"]) == 8
