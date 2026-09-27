"""Every published P04 reference number is recomputed from the shipped files.

The kit's notebooks cut the town scene into chips, label them from a land cover map and train a forest, and
these tests hold the pipeline to its promises: every raster carries its provenance, the labels collapse as the
notebooks say, the chip table has the size the lecture quoted, and the forest's two scores reproduce.
"""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold

from conftest import P04_DATA

FEATURES = ["blue", "green", "red", "nir", "swir16", "swir22", "ndvi", "ndbi"]
CLASSES = {1: "tree cover", 2: "grass", 3: "built-up", 4: "water"}
WC_COLLAPSE = {10: 1, 30: 2, 40: 2, 90: 2, 50: 3, 80: 4}


def scene_and_labels(p04):
    """The features, the labels and the town mask exactly as notebook 04_01 builds them."""
    with rasterio.open(P04_DATA / f"s2_{p04['scene']['date'].replace('-', '')}_town_10m.tif") as f:
        b, g, r, n = (f.read(i).astype("float32") for i in (1, 2, 3, 4))
        transform, crs, shape = f.transform, f.crs, (f.height, f.width)
    with rasterio.open(P04_DATA / f"s2_{p04['scene']['date'].replace('-', '')}_town_20m.tif") as f:
        up = lambda a: np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)  # noqa: E731
        s16, s22 = up(f.read(1).astype("float32")), up(f.read(2).astype("float32"))
    with np.errstate(invalid="ignore", divide="ignore"):
        feat = np.stack([b, g, r, n, s16, s22, (n - r) / (n + r), (s16 - n) / (s16 + n)])
    feat[:, (r <= 0) | (n <= 0)] = np.nan
    town = gpd.read_file(P04_DATA / "princeton_boundary.geojson").to_crs(crs)
    mask = rasterize([(town.geometry.iloc[0], 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    with rasterio.open(P04_DATA / "worldcover_2021_town_10m.tif") as f:
        raw = f.read(1)
    lab = np.zeros(raw.shape, dtype="uint8")
    for code, cls in WC_COLLAPSE.items():
        lab[raw == code] = cls
    lab[~mask] = 0
    return feat, lab, mask, transform


def chip_table(feat, lab, size, stride):
    rows, cols = lab.shape
    recs = []
    for r0 in range(0, rows - size + 1, stride):
        for c0 in range(0, cols - size + 1, stride):
            l = lab[r0:r0 + size, c0:c0 + size]
            f = feat[:, r0:r0 + size, c0:c0 + size].reshape(len(feat), -1)
            if np.mean(l > 0) < 0.9 or not np.isfinite(f).all():
                continue
            counts = np.bincount(l[l > 0], minlength=5)[1:]
            recs.append((r0, c0, int(np.argmax(counts)) + 1, counts.max() / counts.sum(), *f.mean(axis=1), *f.std(axis=1)))
    return np.array(recs)


def test_every_raster_carries_its_provenance(rasters04):
    assert len(rasters04) == 6, sorted(rasters04)
    for name, src in rasters04.items():
        tags = src.tags()
        assert {"source", "licence", "retrieved"} <= set(tags), f"{name} lacks provenance tags: {sorted(tags)}"
    assert "Copernicus" in rasters04["s2_20250822_town_10m.tif"].tags()["licence"], "the Sentinel credit is a licence condition"
    assert "CC BY 4.0" in rasters04["worldcover_2021_town_10m.tif"].tags()["licence"]
    assert "CC BY 4.0" in rasters04["io_lulc_2023_town_10m.tif"].tags()["licence"]


def test_every_raster_shares_the_scene_grid(rasters04, p04):
    ten = rasters04["s2_20250822_town_10m.tif"]
    assert ten.crs.to_string() == p04["scene"]["crs"] and (ten.width, ten.height) == (p04["scene"]["width"], p04["scene"]["height"])
    assert round(ten.res[0], 3) == p04["scene"]["res_m"]
    for name in ("s2_20250310_town_10m.tif", "worldcover_2021_town_10m.tif", "io_lulc_2023_town_10m.tif"):
        src = rasters04[name]
        assert src.transform == ten.transform and (src.width, src.height) == (ten.width, ten.height), f"{name} is not on the scene's grid"
    for name in ("s2_20250822_town_20m.tif", "s2_20250310_town_20m.tif"):
        src = rasters04[name]
        assert round(src.res[0], 3) == 2 * p04["scene"]["res_m"] and src.count == 3, f"{name} should hold the two shortwave bands and the scene classification at 20 m"
    assert "do not apply them again" in ten.tags()["scale_offset_note"]


def test_the_labels_collapse_as_the_notebooks_say(p04):
    feat, lab, mask, _ = scene_and_labels(p04)
    assert int(mask.sum()) == p04["scene"]["cells_in_town"]
    assert int(((lab > 0) & np.isfinite(feat).all(axis=0)).sum()) == p04["scene"]["cells_labelled"]
    for cls, name in CLASSES.items():
        assert round(100 * float(np.mean(lab[mask] == cls)), 1) == p04["labels"]["share_percent"][name], name
    assert p04["labels"]["share_percent"]["tree cover"] > 75, "the lesson of the majority baseline needs a town that is mostly trees"


def test_the_chip_table_has_the_lectures_size(p04):
    feat, lab, mask, _ = scene_and_labels(p04)
    ch = p04["chips"]
    over = chip_table(feat, lab, ch["size_cells"], ch["stride_cells"])
    tile = chip_table(feat, lab, ch["size_cells"], ch["size_cells"])
    assert len(over) == ch["overlapping"]["n"] and len(tile) == ch["tiled"]["n"]
    assert over.shape[1] == 4 + ch["numbers_per_chip"]
    counts = {name: int(np.sum(over[:, 2].astype(int) == cls)) for cls, name in CLASSES.items()}
    assert counts == ch["overlapping"]["class_counts"]


@pytest.mark.slow
def test_the_forest_scores_reproduce(p04):
    feat, lab, mask, transform = scene_and_labels(p04)
    ch, fo = p04["chips"], p04["forest"]
    over = chip_table(feat, lab, ch["size_cells"], ch["stride_cells"])
    X, y = over[:, 4:], over[:, 2].astype(int)
    xs, ys = transform * (over[:, 1] + ch["size_cells"] / 2, over[:, 0] + ch["size_cells"] / 2)
    xs, ys = np.asarray(xs), np.asarray(ys)
    groups = ((xs - xs.min()) // fo["block_m"]).astype(int) * 1000 + ((ys - ys.min()) // fo["block_m"]).astype(int)
    make = lambda: RandomForestClassifier(n_estimators=fo["trees"], min_samples_leaf=fo["min_samples_leaf"], random_state=fo["seed"], n_jobs=-1)  # noqa: E731
    random_folds = [float(np.mean(make().fit(X[tr], y[tr]).predict(X[te]) == y[te])) for tr, te in KFold(fo["folds"], shuffle=True, random_state=fo["seed"]).split(X)]
    block_folds = [float(np.mean(make().fit(X[tr], y[tr]).predict(X[te]) == y[te])) for tr, te in GroupKFold(fo["folds"]).split(X, y, groups)]
    assert round(100 * float(np.mean(random_folds)), 1) == fo["random_mean_percent"]
    assert round(100 * float(np.mean(block_folds)), 1) == fo["blocked_mean_percent"]
    assert fo["random_mean_percent"] > fo["blocked_mean_percent"], "the random split must overestimate, or the lesson is gone"


def test_published_p04_ground_truth_carries_no_answers():
    published = json.loads((P04_DATA / "ground_truth.json").read_text())
    assert set(published) == {"generated", "p04"}, "the your-turn answers must not ship"
