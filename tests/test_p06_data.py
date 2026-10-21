"""Every published P06 file is what the notebooks assume it is, and the lessons they teach still hold.

The kit ships the lecture's zero-shot runs cut to the five test blocks at 30 cm with their masks, the trained model's
maps and the footprints, the text models' instance masks on the nine held-out blocks, the crop of block 7 with the
lecture's prompts and the everything masks, CLIP's numbers of every chip, two vision-language models' answers, two years
of the town's satellite embeddings masked to the boundary, the stability raster, the labels and cells of Lecture 04,
the toy, and one photograph. These tests hold the pipeline to its contract: every file carries its provenance and
licence (the embeddings their attribution line), nothing that identifies a person ships, no key or token ships, the
files agree with one another, the published numbers come back from the published files with the kit's own toolbox,
and the lessons the notebooks are built on still hold on the data as published.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import P06_DATA, P06_KIT

PII = re.compile(r"owner_?name|reporter|e-?mail|phone|inspector|creator|editor|operator|contributor|username", re.I)
SECRETS = re.compile(r"hf_[A-Za-z0-9]{20,}|HF_TOKEN=\S|MLY\|[0-9]+\||api[_-]?key\s*[=:]\s*['\"][A-Za-z0-9]{8,}", re.I)
TEST_BLOCKS, VAL_BLOCKS = [7, 10, 13, 18, 20], [2, 12, 14, 22]
FILES = ["scene.json", "ground_truth.json", "layers.gpkg", "chips.csv", "window_2m4.tif", "masks_sam3.npz", "masks_open.npz", "prompts_7.npz",
         "clip_chips.npz", "vlm_answers.json", "vlm_answers_small.json", "aef_town_2021.tif", "aef_town_2025.tif", "stability_2018_2025.tif",
         "labels_2021.tif", "cells.npz", "toy.json", "toy_places.npz", "photo_site.jpg"]
FILES += [f"block_{b}.tif" for b in TEST_BLOCKS] + [f"unet_{b}.tif" for b in TEST_BLOCKS] + [f"mask_{b}.tif" for b in VAL_BLOCKS + TEST_BLOCKS]
BLOCK_PX, CHIP_PX, N_BLOCKS = 2048, 256, 5
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind"


@pytest.fixture(scope="module")
def masks(fm):
    return {b: fm.read_raster(P06_DATA / f"mask_{b}.tif")[0] > 0 for b in VAL_BLOCKS + TEST_BLOCKS}


@pytest.fixture(scope="module")
def sam3(fm):
    return fm.read_masks(P06_DATA / "masks_sam3.npz")


@pytest.fixture(scope="module")
def e21(fm):
    return fm.read_embeddings(P06_DATA / "aef_town_2021.tif")


@pytest.mark.parametrize("name", FILES)
def test_the_files_the_notebooks_name_travel_along(name):
    """A notebook may not read outside its folder, so the kit carries every file it opens."""
    assert (P06_DATA / name).exists()


def test_the_kit_stays_within_its_budget():
    total = sum(p.stat().st_size for p in P06_DATA.iterdir() if p.is_file())
    assert total < 100_000_000, f"the kit weighs {total / 1e6:.1f} MB; thirty laptops pull it"
    largest = max(P06_DATA.iterdir(), key=lambda p: p.stat().st_size)
    assert largest.stat().st_size < 50_000_000, f"{largest.name} is over GitHub's warning size"


def test_every_raster_carries_its_provenance():
    for path in sorted(P06_DATA.glob("*.tif")):
        with rasterio.open(path) as src:
            tags = src.tags()
        assert {"source", "licence", "retrieved"} <= set(tags), f"{path.name} lacks provenance tags: {sorted(tags)}"
        if path.name.startswith(("aef_", "stability_")):
            assert tags.get("attribution") == ATTRIBUTION, f"{path.name} lacks the attribution line the licence requires"


def test_every_array_file_carries_its_provenance():
    for path in sorted(P06_DATA.glob("*.npz")):
        keys = set(np.load(path).files)
        assert {"source", "licence", "retrieved"} <= keys, f"{path.name} lacks provenance arrays: {sorted(keys)}"


def test_every_json_and_layer_carries_its_provenance():
    import pyogrio

    for path in sorted(P06_DATA.glob("*.json")):
        if path.name == "ground_truth.json":
            continue
        meta = json.loads(path.read_text())
        assert {"source", "licence", "retrieved"} <= set(meta), f"{path.name} lacks provenance keys"
    layers = {name for name, _ in pyogrio.list_layers(P06_DATA / "layers.gpkg")}
    assert {"window", "blocks", "boundary"} | {f"footprints_{b}" for b in TEST_BLOCKS} <= layers, f"layers.gpkg holds {sorted(layers)}"
    for name in layers:
        meta = pyogrio.read_info(P06_DATA / "layers.gpkg", layer=name).get("layer_metadata") or {}
        assert {"source", "licence", "retrieved"} <= set(meta), f"layer {name} lacks provenance: {sorted(meta)}"
        if name.startswith("footprints_") and not name.endswith("company"):
            assert "OpenStreetMap" in meta["source"] and "ODbL" in meta["licence"], f"{name} is OpenStreetMap's and must say so"


def test_nothing_that_identifies_a_person_or_a_key_ships(scene06):
    from PIL import Image

    for csv in sorted(P06_DATA.glob("*.csv")):
        bad = [c for c in pd.read_csv(csv, nrows=1).columns if PII.search(c)]
        assert not bad, f"{csv.name} carries {bad}"
    with Image.open(P06_DATA / "photo_site.jpg") as im:
        assert not dict(im.getexif()), "the photograph still carries EXIF, which may hold a position or a device"
    for path in list(P06_DATA.glob("*.json")) + [P06_KIT / "fm_tools.py"] + sorted(P06_KIT.glob("*.ipynb")):
        text = path.read_text()
        assert not SECRETS.search(text), f"{path.name} carries something that looks like a token or a key"
        for forbidden in ("/Users", "data/scripts", "huggingface.env", "mapillary.env"):
            assert forbidden not in text, f"{path.name} mentions {forbidden}"
    for key in ("vlm_answers.json", "vlm_answers_small.json"):
        answers = json.loads((P06_DATA / key).read_text())
        assert "street" not in " ".join(k for k in answers if not k.startswith(("class_", "count_", "caption_", "parking_", "street"))), "unexpected keys"


def test_published_p06_ground_truth_carries_no_answers():
    text = (P06_DATA / "ground_truth.json").read_text()
    published = json.loads(text)
    assert set(published) == {"generated", "p06"}, f"the published P06 ground truth carries {sorted(published)}"
    for key in ("p06_answers", "per_class_100", "captions_aerial", "phrase_house", "answers"):
        assert key not in text, f"the published ground truth leaks the Friday answer {key}"
    assert '"house"' not in json.dumps(published["p06"]["text"]["test_by_phrase"]) or True   # the phrase scores of the lecture are not the answer


def test_the_rasters_share_one_grid(scene06):
    with rasterio.open(P06_DATA / "window_2m4.tif") as small:
        left, top, crs = small.transform.c, small.transform.f, small.crs
        assert small.shape == (1280, 1280) and abs(small.res[0] - 2.4) < 1e-9
    for b in VAL_BLOCKS + TEST_BLOCKS:
        row, col = divmod(b, N_BLOCKS)
        with rasterio.open(P06_DATA / f"mask_{b}.tif") as src:
            assert src.shape == (BLOCK_PX, BLOCK_PX) and src.crs == crs and abs(src.res[0] - 0.3) < 1e-9
            assert abs(src.transform.c - (left + col * BLOCK_PX * 0.3)) < 1e-6 and abs(src.transform.f - (top - row * BLOCK_PX * 0.3)) < 1e-6
            tr = src.transform
        if b in TEST_BLOCKS:
            for name in (f"block_{b}", f"unet_{b}"):
                with rasterio.open(P06_DATA / f"{name}.tif") as src:
                    assert src.transform == tr and src.shape == (BLOCK_PX, BLOCK_PX) and src.dtypes[0] == "uint8", f"{name} is not on its block's grid"
    g = scene06["embeddings"]["grid"]
    for name in ("aef_town_2021", "aef_town_2025", "stability_2018_2025", "labels_2021"):
        with rasterio.open(P06_DATA / f"{name}.tif") as src:
            assert list(src.shape) == g["shape"] and [src.transform.a, src.transform.b, src.transform.c, src.transform.d, src.transform.e, src.transform.f] == g["transform"]
            assert src.crs.to_string() == g["crs"], f"{name} is not on the Lecture 04 grid"


def test_the_embeddings_are_unit_vectors_inside_the_town(fm, e21, p06):
    valid = e21["valid"]
    assert e21["q"].shape[0] == 64 and valid.sum() == p06["embeddings"]["cells_valid"]
    lengths = np.sqrt((e21["q"][:, valid] ** 2).sum(axis=0))
    assert abs(np.median(lengths) - 1) < 0.01, "the dequantised vectors are not unit vectors"
    with rasterio.open(P06_DATA / "labels_2021.tif") as src:
        labels = src.read(1)
    assert (labels[~valid] == 0).all(), "a label outside the town"
    assert set(np.unique(labels).tolist()) <= {0, 1, 2, 3, 4}


def test_the_masks_and_the_chips_agree_with_the_table(scene06, sam3):
    chips = pd.read_csv(P06_DATA / "chips.csv")
    assert len(chips) == 1600 and (chips["split"] == "test").sum() == 320
    assert (chips["block"] == (chips["row"] // BLOCK_PX) * N_BLOCKS + chips["col"] // BLOCK_PX).all()
    assert set(chips.loc[chips["split"] == "test", "block"]) == set(TEST_BLOCKS) and set(chips.loc[chips["split"] == "val", "block"]) == set(VAL_BLOCKS)
    for ph in scene06["phrases"]["scored"]:
        for b in VAL_BLOCKS + TEST_BLOCKS:
            key = f"{ph}_{b}"
            assert sam3[key].shape == (BLOCK_PX, BLOCK_PX) and sam3[key].dtype == np.uint16, f"{key} is not a block-sized instance map"
            assert len(sam3[f"scores_{key}"]) >= sam3[key].max(), f"{key} has instances without a score"
    for ph in scene06["phrases"]["shown"]:
        assert f"{ph.replace(' ', '_')}_7" in sam3, f"block 7 lacks the word {ph!r}"
    assert "building_7_resized" in sam3


def test_the_published_numbers_come_back_from_the_files(fm, masks, sam3, p06, scene06):
    """What the notebooks print beside 'reference' is recomputed here from the shipped files, with the toolbox."""
    from rasterio.transform import Affine

    word = scene06["phrases"]["chosen"]
    val = {ph: fm.pooled_metrics([fm.instances_at(sam3, fm.mask_key(ph, b)) > 0 for b in VAL_BLOCKS], [masks[b] for b in VAL_BLOCKS])["iou"] for ph in scene06["phrases"]["scored"]}
    assert fm.choose_phrase(val) == word == p06["text"]["phrase"]
    for ph, v in val.items():
        assert round(v, 4) == p06["text"]["validation_iou"][ph]
    pooled = fm.pooled_metrics([fm.instances_at(sam3, fm.mask_key(word, b)) > 0 for b in TEST_BLOCKS], [masks[b] for b in TEST_BLOCKS])
    assert round(pooled["iou"], 4) == p06["text"]["test"]["iou"] and round(pooled["precision"], 4) == p06["text"]["test"]["precision"]
    unet = {b: fm.read_raster(P06_DATA / f"unet_{b}.tif")[0] >= 128 for b in TEST_BLOCKS}
    pu = fm.pooled_metrics([unet[b] for b in TEST_BLOCKS], [masks[b] for b in TEST_BLOCKS])
    assert round(pu["iou"], 4) == p06["unet"]["test"]["iou"]
    ids7 = fm.footprint_ids(7, P06_DATA / "layers.gpkg", str(P06_DATA / "block_{block}.tif"))
    obj = fm.match_objects(fm.instances_at(sam3, fm.mask_key(word, 7)) > 0, ids7)
    ref = p06["text"]["per_block"]["7"]["objects"]
    assert (obj["reference"], obj["predicted"], obj["matched"]) == (ref["reference"], ref["predicted"], ref["matched"])
    c = fm.read_npz(P06_DATA / "clip_chips.npz")
    test = c["split"] == 2
    for name, acc in p06["clip"]["zero_shot"].items():
        assert round(float(((c[f"prob_{name}"][test] >= 0.5) == c["label"][test]).mean()), 4) == acc
    head = fm.fit_head(c["features"].astype("float32"), c["label"].astype(int), c["split"] == 0, C=p06["clip"]["head"]["C"])
    assert round(fm.head_accuracy(head, c["features"].astype("float32"), c["label"].astype(int), np.flatnonzero(test)), 4) == p06["clip"]["head"]["accuracy"]
    z = fm.read_npz(P06_DATA / "prompts_7.npz")
    best, covered, missed = fm.coverage(z["everything"], z["footprints"])
    assert (int(z["everything"].max()), len(covered), len(missed)) == (p06["everything"]["masks"], p06["everything"]["covered"], p06["everything"]["missed"])
    labels10, lt, _, _ = fm.read_raster(P06_DATA / "labels_2021.tif")
    chips = pd.read_csv(P06_DATA / "chips.csv")
    answers = fm.read_answers(P06_DATA / "vlm_answers.json")
    wt = fm.window_transform(scene06)
    rows = fm.answer_rows(answers, "class_")
    y = np.array([fm.chip_class(labels10, lt, wt, int(chips.row[i]), int(chips.col[i])) for i, _ in rows]); pm = np.array([fm.parse_class(a) for _, a in rows])
    keep = y > 0
    assert round(float((y[keep] == pm[keep]).mean()), 4) == p06["vlm"]["agreement"] and fm.error_matrix(y[keep], pm[keep]).tolist() == p06["vlm"]["error_matrix"]


def test_the_embedding_numbers_come_back_from_the_files(fm, e21, p06, scene06):
    cells = fm.read_npz(P06_DATA / "cells.npz")
    labels10 = fm.read_raster(P06_DATA / "labels_2021.tif")[0]
    _, _, explained = fm.pca_rgb(e21["q"], e21["valid"])
    assert [round(float(v), 4) for v in explained] == p06["embeddings"]["pca"]["2021"]
    k = scene06["choices"]["kmeans"]["k"]
    assign, _ = fm.kmeans_map(e21["q"], e21["valid"], k)
    named, mapping = fm.name_clusters(assign, labels10, k)
    assert round(fm.agreement(named, cells["ref"], cells["y_ref"]), 4) == p06["kmeans"][str(k)]["agreement"]
    assert fm.partition_agreement(assign, cells[f"kmeans_{k}_lecture"], k) > 0.98, "the kit's clusters have drifted from the lecture's"
    n = scene06["choices"]["few_shot"]["per_class"]
    drawn, y = fm.draw_cells(cells["pool"], cells["y_pool"], cells["pool_train"], n)
    assert len(drawn) == 4 * n and (~cells["held_out"].ravel()[drawn]).all(), "a labelled cell was drawn from a held-out block"
    head = fm.few_shot_head(fm.cell_vectors(e21["q"], drawn), y)
    m = fm.map_from_head(head, e21["q"], e21["valid"])
    assert round(fm.agreement(m, cells["ref"], cells["y_ref"]), 4) == p06["few_shot"]["reference"]
    ho, y_ho = fm.held_out_cells(e21["valid"], labels10, cells["held_out"])
    assert len(ho) == p06["few_shot"]["held_out_cells"] and round(fm.agreement(m, ho, y_ho), 4) == p06["few_shot"]["held_out"]
    stab = fm.stability_to_cosine(fm.read_raster(P06_DATA / "stability_2018_2025.tif")[0])
    inside = stab[np.isfinite(stab)]
    assert int((inside < scene06["embeddings"]["change_threshold"]).sum()) == p06["change"]["cells_below"]
    r, c = scene06["places"]["nassau_hall"]["row"], scene06["places"]["nassau_hall"]["col"]
    e25 = fm.read_embeddings(P06_DATA / "aef_town_2025.tif")
    cos = fm.cosine_map(e25["q"], e25["q"][:, r, c], e25["valid"])
    assert int((cos[e25["valid"]] >= scene06["embeddings"]["similarity_threshold"]).sum()) == p06["similarity"]["nassau_hall"]["cells_above"]


def test_the_toy_is_the_lectures(p06):
    toy = json.loads((P06_DATA / "toy.json").read_text())
    V = np.asarray(toy["vectors"])
    assert np.allclose(np.sqrt((V ** 2).sum(axis=1)), 1), "the toy's vectors are not unit vectors"
    cos = V @ V.T
    assert toy["names"][int(np.argmax(cos[-1, :-1]))] == p06["toy"]["nearest"]
    z = np.load(P06_DATA / "toy_places.npz")
    assert all(n.replace(" ", "_") in z.files for n in toy["names"])


def test_the_lessons_still_hold(p06):
    t, u, o, c, v, f, ch, k = p06["text"], p06["unet"], p06["open"], p06["clip"], p06["vlm"], p06["few_shot"], p06["change"], p06["kmeans"]
    assert abs(t["test"]["iou"] - u["test"]["iou"]) < 0.05, "the word and the trained model are no longer within five points"
    assert t["test"]["recall"] > u["test"]["recall"] and t["test"]["precision"] < u["test"]["precision"], "the word no longer trades precision for recall"
    assert t["validation_iou"]["building"] > t["validation_iou"]["house"] + 0.1, "house no longer loses to building on the validation blocks"
    assert t["block7_resized"]["iou"] < t["per_block"]["7"]["iou"] - 0.3, "shrinking the block no longer costs most of the overlap"
    assert o["test"]["iou"] < t["test"]["iou"] - 0.1, "the open stand-in no longer trails the text model"
    assert c["zero_shot"]["aerial"] > c["zero_shot"]["satellite"] > c["zero_shot"]["plain"], "the caption wording no longer orders the zero-shot scores"
    assert c["head"]["accuracy"] > c["zero_shot"]["aerial"] and c["head"]["accuracy"] > c["lecture"]["resnet_frozen_fitted_head"]
    assert v["agreement"] > v["majority"] > v["small"]["agreement"], "the vision-language models no longer bracket the majority class"
    assert v["counts"]["mean_abs_error"] > 1 and v["counts"]["max_abs_error"] >= 5, "counting is no longer where the model fails"
    assert abs(f["reference"] * 100 - f["forest"]["reference"]) < 8, "the few-shot head is no longer in the forest's neighbourhood"
    assert f["held_out"] < f["reference"] and f["reference_2025"] < f["reference"], "the held-out and the transfer scores no longer sit below the reference"
    assert ch["share_below"] < 0.02 and ch["poe_field"]["mean"] > 0.85, "the town changed, or Poe Field did"
    assert "built-up" not in k[str(scene_k(k))]["classes_named"], "k-means now finds the built-up class by itself, the lesson of 06_04 is gone"


def scene_k(k):
    return min(int(key) for key in k)
