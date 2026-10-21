"""The toolbox of Precept 06, Zero-Shot City.

Every algorithm the four notebooks use lives here once, documented, so that a code cell in a notebook is a few
lines that call a named tool: the prompts to the open segment-anything model, the scoring of a prompted mask
against the volunteers' footprints with the metrics of Precept 05, the captions of CLIP and the answers of a
vision-language model held against the labels of Lecture 04, and the three operations on the satellite
embeddings of the town, similarity search, clustering and few-shot mapping. The course pipeline that computes
the reference numbers imports the same file, so the notebooks and their numbers cannot drift apart.

The model section needs PyTorch and the transformers library. Both are imported inside the functions that use
them, so the module loads without them and every tool that works on arrays, rasters and polygons keeps working.
`torch_available()` says whether the deep learning library is there, `model_ready(name)` whether a model's weights
can be loaded (they are cached in the container, and downloaded once on the uv path), and `NO_TORCH` and
`NO_MODEL` are the one line a notebook prints when a cell is skipped.

Sections: loading, the models, prompts and masks, scoring, words about pictures, embeddings, plotting.
"""

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

# a kernel without the widget library prints a progress-bar warning with a local path when the model libraries load
warnings.filterwarnings("ignore", message="IProgress not found")

__all__ = [
    # loading
    "load_scene", "read_raster", "read_rgb", "read_npz", "chip_table", "block_chips", "cut_chips", "read_embeddings",
    "dequantise", "read_masks", "mask_key", "instances_at", "count_instances", "read_answers", "load_image", "first_image", "downsample",
    "window_transform", "footprint_ids",
    # the models
    "torch_available", "transformers_available", "weights_cached", "model_ready", "NO_TORCH", "NO_MODEL", "MODELS",
    "load_sam", "count_weights", "embed_image", "prompt", "load_clip", "clip_image_features", "clip_text_features",
    "logit_scale", "zero_shot", "fit_head", "head_accuracy",
    # prompts and masks
    "write_mask", "polygonise", "coverage", "instance_colours", "overlay_instances",
    # scoring
    "confusion", "pixel_metrics", "iou", "chip_scores", "rasterise", "match_objects", "agreement_rgb", "block_metrics",
    "pooled_metrics", "choose_phrase", "where_each_wins",
    # words about pictures
    "parse_class", "parse_count", "chip_class", "error_matrix", "count_footprints", "answer_rows",
    # embeddings
    "SEED", "CLASSES", "CLASS_NAMES", "CLASS_COLOURS", "cell_of_xy", "xy_of_cell", "cell_vectors", "pca_rgb", "cosine_map",
    "top_cells", "kmeans_map", "name_clusters", "partition_agreement", "agreement", "agreement_by_class", "draw_cells", "few_shot_head",
    "map_from_head", "change_score", "stability_to_cosine", "held_out_cells", "extent_of",
    # plotting
    "stretch", "show_image", "show_window", "show_mask", "show_masks", "show_instances", "show_agreement", "show_objects",
    "show_chips", "show_matrix", "show_error_matrix", "show_counts", "show_map", "show_maps", "show_vectors", "show_toy",
    "show_similarity", "mark_cells", "show_curve", "class_cmap", "km_ticks",
]

SEED = 20261026                    # the lecture's seed, the date of Lecture 06
CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {1: "tree cover", 2: "grass", 3: "built-up", 4: "water"}
CLASS_COLOURS = {"tree cover": "#2f7d32", "grass": "#a8d05f", "built-up": "#b0413e", "water": "#3b6fb6"}
MODELS = {"sam2": "facebook/sam2.1-hiera-tiny", "clip": "openai/clip-vit-base-patch16", "vlm": "HuggingFaceTB/SmolVLM-500M-Instruct"}
NO_TORCH = ("PyTorch is not installed in this environment, so this cell is skipped; the masks and numbers it would "
            "compute are read from the shipped files instead (see docs/troubleshooting.org for the environment update).")
NO_MODEL = ("The model cannot be loaded here, because the transformers library or its cached weights are missing and the "
            "hub cannot be reached, so this cell is skipped; the masks and numbers it would compute are read from the "
            "shipped files instead (the container ships the weights, the uv path downloads them once).")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_scene(path="data/scene.json"):
    """The kit's scene.json as a dict: the window and its blocks, the prompts, the phrases and captions, the places, the
    lecture's numbers and the choices the pipeline made."""
    return json.loads(Path(path).read_text())


def read_raster(path):
    """A GeoTIFF as (array, transform, crs, resolution in metres): bands first for a multi-band file, a 2-D array for
    one band. The tags travel with the file; open it with rasterio to read them."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read()
        return (arr if src.count > 1 else arr[0]), src.transform, src.crs, float(src.res[0])


def read_rgb(path):
    """A three-band GeoTIFF as a stretched (H, W, 3) picture plus its transform."""
    arr, transform, crs, res = read_raster(path)
    return stretch(arr), transform


def read_npz(path):
    """A shipped npz as a dict of arrays; the zero-dimensional string arrays (the provenance) become strings."""
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        a = z[k]
        out[k] = str(a) if a.dtype.kind in "US" and a.ndim == 0 else a
    return out


def chip_table(path="data/chips.csv"):
    """The 1,600 tiled chips of the window as a table: row and column of the chip's north-west corner in the window's
    pixels, its block, its split, its roof share and its label."""
    import pandas as pd
    return pd.read_csv(path)


def block_chips(table, block):
    """The rows of one block, in the order the table has them."""
    return table[table["block"] == block].reset_index(drop=True)


def cut_chips(image, rows, cols, origin=(0, 0), size=256):
    """The chips of a (3, H, W) image at the given window rows and columns, as (N, 3, size, size); `origin` is the
    window row and column of the image's first pixel, for a block cut out of the window."""
    r0, c0 = origin
    return np.stack([image[:, r - r0:r - r0 + size, c - c0:c - c0 + size] for r, c in zip(rows, cols)])


def dequantise(arr):
    """The published bytes of an embedding back to numbers: ((v / 127.5) ** 2) * sign(v), which gives unit vectors;
    the byte minus 128 marks a cell without data. Returns the vectors as float32 and the nodata mask."""
    v = np.asarray(arr).astype(np.float32)
    return ((v / 127.5) ** 2) * np.sign(v), np.asarray(arr)[0] == -128


def read_embeddings(path):
    """A shipped embedding raster: a dict with `q` (64, rows, cols) float32 unit vectors, `valid` (rows, cols), the
    transform, the CRS, the resolution and the tags, among them the required attribution line."""
    import rasterio
    with rasterio.open(path) as src:
        q, nodata = dequantise(src.read())
        return dict(q=q, valid=~nodata, transform=src.transform, crs=src.crs, res=float(src.res[0]), tags=src.tags())


def read_masks(path):
    """A shipped file of instance masks: a dict with one uint16 map per key (0 is background, every other value one
    instance), the scores of the instances under `scores_<key>`, and the provenance strings."""
    return read_npz(path)


def mask_key(phrase, block, resized=False):
    """The key of a phrase and a block in a shipped mask file: the phrase with underscores, the block, and `_resized`
    for the pass that shrank the whole block to the model's input."""
    return f"{phrase.replace(' ', '_')}_{block}" + ("_resized" if resized else "")


def instances_at(masks, key, threshold=None):
    """The instance map of `key` with every instance scoring below `threshold` removed (none removed when the threshold
    is None, the text model's own cut)."""
    inst = np.asarray(masks[key])
    scores = np.asarray(masks.get(f"scores_{key}", []))
    if threshold is None or len(scores) == 0:
        return inst
    keep = np.flatnonzero(scores >= threshold) + 1
    return np.where(np.isin(inst, keep), inst, 0)


def count_instances(masks, key, threshold=None):
    """How many instances a shipped map holds once the ones below the threshold are removed."""
    return int(len(np.unique(instances_at(masks, key, threshold))) - 1)


def window_transform(scene):
    """The transform of the 30 cm window from the scene file: its origin and its resolution."""
    from rasterio.transform import Affine
    x, y = scene["window"]["origin"]
    res = scene["window"]["resolution_m"]
    return Affine(res, 0.0, x, 0.0, -res, y)


def footprint_ids(block, layers="data/layers.gpkg", image="data/block_{block}.tif", layer="footprints_{block}"):
    """The footprints of a block burnt onto its grid with their ids, the reference objects of the object metrics."""
    import geopandas as gpd
    arr, transform, crs, res = read_raster(image.format(block=block))
    return rasterise(gpd.read_file(layers, layer=layer.format(block=block)), transform, arr.shape[-2:])


def read_answers(path):
    """A shipped file of a vision-language model's answers, keyed by question and chip, with `_run` naming the model."""
    return json.loads(Path(path).read_text())


def load_image(path):
    """A picture for a prompt: a JPEG or PNG through Pillow (turned the way its EXIF says), or a GeoTIFF through rasterio
    (its first three bands, stretched to 8 bits when they are not). Returns the (H, W, 3) uint8 picture, the transform
    and the CRS, the last two None for a plain picture."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()[:3]
            transform, crs = src.transform, src.crs
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        pic = arr if arr.dtype == np.uint8 else (stretch(arr) * 255).astype("uint8")
        return np.ascontiguousarray(np.moveaxis(pic, 0, -1)), transform, crs
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        pic = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return pic, None, None


def first_image(folder="my_images", fallback="data/photo_site.jpg"):
    """The first picture in a folder (JPEG, PNG or GeoTIFF, by name), or the kit's own photograph when the folder holds
    none."""
    found = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    return found[0] if found else Path(fallback)


def downsample(arr, factor):
    """A (H, W) array averaged in blocks of `factor` pixels, for drawing a block's mask on the window at 2.4 m."""
    a = np.asarray(arr, dtype="float32")
    H, W = a.shape
    return a[:H - H % factor, :W - W % factor].reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))


# --------------------------------------------------------------------------- #
# the models: the guards, SAM 2.1, CLIP, a head on features
# --------------------------------------------------------------------------- #

def torch_available():
    """True when torch imports, False otherwise. Printing `NO_TORCH` is the notebooks' answer to False."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def transformers_available():
    """True when the transformers library imports, the library that holds the loaders of SAM 2.1 and CLIP."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def weights_cached(name):
    """Whether the weights of a model (`sam2`, `clip` or `vlm`) sit in the local Hugging Face cache, the container's
    or the one a first run on the uv path filled."""
    try:
        from huggingface_hub import scan_cache_dir
        repos = scan_cache_dir().repos
    except Exception:  # noqa: BLE001  a missing or unreadable cache is an empty one
        return False
    for repo in repos:
        if repo.repo_id == MODELS[name]:
            files = {f.file_name for rev in repo.revisions for f in rev.files}
            return "config.json" in files and bool(files & {"model.safetensors", "pytorch_model.bin"})
    return False


def model_ready(name):
    """True when the model `name` can be loaded here: torch and transformers import, and the weights are cached or the
    hub can be reached to download them once (150 MB for SAM 2.1 tiny, 600 MB for CLIP). Printing `NO_MODEL` is the
    notebooks' answer to False."""
    if not torch_available() or not transformers_available():
        return False
    if weights_cached(name):
        return True
    if os.environ.get("HF_HUB_OFFLINE", "0").lower() not in ("", "0", "false"):
        return False
    try:
        from huggingface_hub import model_info
        model_info(MODELS[name], timeout=5)
        return True
    except Exception:  # noqa: BLE001  no network, no download
        return False


def _quiet():
    import logging as std_logging
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging
    std_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(std_logging.ERROR)   # a Mac notice
    hub_logging.set_verbosity_error()          # no line about unauthenticated requests on a laptop that is online
    logging.set_verbosity_error()
    logging.disable_progress_bar()


def _pil(arr):
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(_hwc(arr)))


def _hwc(arr):
    a = np.asarray(arr)
    return np.moveaxis(a, 0, -1) if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3 else a


def load_sam(device="cpu"):
    """SAM 2.1 hiera-tiny (Meta, Apache 2.0) with its processor, in evaluation mode on the CPU unless told otherwise.
    Returns the pair the prompt functions take."""
    _quiet()
    from transformers import Sam2Model, Sam2Processor
    model = Sam2Model.from_pretrained(MODELS["sam2"]).to(device).eval()
    return model, Sam2Processor.from_pretrained(MODELS["sam2"])


def count_weights(model):
    """The number of weights of a loaded model."""
    return int(sum(p.numel() for p in model.parameters()))


def embed_image(sam, image):
    """The image encoder run once over an (H, W, 3) uint8 picture, the costly half of the model. Returns what `prompt`
    needs (the embedding, the picture and its size) with the seconds it took; every prompt on the picture reuses it."""
    import torch
    model, proc = sam
    pic = _pil(image)
    t0 = time.time()
    with torch.no_grad():
        inputs = proc(images=pic, return_tensors="pt")
        emb = model.get_image_embeddings(inputs["pixel_values"].to(model.device))
    return dict(embeddings=emb, image=pic, shape=tuple(np.asarray(_hwc(image)).shape[:2]), seconds=time.time() - t0)


def prompt(sam, embedded, points=None, labels=None, box=None, multimask=True):
    """A prompt through the light half of the model on an embedded picture: `points` as (x, y) pairs with `labels` 1
    for the object and 0 for the background, or a `box` as (x0, y0, x1, y1), in the picture's pixels. Returns the masks
    as a (k, H, W) boolean array (three when `multimask`, one otherwise) and the model's predicted overlap of each."""
    import torch
    model, proc = sam
    kw = {}
    if points is not None:
        kw["input_points"] = [[[list(map(float, p)) for p in points]]]
        kw["input_labels"] = [[[int(v) for v in labels]]]
    if box is not None:
        kw["input_boxes"] = [[list(map(float, box))]]
    with torch.no_grad():
        inputs = proc(images=embedded["image"], return_tensors="pt", **kw)
        given = {k: v.to(model.device) for k, v in inputs.items() if k in ("input_points", "input_labels", "input_boxes")}
        out = model(image_embeddings=embedded["embeddings"], multimask_output=multimask, **given)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
    return masks.astype(bool), out.iou_scores.float().cpu().numpy().ravel()


def load_clip(device="cpu"):
    """CLIP ViT-B/16 (OpenAI, MIT) with its processor, in evaluation mode on the CPU unless told otherwise."""
    _quiet()
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(MODELS["clip"]).to(device).eval()
    return model, CLIPProcessor.from_pretrained(MODELS["clip"])


def clip_image_features(clip, images, batch=32):
    """The 512 numbers CLIP's image tower gives every picture, unit length, (N, 512) float32, for (N, 3, H, W) uint8
    chips or a list of (H, W, 3) pictures: the vision transformer, its pooled token and the projection."""
    import torch
    model, proc = clip
    pics = [_pil(im) for im in images]
    out = []
    with torch.no_grad():
        for i in range(0, len(pics), batch):
            pv = proc(images=pics[i:i + batch], return_tensors="pt").pixel_values.to(model.device)
            f = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
            out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out)


def clip_text_features(clip, captions):
    """The 512 numbers CLIP's text tower gives every caption, unit length, (M, 512) float32, in the same space as the
    pictures' numbers."""
    import torch
    model, proc = clip
    with torch.no_grad():
        t = proc(text=list(captions), return_tensors="pt", padding=True).to(model.device)
        f = model.text_projection(model.text_model(input_ids=t.input_ids, attention_mask=t.attention_mask).pooler_output)
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()


def logit_scale(clip):
    """The learned scale CLIP multiplies the cosines by before the softmax, the inverse of a temperature."""
    return float(clip[0].logit_scale.detach().exp())


def zero_shot(features, text_features, scale=100.0):
    """The probability of every caption for every picture: the cosines times the scale, softmax over the captions.
    The first caption's probability at or above one half is the zero-shot answer of the lecture."""
    logits = scale * np.asarray(features, np.float32) @ np.asarray(text_features, np.float32).T
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fit_head(features, labels, train, C=0.1, seed=SEED):
    """A logistic regression on standardised features, fitted on the rows where `train` is True: a head of 513 weights
    on the 512 numbers, with a penalty (`C`, smaller is stronger) chosen on the validation chips in the lecture."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C, random_state=seed))
    return head.fit(np.asarray(features)[train], np.asarray(labels)[train])


def head_accuracy(head, features, labels, idx):
    """The share of the chips at `idx` the head gets right."""
    return float((head.predict(np.asarray(features)[idx]) == np.asarray(labels)[idx]).mean())


# --------------------------------------------------------------------------- #
# prompts and masks
# --------------------------------------------------------------------------- #

def polygonise(mask, transform, tolerance=0.6):
    """The blobs of a boolean mask as shapely polygons in the raster's system, simplified to `tolerance` metres."""
    import rasterio.features
    from shapely.geometry import shape
    m = np.asarray(mask, dtype=bool)
    found = rasterio.features.shapes(m.astype("uint8"), mask=m, transform=transform)
    return [shape(g).simplify(tolerance) for g, v in found if v == 1]


def write_mask(mask, path, transform=None, crs=None):
    """A mask written beside the notebook: as GeoJSON in WGS 84 when the picture was georeferenced (the mask's polygons
    in the raster's system, reprojected), else as a PNG of the same size. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is not None and crs is not None:
        import geopandas as gpd
        polys = polygonise(mask, transform, tolerance=0)
        g = gpd.GeoDataFrame({"id": list(range(1, len(polys) + 1))}, geometry=polys, crs=crs).to_crs("EPSG:4326")
        out = path.with_suffix(".geojson")
        g.to_file(out, driver="GeoJSON")
        return out
    from PIL import Image
    out = path.with_suffix(".png")
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype("uint8")).save(out)
    return out


def coverage(instances, footprints, threshold=0.5):
    """For every footprint of a crop, its best overlap with any instance mask; covered when that reaches the threshold.
    Returns the overlaps by footprint id, the covered ids and the missed ids."""
    every = np.asarray(instances)
    areas = np.bincount(every.ravel(), minlength=int(every.max()) + 1)
    best = {}
    for v in np.unique(footprints[footprints > 0]):
        sel = footprints == v
        n_sel = int(sel.sum())
        labels = np.unique(every[sel])
        b = 0.0
        for k in labels[labels > 0]:
            inter = int((sel & (every == k)).sum())
            b = max(b, inter / (n_sel + areas[k] - inter))
        best[int(v)] = b
    covered = [v for v, b in best.items() if b >= threshold]
    return best, covered, [v for v, b in best.items() if b < threshold]


def instance_colours(n, seed=SEED):
    """n distinct pastel colours for instance masks, seeded, the lecture's."""
    from matplotlib.colors import hsv_to_rgb
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875 + rng.random()) % 1.0
    return np.array([hsv_to_rgb((h, 0.55, 0.95)) for h in hues])


def overlay_instances(picture, instances, alpha=0.55, seed=SEED):
    """Instance ids painted in distinct colours over the faded picture, as an (H, W, 3) image in 0 to 1."""
    out = 0.45 + 0.55 * np.asarray(picture, dtype="float32")
    n = int(np.asarray(instances).max())
    if n:
        cols = instance_colours(n, seed)
        sel = instances > 0
        out[sel] = (1 - alpha) * out[sel] + alpha * cols[instances[sel] - 1]
    return out


# --------------------------------------------------------------------------- #
# scoring, the metrics of Precept 05 on a prompted mask
# --------------------------------------------------------------------------- #

def confusion(pred, ref):
    """The four counts of two boolean masks: tp, fp, fn, tn."""
    pred = np.asarray(pred, dtype=bool); ref = np.asarray(ref, dtype=bool)
    return dict(tp=int((pred & ref).sum()), fp=int((pred & ~ref).sum()), fn=int((~pred & ref).sum()), tn=int((~pred & ~ref).sum()))


def pixel_metrics(cm):
    """Accuracy, precision, recall, the IoU of the roof class, Dice, the background IoU, the mean IoU, the share of
    pixels a model saying no building would get right, and the pixel count, from the four counts."""
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    iou_roof = tp / max(tp + fp + fn, 1)
    iou_back = tn / max(tn + fp + fn, 1)
    return dict(accuracy=(tp + tn) / max(n, 1), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                iou=iou_roof, dice=2 * tp / max(2 * tp + fp + fn, 1), iou_background=iou_back,
                mean_iou=(iou_roof + iou_back) / 2, no_building_accuracy=(fp + tn) / max(n, 1), n=n)


def iou(pred, ref):
    """The intersection over union of two boolean masks, the roof pixels both share over the roof pixels either has."""
    return pixel_metrics(confusion(pred, ref))["iou"]


def chip_scores(pred, mask, size=256):
    """The score of every tiled chip of a boolean prediction against its mask: a table with the chip's row and column
    in the map, its roof share, its share of pixels right and its IoU (None for a chip without a roof pixel and
    without a predicted one), plus the pooled metrics over all the pixels."""
    H, W = mask.shape
    rows = []
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            p = np.asarray(pred[r:r + size, c:c + size], bool); m = mask[r:r + size, c:c + size] > 0
            cm = confusion(p, m); pm = pixel_metrics(cm)
            defined = cm["tp"] + cm["fp"] + cm["fn"] > 0
            rows.append(dict(row=r, col=c, share=float(m.mean()), accuracy=pm["accuracy"], iou=pm["iou"] if defined else None, **cm))
    return rows, pixel_metrics(confusion(pred, mask > 0))


def rasterise(footprints, transform, shape, column="bid"):
    """The footprints burnt onto a grid: the value of `column` on every pixel whose centre lies inside one, 0 elsewhere."""
    import rasterio.features
    shapes = ((g, int(v)) for g, v in zip(footprints.geometry, footprints[column]))
    return rasterio.features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def match_objects(pred, ids, threshold=0.5, min_pred_px=222, min_ref_px=111):
    """Objects matched one to one by IoU, as the lectures count them. Reference objects are the footprints of `ids`
    (at least `min_ref_px` pixels), predicted objects the 8-connected blobs of the boolean `pred` (at least
    `min_pred_px` pixels). Pairs are taken in descending IoU, each object used once, and a pair counts as a match at
    or above `threshold`. Returns the counts, precision and recall per object, the IoU of every reference object
    (0 when unmatched), the labelled blobs and the set of matched blobs."""
    from scipy import ndimage
    pred_ids, n_pred = ndimage.label(np.asarray(pred, dtype=bool), structure=np.ones((3, 3)))
    pred_area = np.bincount(pred_ids.ravel(), minlength=n_pred + 1)
    small = pred_area < min_pred_px
    small[0] = False
    if small.any():
        pred_ids[np.isin(pred_ids, np.flatnonzero(small))] = 0
    keep_pred = {k for k in range(1, n_pred + 1) if not small[k]}
    ref_area = np.bincount(ids.ravel().astype("int64"), minlength=int(ids.max()) + 1)
    keep_ref = {k for k in range(1, len(ref_area)) if ref_area[k] >= min_ref_px}
    both = (ids > 0) & (pred_ids > 0)
    keys, counts = np.unique(ids.astype("int64")[both] * (n_pred + 1) + pred_ids.astype("int64")[both], return_counts=True)
    pairs = []
    for key, inter in zip(keys, counts):
        rk, pk = divmod(int(key), n_pred + 1)
        if rk in keep_ref and pk in keep_pred:
            pairs.append((inter / (ref_area[rk] + pred_area[pk] - inter), rk, pk))
    pairs.sort(reverse=True)
    best, used = {}, set()
    for v, rk, pk in pairs:
        if rk not in best and pk not in used:
            best[rk] = (float(v), pk); used.add(pk)
    matched = {pk for rk, (v, pk) in best.items() if v >= threshold}
    m = len(matched)
    return dict(reference=len(keep_ref), predicted=len(keep_pred), matched=m, precision=m / max(len(keep_pred), 1),
                recall=m / max(len(keep_ref), 1), ref_iou={int(rk): (best[rk][0] if rk in best else 0.0) for rk in keep_ref},
                pred_ids=pred_ids, matched_pred=matched)


AGREEMENT = dict(tp=(0.20, 0.60, 0.25), fp=(0.85, 0.25, 0.20), fn=(0.15, 0.35, 0.75))


def agreement_rgb(pred, ref, image=None):
    """A picture of agreement: the roof found in green, false alarms in red, roof missed in blue, over the faded
    image (or white)."""
    pred = np.asarray(pred, dtype=bool); ref = np.asarray(ref, dtype=bool)
    out = np.ones(pred.shape + (3,)) if image is None else 0.35 + 0.65 * np.asarray(image)
    out[pred & ref] = AGREEMENT["tp"]; out[pred & ~ref] = AGREEMENT["fp"]; out[~pred & ref] = AGREEMENT["fn"]
    return out


def block_metrics(pred, mask, ids=None, threshold=0.5):
    """One block scored: the pixel metrics of a boolean prediction against the mask, and, when the footprint ids are
    given, the objects matched at the threshold."""
    out = pixel_metrics(confusion(pred, mask > 0))
    if ids is not None:
        o = match_objects(pred, ids, threshold)
        out["objects"] = dict(reference=o["reference"], predicted=o["predicted"], matched=o["matched"], precision=o["precision"],
                              recall=o["recall"])
    return out


def pooled_metrics(preds, masks):
    """The pixel metrics over several blocks pooled, every pixel counted once, the lecture's number for the test blocks."""
    cm = dict(tp=0, fp=0, fn=0, tn=0)
    for p, m in zip(preds, masks):
        c = confusion(p, m > 0)
        for k in cm:
            cm[k] += c[k]
    return pixel_metrics(cm)


def choose_phrase(scores):
    """The phrase with the highest pooled IoU, from a dict of phrase to score computed on the validation blocks alone;
    the test blocks never take part in the choice."""
    return max(scores, key=scores.get)


def where_each_wins(word, unet, mask, chips, origin, size=256):
    """Among the chips of a block that hold a building, the chip where the word's IoU beats the trained model's by most
    and the chip where the trained model beats the word by most; `chips` is the block's rows of the chip table and
    `origin` the window row and column of the block. Returns the two rows with both IoUs."""
    rows = []
    for _, c in chips[chips["label"] == 1].iterrows():
        r, k = int(c["row"]) - origin[0], int(c["col"]) - origin[1]
        sl = (slice(r, r + size), slice(k, k + size))
        rows.append(dict(row=r, col=k, share=float(c["share"]), unet=iou(unet[sl], mask[sl] > 0), word=iou(word[sl], mask[sl] > 0)))
    return max(rows, key=lambda d: d["word"] - d["unet"]), max(rows, key=lambda d: d["unet"] - d["word"]), len(rows)


# --------------------------------------------------------------------------- #
# words about pictures: the answers of a vision-language model against the labels
# --------------------------------------------------------------------------- #

NAMES_TO_CLASS = {"tree": 1, "forest": 1, "wood": 1, "grass": 2, "lawn": 2, "field": 2, "built": 3, "building": 3, "urban": 3,
                  "house": 3, "water": 4, "lake": 4, "river": 4}


def parse_class(answer):
    """The class (1 to 4) a free-text answer names, by the first of the lecture's key words it contains; 0 when none."""
    a = str(answer).lower()
    for key, cl in NAMES_TO_CLASS.items():
        if key in a:
            return cl
    return 0


def parse_count(answer):
    """The first whole number in a free-text answer, or None."""
    import re
    m = re.search(r"\d+", str(answer).replace(",", ""))
    return int(m.group()) if m else None


def chip_class(labels, label_transform, window_transform, row, col, size=256):
    """The majority class among the 10 m label cells under a chip of the 30 cm window, sampled at nine points, the
    lecture's rule for the class question; 0 when no cell is labelled."""
    classes = []
    for dr in (size // 8, size // 2, size - size // 8):
        for dc in (size // 8, size // 2, size - size // 8):
            x, y = window_transform @ (col + dc, row + dr)
            c, r = ~label_transform @ (x, y)
            r, c = int(r), int(c)
            if 0 <= r < labels.shape[0] and 0 <= c < labels.shape[1] and labels[r, c] > 0:
                classes.append(int(labels[r, c]))
    if not classes:
        return 0
    vals, counts = np.unique(classes, return_counts=True)
    return int(vals[np.argmax(counts)])


def error_matrix(y_true, y_pred, classes=CLASSES):
    """The error matrix of Lecture 04: rows the reference classes, columns the answers."""
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            m[i, j] = int(np.sum((np.asarray(y_true) == a) & (np.asarray(y_pred) == b)))
    return m


def count_footprints(ids, min_px=100):
    """How many footprints have at least `min_px` pixels inside a chip's id map, the lecture's reference count."""
    vals, counts = np.unique(ids[ids > 0], return_counts=True)
    return int((counts >= min_px).sum())


def answer_rows(answers, prefix):
    """The (chip, answer) pairs of one question of a shipped answers file, in file order."""
    return [(int(a["chip"]), a["answer"]) for key, a in answers.items() if key.startswith(prefix)]


# --------------------------------------------------------------------------- #
# embeddings: a vector for every place
# --------------------------------------------------------------------------- #

def cell_of_xy(transform, x, y):
    """Row and column of a point in the raster's system."""
    c, r = ~transform @ (x, y)
    return int(r), int(c)


def xy_of_cell(transform, row, col):
    """The centre of a cell in the raster's system."""
    x, y = transform @ (col + 0.5, row + 0.5)
    return float(x), float(y)


def cell_vectors(q, cells):
    """The vectors of flat cell indices on the grid, (n, 64)."""
    rows, cols = np.unravel_index(np.asarray(cells), q.shape[1:])
    return q[:, rows, cols].T


def pca_rgb(q, valid, seed=SEED, sample=60000):
    """The 64 numbers of every cell as three colours: a principal component analysis fitted on a seeded sample of the
    valid cells, every cell projected on the first three components, each stretched between its 2nd and 98th
    percentile; white outside. Returns the picture as uint8, the components and the variance each explains."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(np.asarray(valid).ravel())
    chosen = rng.choice(idx, min(sample, len(idx)), replace=False)
    X = q.reshape(64, -1).T
    pca = PCA(n_components=3, random_state=seed).fit(X[chosen])
    comp = pca.transform(X).reshape(q.shape[1], q.shape[2], 3)
    inside = comp.reshape(-1, 3)[np.asarray(valid).ravel()]
    lo, hi = np.percentile(inside, 2, axis=0), np.percentile(inside, 98, axis=0)
    rgb = np.clip((comp - lo) / (hi - lo), 0, 1)
    rgb[~np.asarray(valid)] = 1.0
    return (rgb * 255).astype(np.uint8), pca.components_.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32)


def cosine_map(q, vector, valid=None):
    """The cosine of every cell's vector with one vector, as a (rows, cols) map; not a number outside the valid cells."""
    cos = np.einsum("k,kij->ij", np.asarray(vector, np.float32), q).astype(np.float32)
    if valid is not None:
        cos[~np.asarray(valid)] = np.nan
    return cos


def top_cells(cos, n=10, min_distance=0, exclude=None):
    """The rows and columns of the n cells with the largest value, taken in order and at least `min_distance` cells
    apart from every cell already taken and from the `exclude` cell (the query itself, usually)."""
    flat = np.where(np.isnan(cos), -2, cos).ravel()
    order = np.argsort(-flat)
    rows, cols = ([exclude[0]], [exclude[1]]) if exclude is not None else ([], [])
    for i in order:
        r, c = divmod(int(i), cos.shape[1])
        if all((r - rr) ** 2 + (c - cc) ** 2 >= max(min_distance, 1) ** 2 for rr, cc in zip(rows, cols)):
            rows.append(r); cols.append(c)
        if len(rows) == n + (1 if exclude is not None else 0):
            break
    skip = 1 if exclude is not None else 0
    return np.array(rows[skip:]), np.array(cols[skip:])


def kmeans_map(q, valid, k, seed=SEED, sample=120000):
    """k-means on a seeded sample of the valid cells, then every cell assigned to its nearest centre: the lecture's
    recipe with the same sample and seed, run on one thread so that every laptop gets the same map (the lecture's own
    run used ten threads and differs on a few cells in a thousand). Returns the assignment (minus one outside) and
    the centres."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(np.asarray(valid).ravel())
    chosen = rng.choice(idx, min(sample, len(idx)), replace=False)
    X = q.reshape(64, -1).T
    with threadpool_limits(limits=1):          # the sums inside k-means follow the thread count; one thread is the same everywhere
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X[chosen])
    assign = km.predict(X).reshape(q.shape[1:]).astype(np.int16)
    assign[~np.asarray(valid)] = -1
    return assign, km.cluster_centers_.astype(np.float32)


def name_clusters(assign, labels, k):
    """Each cluster named by the majority label among its labelled cells; returns the named map and the mapping."""
    mapping = {}
    for j in range(k):
        sel = (assign == j) & (labels > 0)
        if sel.any():
            vals, counts = np.unique(labels[sel], return_counts=True)
            mapping[j] = int(vals[np.argmax(counts)])
        else:
            mapping[j] = 0
    matched = np.zeros(labels.shape, dtype=np.uint8)
    for j, cl in mapping.items():
        matched[assign == j] = cl
    return matched, mapping


def partition_agreement(a, b, k):
    """The share of cells on which two cluster maps of k clusters agree once their cluster numbers are matched one to
    one as well as possible, since k-means numbers its clusters arbitrarily; cells outside (minus one) are left out."""
    from scipy.optimize import linear_sum_assignment
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    inside = (a >= 0) & (b >= 0)
    m = np.zeros((k, k), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            m[i, j] = int(((a == i) & (b == j) & inside).sum())
    rows, cols = linear_sum_assignment(-m)
    return float(m[rows, cols].sum() / max(inside.sum(), 1))


def agreement(pred_map, cells, y):
    """The share of the given cells (flat indices) on which a class map agrees with their labels."""
    return float((np.asarray(pred_map).ravel()[np.asarray(cells)] == np.asarray(y)).mean())


def agreement_by_class(pred_map, cells, y, classes=CLASSES):
    """The share of each class's cells the map gets right, the producer's accuracy of Lecture 04, with the counts."""
    p = np.asarray(pred_map).ravel()[np.asarray(cells)]; y = np.asarray(y)
    return {CLASS_NAMES[c]: (float(((p == y) & (y == c)).sum() / max((y == c).sum(), 1)), int((y == c).sum())) for c in classes}


def draw_cells(pool, y_pool, allowed, n, seed=SEED, draw=0, classes=CLASSES):
    """n cells per class drawn from the balanced pool among the cells `allowed` (not in a held-out block), with the
    lecture's generator for the draw. Returns the cells (flat indices) and their labels."""
    rng = np.random.default_rng(seed + 1000 * n + draw)
    take = []
    for c in classes:
        cand = np.flatnonzero((np.asarray(y_pool) == c) & np.asarray(allowed))
        take.extend(rng.choice(cand, min(n, len(cand)), replace=False).tolist())
    take = np.array(take)
    return np.asarray(pool)[take], np.asarray(y_pool)[take]


def few_shot_head(X, y, C=1.0):
    """A logistic regression on the 64 numbers of a handful of labelled cells, the lecture's head, unpenalised beyond
    the default."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=2000).fit(np.asarray(X), np.asarray(y))


def map_from_head(head, q, valid):
    """The head's class for every valid cell, as a uint8 map, 0 outside."""
    X = q.reshape(64, -1).T
    full = np.zeros(X.shape[0], np.uint8)
    v = np.asarray(valid).ravel()
    full[v] = head.predict(X[v])
    return full.reshape(q.shape[1:])


def change_score(q_a, q_b, valid=None):
    """The cosine of every cell's vector in one year with its vector in another, a stability score: 1 for a place the
    model reads the same way, lower where it changed."""
    cos = np.einsum("kij,kij->ij", q_a, q_b).astype(np.float32)
    if valid is not None:
        cos[~np.asarray(valid)] = np.nan
    return cos


def stability_to_cosine(arr, scale=10000.0, nodata=-32768):
    """The shipped stability raster (the cosine times 10,000 as a whole number, the nodata value outside the town)
    back to the cosine it stores."""
    a = np.asarray(arr)
    cos = a.astype(np.float32) / scale
    cos[a == nodata] = np.nan
    return cos


def held_out_cells(valid, labels, held_out, n=60000, seed=SEED):
    """The labelled cells of the held-out blocks the lecture scored its few-shot heads on: a seeded sample of n among
    the valid, labelled cells inside the held-out blocks, as flat indices with their labels."""
    lab = np.asarray(valid).ravel() & (np.asarray(labels).ravel() > 0) & np.asarray(held_out).ravel()
    ho = np.flatnonzero(lab)
    ho = np.random.default_rng(seed).choice(ho, min(n, len(ho)), replace=False)
    return ho, np.asarray(labels).ravel()[ho]


def extent_of(transform, shape):
    """(left, right, bottom, top) of a raster for drawing it in its own system."""
    rows, cols = shape
    return (transform.c, transform.c + cols * transform.a, transform.f + rows * transform.e, transform.f)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

ROLE_COLOUR = {"train": None, "val": "#F6BD60", "test": "#DB5461"}
STAR, CROSS, BOX = "#ffdc00", "#ff3b3b", "#ffdc00"


def stretch(bands, low=2, high=98):
    """(3, H, W) or (H, W, 3) uint8 to (H, W, 3) floats in 0 to 1, stretched between two percentiles, for drawing."""
    a = np.asarray(bands).astype("float32")
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.moveaxis(a, 0, -1)
    lo, hi = np.percentile(a, [low, high])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def show_image(ax, picture, title=None, cmap=None, vmin=None, vmax=None, label=None, extent=None):
    """A picture or a single band on an axes, with a colour bar when `label` is given."""
    import matplotlib.pyplot as plt
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax, extent=extent) if np.ndim(picture) == 2 else dict(extent=extent)
    im = ax.imshow(picture, interpolation="nearest", **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if label:
        plt.colorbar(im, ax=ax, shrink=0.7, label=label)
    return im


def show_window(ax, picture, blocks=None, title=None, mark=None, overlay=None):
    """The window with the blocks drawn over it: a square per block, tinted by role (test red, validation yellow)
    and numbered; `mark` is an optional (row, col, size) rectangle in the picture's pixels, drawn in white, and
    `overlay` an optional array in 0 to 1 painted red where it is not zero."""
    from matplotlib.patches import Rectangle
    ax.imshow(picture, interpolation="nearest"); ax.set_axis_off()
    if overlay is not None:
        ax.imshow(np.ma.masked_where(np.asarray(overlay) == 0, overlay), cmap="Reds", alpha=0.85, vmin=0, vmax=1)
    if blocks is not None:
        n = int(np.sqrt(len(blocks))); size = picture.shape[0] / n
        for _, b in blocks.iterrows():
            br, bc = divmod(int(b["block"]), n)
            col = ROLE_COLOUR.get(str(b["role"]))
            ax.add_patch(Rectangle((bc * size, br * size), size, size, facecolor=col or "none", alpha=0.35 if col else 1,
                                   edgecolor="white", linewidth=0.6))
            ax.text(bc * size + 0.05 * size, br * size + 0.18 * size, str(int(b["block"])), color="white", fontsize=8)
    if mark is not None:
        r, c, s = mark
        ax.add_patch(Rectangle((c, r), s, s, fill=False, edgecolor="white", linewidth=1.5))
    if title:
        ax.set_title(title)


def show_mask(ax, picture, mask, title=None, colour=(0.90, 0.25, 0.20), alpha=0.6, points=(), labels=(), box=None):
    """A boolean mask drawn over a picture in one colour, with stars for the clicks on the object, crosses for the
    background clicks, and a box when given."""
    from matplotlib.patches import Rectangle
    ax.imshow(picture, interpolation="nearest")
    over = np.zeros(np.asarray(mask).shape + (4,)); over[..., :3] = colour; over[..., 3] = np.asarray(mask, dtype=bool) * alpha
    ax.imshow(over, interpolation="nearest"); ax.set_axis_off()
    for (x, y), lab in zip(points, labels):
        if lab:
            ax.scatter([x], [y], s=120, marker="*", color=STAR, edgecolor="black", linewidth=0.5, zorder=5)
        else:
            ax.scatter([x], [y], s=90, marker="x", color=CROSS, linewidth=2.0, zorder=5)
    if box is not None:
        ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=BOX, linewidth=1.5))
    if title:
        ax.set_title(title)


def show_masks(picture, masks, scores, points=(), labels=(), box=None, window=384, titles=None, suptitle=None, size=4.0):
    """The clicked window of a picture, then every mask over it with the model's predicted overlap in its title, stars
    for the clicks and a box when given: the strip of the lecture's prompts figure."""
    import matplotlib.pyplot as plt
    pic = np.asarray(picture)
    H, W = pic.shape[:2]
    if points:
        cx, cy = points[0]
    elif box is not None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    else:
        cx, cy = W / 2, H / 2
    r0 = int(max(min(cy - window // 2, H - window), 0)); c0 = int(max(min(cx - window // 2, W - window), 0))
    win = (slice(r0, r0 + window), slice(c0, c0 + window))
    pts = [(x - c0, y - r0) for x, y in points]
    bx = None if box is None else (box[0] - c0, box[1] - r0, box[2] - c0, box[3] - r0)
    n = len(masks) + 1
    fig, axes = plt.subplots(1, n, figsize=(size * n, size))
    show_mask(axes[0], pic[win], np.zeros(pic[win].shape[:2], bool), "the prompt", points=pts, labels=labels, box=bx)
    for k, (ax, m) in enumerate(zip(axes[1:], masks)):
        t = titles[k] if titles else f"mask {k + 1}, predicted overlap {float(scores[k]):.2f}"
        show_mask(ax, pic[win], np.asarray(m)[win], t, points=pts, labels=labels, box=bx)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def show_instances(ax, picture, instances, title=None, alpha=0.55):
    """Instance masks in distinct colours over a picture."""
    ax.imshow(overlay_instances(picture, instances, alpha), interpolation="nearest"); ax.set_axis_off()
    if title:
        ax.set_title(title)


def show_agreement(ax, pred, ref, picture=None, title=None):
    """The agreement map of a prediction and a mask over a picture: roof found in green, false alarms in red, roof
    missed in blue."""
    ax.imshow(agreement_rgb(pred, ref, picture), interpolation="nearest"); ax.set_axis_off()
    if title:
        ax.set_title(title)


def show_objects(ax, picture, result, ids, title=None):
    """The matched objects of `match_objects` over a picture: predicted blobs with a match in green, without in red,
    and the outline of every reference building that was missed in blue."""
    from matplotlib.patches import Patch
    from scipy import ndimage
    pic = 0.35 + 0.65 * np.asarray(picture)
    pred_ids = result["pred_ids"]
    ok = np.isin(pred_ids, list(result["matched_pred"])) & (pred_ids > 0)
    false = (pred_ids > 0) & ~ok
    missed_ids = [k for k, v in result["ref_iou"].items() if v < 0.5]
    missed = np.isin(ids, missed_ids) & (ids > 0)
    pic[ok] = AGREEMENT["tp"]; pic[false] = AGREEMENT["fp"]
    edge = missed & ~ndimage.binary_erosion(missed, iterations=3); pic[edge] = AGREEMENT["fn"]
    ax.imshow(pic, interpolation="nearest"); ax.set_axis_off()
    handles = [Patch(color=AGREEMENT["tp"], label="predicted, matched"), Patch(color=AGREEMENT["fp"], label="predicted, no match"),
               Patch(color=AGREEMENT["fn"], label="building, missed")]
    ax.legend(handles=handles, loc="lower right", framealpha=0.85)
    if title:
        ax.set_title(title)


def show_chips(images, titles=None, ncols=8, size=1.9, frames=None, suptitle=None, wrap=28):
    """A mosaic of chips with a title under each (wrapped), framed red where `frames` is True and grey otherwise."""
    import matplotlib.pyplot as plt
    import textwrap
    n = len(images); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.25))
    axes = np.atleast_1d(axes).ravel()
    for a in axes:
        a.set_axis_off()
    for k in range(n):
        ax = axes[k]; ax.imshow(stretch(images[k]), interpolation="nearest"); ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
        if frames is not None:
            for sp in ax.spines.values():
                sp.set_edgecolor("#DB5461" if frames[k] else "#bbbbbb"); sp.set_linewidth(2.5 if frames[k] else 1)
        if titles is not None:
            ax.set_xlabel("\n".join(textwrap.wrap(str(titles[k]), wrap)), fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def show_matrix(chips, captions, values, title=None, size=1.6, vmin=0.0, vmax=1.0, rows=None):
    """Chips as rows and captions as columns, a number per cell (a probability or a cosine) printed and shaded between
    `vmin` and `vmax`, the lecture's picture of CLIP's answers; `rows` names the chips on the left when given."""
    import matplotlib.pyplot as plt
    n, m = len(chips), len(captions)
    fig = plt.figure(figsize=(size * (m + 1.2), size * n * 0.8))
    gs = fig.add_gridspec(n, m + 1, width_ratios=[1.2] + [1] * m, wspace=0.05, hspace=0.08)
    for i in range(n):
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(stretch(chips[i]), interpolation="nearest"); ax.set_axis_off()
        if rows is not None:
            ax.text(-0.08, 0.5, str(rows[i]), transform=ax.transAxes, ha="right", va="center", fontsize=7)
        for j in range(m):
            ax = fig.add_subplot(gs[i, j + 1]); v = float(values[i][j]); p = min(max((v - vmin) / (vmax - vmin + 1e-9), 0.0), 1.0)
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=(1 - 0.75 * p, 1 - 0.45 * p, 1 - 0.1 * p)))
            ax.text(0.5, 0.5, f"{v:.2f}", ha="center", va="center", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                import textwrap
                ax.set_title("\n".join(textwrap.wrap(captions[j], 16)), fontsize=7)
    if title:
        fig.suptitle(title)
    return fig


def show_error_matrix(ax, matrix, names, title=None):
    """An error matrix as the lectures draw it: rows the reference, columns the answers, counts in the cells."""
    m = np.asarray(matrix)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(m.max(), 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("the model's answer"); ax.set_ylabel("WorldCover")
    if title:
        ax.set_title(title)


def show_counts(ax, reference, model, title=None):
    """The model's count against the reference count, one point per chip, with the identity line."""
    r, m = np.asarray(reference), np.asarray(model)
    top = max(int(r.max()), int(m.max())) + 1
    ax.plot([0, top], [0, top], color="#999999", linewidth=0.8)
    ax.scatter(r, m, s=30, color="#DB5461", alpha=0.8, linewidths=0)
    ax.set_xlabel("buildings in the chip (the footprints)"); ax.set_ylabel("the model's count"); ax.set_xlim(0, top); ax.set_ylim(0, top)
    if title:
        ax.set_title(title)


def km_ticks(ax, extent, step=2000):
    """Axis ticks in kilometres from the raster's left and bottom edge."""
    left, right, bottom, top = extent
    xs = np.arange(left, right + 1, step); ys = np.arange(bottom, top + 1, step)
    ax.set_xticks(xs); ax.set_xticklabels([f"{(x - left) / 1000:.0f}" for x in xs])
    ax.set_yticks(ys); ax.set_yticklabels([f"{(y - bottom) / 1000:.0f}" for y in ys])
    ax.set_xlabel("km east"); ax.set_ylabel("km north")


def show_map(ax, arr, extent=None, title=None, cmap=None, vmin=None, vmax=None, label=None, places=None):
    """A map of the town: a colour picture or one band in a colour scale, in the raster's system with kilometre ticks
    when `extent` is given; `places` is a dict of name to (x, y) drawn as black markers with their names."""
    import matplotlib.pyplot as plt
    kw = dict(cmap=cmap, vmin=vmin, vmax=vmax) if np.ndim(arr) == 2 else {}
    im = ax.imshow(arr, extent=extent, interpolation="nearest", **kw)
    if extent is not None:
        km_ticks(ax, extent)
    else:
        ax.set_axis_off()
    if places:
        for name, (x, y) in places.items():
            ax.scatter([x], [y], s=40, marker="o", facecolor="none", edgecolor="black", linewidth=1.2, zorder=5)
            ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color="black")
    if label:
        plt.colorbar(im, ax=ax, shrink=0.7, label=label)
    if title:
        ax.set_title(title)
    return im


def show_similarity(ax, cosmap, extent, rows, cols, title, transform, vmin=0.3):
    """A cosine map of the town in the magma scale with the given cells circled, the lecture's similarity picture."""
    show_map(ax, cosmap, extent, title, cmap="magma", vmin=vmin, vmax=1.0, label="cosine")
    xy = [xy_of_cell(transform, r, c) for r, c in zip(rows, cols)]
    ax.scatter([x for x, _ in xy], [y for _, y in xy], s=40, facecolor="none", edgecolor="cyan", linewidth=1.2, zorder=5)


def mark_cells(ax, cells, transform, shape, colour="black", s=6):
    """Flat cell indices drawn as dots on a map in the raster's system."""
    rows, cols = np.unravel_index(np.asarray(cells), shape)
    xy = [xy_of_cell(transform, r, c) for r, c in zip(rows, cols)]
    ax.scatter([x for x, _ in xy], [y for _, y in xy], s=s, color=colour, zorder=5)


def show_curve(ax, curve, forest=None, title=None):
    """The lecture's few-shot curve: labelled cells per class against the agreement on the reference cells with its
    interval over the draws, and on the held-out cells; the forest of Lecture 04 as a dashed line when given."""
    ns = sorted(int(k) for k in curve)
    mean, lo, hi, held = (np.array([curve[str(k)][key] for k in ns]) for key in ("reference", "low", "high", "held_out"))
    ax.fill_between(ns, lo, hi, color="#e77500", alpha=0.2, linewidth=0)
    ax.plot(ns, mean, color="#e77500", marker="o", label="the 319 reference cells, the mean of 20 draws and their spread")
    ax.plot(ns, held, color="#3b6fb6", marker="s", label="60,000 held-out cells in other blocks")
    if forest is not None:
        ax.axhline(forest, color="#2f7d32", linestyle="--", label="the forest of Lecture 04 on the reference cells")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("labelled cells per class"); ax.set_ylabel("agreement with WorldCover"); ax.legend(frameon=False)
    if title:
        ax.set_title(title)


def class_cmap():
    """The colour scale of the four classes of Lecture 04, white for an unlabelled cell."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(["white"] + [CLASS_COLOURS[CLASS_NAMES[c]] for c in CLASSES])


def show_maps(maps, titles, cmaps=None, vmins=None, vmaxs=None, extent=None, size=3.8, suptitle=None):
    """Several maps of the town abreast, each in its own colour scale, with kilometre ticks on the first."""
    import matplotlib.pyplot as plt
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size * 1.1))
    for k, ax in enumerate(np.atleast_1d(axes)):
        cm = (cmaps or [None] * n)[k]; lo = (vmins or [None] * n)[k]; hi = (vmaxs or [None] * n)[k]
        kw = dict(cmap=cm, vmin=lo, vmax=hi) if np.ndim(maps[k]) == 2 else {}
        ax.imshow(maps[k], extent=extent, interpolation="nearest", **kw)
        ax.set_title(titles[k])
        if extent is not None and k == 0:
            km_ticks(ax, extent)
        else:
            ax.set_xticks([]); ax.set_yticks([])
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def show_vectors(vectors, names, colours=None, title=None):
    """The 64 numbers of a few cells as bar strips, one row each, the lecture's picture of what a vector looks like."""
    import matplotlib.pyplot as plt
    n = len(vectors)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=True)
    for ax, v, name, col in zip(np.atleast_1d(axes), vectors, names, colours or ["#e77500"] * n):
        ax.bar(np.arange(64), np.asarray(v), color=col, width=0.8, linewidth=0)
        ax.axhline(0, color="black", linewidth=0.4); ax.set_xlim(-1, 64); ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    np.atleast_1d(axes)[-1].set_xlabel("the 64 numbers, none of them named")
    if title:
        fig.suptitle(title)
    return fig


def show_toy(names, vectors, cos, pictures=None, title=None):
    """The toy of the lecture: the places as pictures with their three numbers, and the table of their cosines."""
    import matplotlib.pyplot as plt
    n = len(names)
    fig = plt.figure(figsize=(1.3 * n + 5, 1.3 * n * 0.55 + 1.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[n * 0.55, n * 0.75], wspace=0.08)
    left = gs[0].subgridspec(2, n, height_ratios=[1, 0.6], hspace=0.02)
    for k in range(n):
        ax = fig.add_subplot(left[0, k]); ax.set_axis_off()
        if pictures is not None and k < len(pictures):
            ax.imshow(pictures[k], interpolation="nearest")
        ax.set_title(names[k], fontsize=8)
        ax = fig.add_subplot(left[1, k]); ax.set_axis_off()
        ax.text(0.5, 0.5, "\n".join(f"{v:.2f}" for v in vectors[k]), ha="center", va="center", fontsize=8)
    ax = fig.add_subplot(gs[1])
    c = np.asarray(cos)
    ax.imshow(c, cmap="Oranges", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if c[i, j] > 0.7 else "black")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("the cosine of every pair", fontsize=9)
    if title:
        fig.suptitle(title)
    return fig
