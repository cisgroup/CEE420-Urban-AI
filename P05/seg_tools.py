"""The toolbox of Precept 05, Segmenting Princeton.

Every algorithm the four notebooks use lives here once, documented, so that a code cell in a notebook is a few
lines that call a named tool. The lecture's own pipeline imports the same file, so the U-Net the deck trained is
the U-Net the kit loads, weight for weight.

The model section needs PyTorch and torchvision. They are imported inside the functions that use them, so the
module loads in an environment without torch and every tool that works on arrays, rasters and polygons keeps
working; `torch_available()` says which world you are in and `NO_TORCH` is the one line a notebook prints then.

Sections: loading, chips and masks, metrics, the chip head, the model, plotting, street level.
"""

import json
import time
from pathlib import Path

import numpy as np

__all__ = [
    # loading
    "load_scene", "read_raster", "read_rgb", "chip_table", "read_npz", "read_curves", "read_street",
    # chips and masks
    "block_chips", "cut_chips", "first_batch", "split_by_blocks", "random_split", "label_from_share",
    # metrics
    "confusion", "pixel_metrics", "iou", "chip_scores", "rasterise", "match_objects", "clean_mask", "polygonise",
    "agreement_rgb", "equal_accuracy_pair",
    # the chip head
    "fit_head", "head_accuracy", "fisher_plane",
    # the model
    "torch_available", "NO_TORCH", "DECODER_CHANNELS", "IMAGENET_MEAN", "IMAGENET_STD",
    "build_unet", "build_classifier", "count_parameters", "fresh_decoder", "set_weights", "save_weights", "load_weights",
    "weights_provenance", "device", "normalise", "predict", "predict_tiled", "encoder_features", "dihedral",
    "load_model", "predict_block", "train_decoder", "encoder_maps",
    # plotting
    "stretch", "show_image", "show_window", "show_mask", "show_chips", "show_batch", "show_feature_space", "show_feature_maps",
    "show_evolution", "show_two_chips", "show_curves", "show_agreement", "show_objects", "show_grid",
    # street level
    "class_shares", "green_view_index", "show_street",
]

# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

NO_TORCH = ("PyTorch is not installed in this environment, so this cell is skipped; the maps and numbers it would "
            "compute are read from the shipped files instead (see docs/troubleshooting.org for the environment update).")
DECODER_CHANNELS = (128, 64, 32, 16, 16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def torch_available():
    """True when torch and torchvision import, False otherwise. Printing `NO_TORCH` is the notebooks' answer to False."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F

    class _Up(_nn.Module):
        """One decoder level: upsample by two (nearest), concatenate the encoder's map of that size, two 3 by 3
        convolutions with batch normalisation and the rectifier."""

        def __init__(self, c_in, c_skip, c_out):
            super().__init__()
            self.block = _nn.Sequential(
                _nn.Conv2d(c_in + c_skip, c_out, 3, padding=1, bias=False), _nn.BatchNorm2d(c_out), _nn.ReLU(inplace=True),
                _nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), _nn.BatchNorm2d(c_out), _nn.ReLU(inplace=True))

        def forward(self, x, skip=None):
            x = _F.interpolate(x, scale_factor=2, mode="nearest")
            if skip is not None:
                x = _torch.cat([x, skip], dim=1)
            return self.block(x)

    class Unet(_nn.Module):
        """A U-Net whose encoder is torchvision's ResNet-18 (the network of Lecture 04) and whose decoder is five
        `_Up` levels back to the input size, ending in a 1 by 1 convolution to one logit per pixel (roof against not
        roof). With `skips=False` the decoder sees only the bottleneck, the ablation of the lecture."""

        def __init__(self, encoder, decoder_channels=DECODER_CHANNELS, skips=True):
            super().__init__()
            self.skips = skips
            self.stem = _nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)        # 128 by 128 by 64
            self.pool = encoder.maxpool
            self.layer1, self.layer2, self.layer3, self.layer4 = encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
            enc = (512, 256, 128, 64, 64)                                                 # layer4, layer3, layer2, layer1, stem
            d = decoder_channels
            sk = (lambda c: c) if skips else (lambda c: 0)
            self.up1 = _Up(enc[0], sk(enc[1]), d[0])       # 8 to 16
            self.up2 = _Up(d[0], sk(enc[2]), d[1])         # 16 to 32
            self.up3 = _Up(d[1], sk(enc[3]), d[2])         # 32 to 64
            self.up4 = _Up(d[2], sk(enc[4]), d[3])         # 64 to 128
            self.up5 = _Up(d[3], 0, d[4])                  # 128 to 256
            self.head = _nn.Conv2d(d[4], 1, 1)

        def encoder_modules(self):
            return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

        def decoder_modules(self):
            return [self.up1, self.up2, self.up3, self.up4, self.up5, self.head]

        def encode(self, x):
            s0 = self.stem(x)
            s1 = self.layer1(self.pool(s0))
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            s4 = self.layer4(s3)
            return s0, s1, s2, s3, s4

        def forward(self, x):
            s0, s1, s2, s3, s4 = self.encode(x)
            k = (lambda s: s) if self.skips else (lambda s: None)
            y = self.up1(s4, k(s3))
            y = self.up2(y, k(s2))
            y = self.up3(y, k(s1))
            y = self.up4(y, k(s0))
            y = self.up5(y)
            return self.head(y)

except ImportError:                      # the course environment without torch: the array tools still work
    Unet = None
    _Up = None


def build_unet(decoder_channels=DECODER_CHANNELS, pretrained=False, skips=True):
    """The U-Net of the lecture: a ResNet-18 encoder (ImageNet weights when `pretrained`, which needs the network or
    the torchvision cache; random otherwise, the form the kit loads shipped weights into) and the decoder above."""
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = torchvision.models.resnet18(weights=weights)
    return Unet(encoder, decoder_channels, skips)


def build_classifier(pretrained=False):
    """The chip classifier of Chapter 1: ResNet-18 with a new two-way head (512 times 2 plus 2, 1,026 weights)."""
    import torch.nn as nn
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(512, 2)
    return model


def count_parameters(model):
    """Weights in the encoder, the decoder and in all, as a dict, for a `Unet`; for a classifier the body and the head."""
    if hasattr(model, "encoder_modules"):
        enc = sum(p.numel() for m in model.encoder_modules() for p in m.parameters())
        dec = sum(p.numel() for m in model.decoder_modules() for p in m.parameters())
        return dict(encoder=enc, decoder=dec, total=sum(p.numel() for p in model.parameters()))
    head = sum(p.numel() for p in model.fc.parameters())
    total = sum(p.numel() for p in model.parameters())
    return dict(encoder=total - head, decoder=head, total=total)


def fresh_decoder(model, seed=20261012):
    """Re-initialise every decoder weight from the seed and leave the encoder as it is."""
    import torch
    torch.manual_seed(seed)
    for m in model.decoder_modules():
        for layer in m.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    return model


def save_weights(model, path, **provenance):
    """Every weight as float16 in one npz, keyed as the state dict, plus the provenance strings (source, licence,
    retrieved, architecture, run, seed and whatever else is given) as zero-dimensional string arrays."""
    arrays = {k: v.detach().cpu().numpy().astype("float16") if v.is_floating_point() else v.detach().cpu().numpy()
              for k, v in model.state_dict().items()}
    meta = {k: np.array(str(v)) for k, v in provenance.items()}
    np.savez(path, **arrays, **meta)
    return Path(path)


def load_weights(path):
    """The weight arrays of a saved npz as float32 (and the integer buffers as they are), without the provenance."""
    z = np.load(path)
    out = {}
    for k in z.files:
        a = z[k]
        if a.dtype == np.float16:
            out[k] = a.astype("float32")
        elif a.dtype.kind in "iu":
            out[k] = a
    return out


def weights_provenance(path):
    """The provenance strings a saved npz carries (source, licence, retrieved, architecture, run, seed and the rest)."""
    z = np.load(path)
    return {k: str(z[k]) for k in z.files if z[k].dtype.kind in "US"}


def set_weights(model, arrays):
    """Load a dict of arrays into the model, strictly, and put it in evaluation mode."""
    import torch
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}, strict=True)
    return model.eval()


def device(prefer="mps"):
    """The Apple GPU when torch sees it and `prefer` asks for it, else the CPU."""
    import torch
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalise(chips):
    """(N, 3, H, W) uint8 red, green, blue to the float tensor the ImageNet weights expect."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(chips)).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def dihedral(image, mask, k, flip):
    """One of the eight symmetries of a square chip: k quarter turns and an optional mirror, applied alike to the
    image (bands first) and the mask."""
    img = np.rot90(image, k, axes=(-2, -1))
    msk = np.rot90(mask, k, axes=(-2, -1))
    if flip:
        img = img[..., ::-1]
        msk = msk[..., ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def predict(model, chips, batch=16, dev=None):
    """Roof probabilities (N, H, W) float32 for (N, 3, H, W) uint8 chips."""
    import torch
    dev = dev or device()
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(chips), batch):
            x = normalise(chips[i:i + batch]).to(dev)
            out.append(torch.sigmoid(model(x)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,) + chips.shape[2:], dtype="float32")


def predict_tiled(model, image, size=256, stride=128, batch=16, dev=None, progress=None):
    """The probability map of a whole (3, H, W) uint8 image, chip by chip. With `stride` equal to `size` the chips tile
    the image and every pixel is predicted once, seams and all. With a smaller stride only the centre `stride` pixels
    of every chip are kept (the rim of the image comes from the outermost chips), which is the overlap remedy of the
    lecture. H and W must be multiples of `size`."""
    _, H, W = image.shape
    assert H % size == 0 and W % size == 0, "the image must be a whole number of chips"
    prob = np.zeros((H, W), dtype="float32")
    keep = (size - stride) // 2
    rows = list(range(0, H - size + 1, stride))
    cols = list(range(0, W - size + 1, stride))
    todo = [(r, c) for r in rows for c in cols]
    for i in range(0, len(todo), batch):
        rc = todo[i:i + batch]
        chips = np.stack([image[:, r:r + size, c:c + size] for r, c in rc])
        p = predict(model, chips, batch=len(rc), dev=dev)
        for (r, c), pk in zip(rc, p):
            if stride == size:
                prob[r:r + size, c:c + size] = pk
                continue
            r0 = 0 if r == 0 else keep
            r1 = size if r == rows[-1] else size - keep
            c0 = 0 if c == 0 else keep
            c1 = size if c == cols[-1] else size - keep
            prob[r + r0:r + r1, c + c0:c + c1] = pk[r0:r1, c0:c1]
        if progress:
            progress(i + len(rc), len(todo))
    return prob


def encoder_features(model, chips, batch=32, dev=None):
    """The 512 numbers of the average-pooled last encoder map per chip, the frozen features of Lecture 04 F17, for a
    `Unet` (its encoder) or a classifier (everything but its head)."""
    import torch
    dev = dev or device()
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(chips), batch):
            x = normalise(chips[i:i + batch]).to(dev)
            if hasattr(model, "encode"):
                s4 = model.encode(x)[-1]
            else:
                body = torch.nn.Sequential(*list(model.children())[:-2])
                s4 = body(x)
            out.append(s4.mean(dim=(2, 3)).float().cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_scene(path="data/scene.json"):
    """The kit's scene.json as a dict: the window, the mask rule, the blocks and the split, the chips, the lecture's
    numbers and the choices the pipeline made."""
    return json.loads(Path(path).read_text())


def read_raster(path):
    """A GeoTIFF as (array, transform, crs, resolution in metres): bands first for a colour image, a 2-D array for one
    band. The tags travel with the file; open it with rasterio to read them."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read()
        return (arr if src.count > 1 else arr[0]), src.transform, src.crs, float(src.res[0])


def stretch(bands, low=2, high=98):
    """(3, H, W) or (H, W, 3) uint8 to (H, W, 3) floats in 0 to 1, stretched between two percentiles, for drawing."""
    a = np.asarray(bands).astype("float32")
    if a.ndim == 3 and a.shape[0] == 3:
        a = np.moveaxis(a, 0, -1)
    lo, hi = np.percentile(a, [low, high])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def read_rgb(path):
    """A three-band GeoTIFF as a stretched (H, W, 3) picture plus its transform."""
    arr, transform, crs, res = read_raster(path)
    return stretch(arr), transform


def chip_table(path="data/chips.csv"):
    """The 1,600 tiled chips as a table: row and column of the chip's north-west corner in the window's pixels, its
    block, its split, its roof share and its label."""
    import pandas as pd
    return pd.read_csv(path)


def read_npz(path):
    """A shipped npz as a dict of arrays; the zero-dimensional string arrays (the provenance) become strings."""
    z = np.load(path, allow_pickle=False)
    out = {}
    for k in z.files:
        a = z[k]
        out[k] = str(a) if a.dtype.kind in "US" and a.ndim == 0 else a
    return out


def read_curves(path="data/curves.json"):
    """The lecture's training curves: a dict with the U-Net runs and the chip classifier runs, each with its curve
    as a list of rows, its best epoch and its test numbers."""
    return json.loads(Path(path).read_text())


def read_street(image_path, labels_path):
    """A Mapillary image as (H, W, 3) uint8 and its label map with the class names and the palette."""
    from PIL import Image
    image = np.asarray(Image.open(image_path).convert("RGB"))
    z = read_npz(labels_path)
    names = [str(n) for n in z["names"]]
    return image, z["labels"], names, z["palette"], z


# --------------------------------------------------------------------------- #
# chips and masks
# --------------------------------------------------------------------------- #

def block_chips(table, block):
    """The rows of one block, in the order the table has them."""
    return table[table["block"] == block].reset_index(drop=True)


def cut_chips(image, mask, rows, cols, size=256, origin=(0, 0)):
    """The chips of a (3, H, W) image and its (H, W) mask at the given window rows and columns, as (N, 3, size, size)
    and (N, size, size); `origin` is the window row and column of the image's first pixel, for a block cut out of
    the window."""
    r0, c0 = origin
    imgs = np.stack([image[:, r - r0:r - r0 + size, c - c0:c - c0 + size] for r, c in zip(rows, cols)])
    msks = np.stack([mask[r - r0:r - r0 + size, c - c0:c - c0 + size] for r, c in zip(rows, cols)])
    return imgs, msks


def first_batch(table, seed=20261012, batch=32):
    """The chips of the first batch of the first epoch of the lecture's chip classifier: numpy's permutation of the
    training chips with the seed, cut after `batch`."""
    train = np.flatnonzero((table["split"] == "train").to_numpy())
    return np.random.default_rng(seed).permutation(train)[:batch]


def split_by_blocks(table, block_m, seed=20261012, test_share=0.2, val_share=0.16, resolution=0.3):
    """A split of the chips by square blocks of `block_m` metres. At the lecture's own block size the table's split
    is returned, so the number you start from is the one printed above. At any other size the blocks are dealt anew
    from the seed, a fifth as the test and four twenty-fifths held out as the validation, the lecture's shares.
    Returns two boolean arrays, train and test, in the table's order."""
    size_px = int(round(block_m / resolution))
    n_cols = int(np.ceil((table["col"].max() + 256) / size_px))
    ids = (table["row"].to_numpy() // size_px) * n_cols + table["col"].to_numpy() // size_px
    if np.array_equal(ids, table["block"].to_numpy()):
        split = table["split"].to_numpy()
        return split == "train", split == "test"
    dealt = np.random.default_rng(seed).permutation(np.unique(ids))
    n_test, n_val = max(1, int(round(test_share * len(dealt)))), int(round(val_share * len(dealt)))
    test, val = np.isin(ids, dealt[:n_test]), np.isin(ids, dealt[n_test:n_test + n_val])
    return ~test & ~val, test


def random_split(n, n_train, n_test, seed=20261012):
    """A random split of `n` chips into `n_train` for training and `n_test` for the test, as two boolean arrays."""
    perm = np.random.default_rng(seed).permutation(n)
    train = np.zeros(n, dtype=bool); train[perm[:n_train]] = True
    test = np.zeros(n, dtype=bool); test[perm[n - n_test:]] = True
    return train, test


def label_from_share(share, rule=0.025):
    """The chip label of the lecture: a building when the roof share is at least the rule."""
    return np.asarray(share) >= rule


# --------------------------------------------------------------------------- #
# metrics
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


def chip_scores(prob, mask, size=256, threshold=0.5):
    """The score of every tiled chip of a probability map against its mask: a table with the chip's row and column
    in the map, its roof share, its share of pixels right and its IoU (None for a chip without a roof pixel and
    without a predicted one), plus the pooled IoU over all the pixels."""
    H, W = mask.shape
    rows = []
    pooled = confusion(prob >= threshold, mask > 0)
    for r in range(0, H - size + 1, size):
        for c in range(0, W - size + 1, size):
            p = prob[r:r + size, c:c + size] >= threshold; m = mask[r:r + size, c:c + size] > 0
            cm = confusion(p, m); pm = pixel_metrics(cm)
            defined = cm["tp"] + cm["fp"] + cm["fn"] > 0
            rows.append(dict(row=r, col=c, share=float(m.mean()), accuracy=pm["accuracy"],
                             iou=pm["iou"] if defined else None, **cm))
    return rows, pixel_metrics(pooled)


def rasterise(footprints, transform, shape, column="bid"):
    """The footprints burnt onto a grid: the value of `column` on every pixel whose centre lies inside one, 0 elsewhere."""
    import rasterio.features
    shapes = ((g, int(v)) for g, v in zip(footprints.geometry, footprints[column]))
    return rasterio.features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def match_objects(pred, ids, threshold=0.5, min_pred_px=222, min_ref_px=111):
    """Objects matched one to one by IoU, as the lecture counts them. Reference objects are the footprints of `ids`
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


def clean_mask(pred, min_px):
    """Drop every blob smaller than `min_px` pixels and fill the holes of the rest; returns the mask and the blob
    counts before and after."""
    from scipy import ndimage
    lab, n = ndimage.label(np.asarray(pred, dtype=bool), structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel()); keep = sizes >= min_px; keep[0] = False
    out = ndimage.binary_fill_holes(keep[lab])
    return out, int(n), int(ndimage.label(out, structure=np.ones((3, 3)))[1])


def polygonise(mask, transform, tolerance=0.6):
    """The blobs of a boolean mask as shapely polygons in the raster's system, simplified to `tolerance` metres."""
    import rasterio.features
    from shapely.geometry import shape
    m = np.asarray(mask, dtype=bool)
    found = rasterio.features.shapes(m.astype("uint8"), mask=m, transform=transform)
    return [shape(g).simplify(tolerance) for g, v in found if v == 1]


AGREEMENT = dict(tp=(0.20, 0.60, 0.25), fp=(0.85, 0.25, 0.20), fn=(0.15, 0.35, 0.75))


def agreement_rgb(pred, ref, image=None):
    """A picture of agreement: the roof found in green, false alarms in red, roof missed in blue, over the faded
    image (or white)."""
    pred = np.asarray(pred, dtype=bool); ref = np.asarray(ref, dtype=bool)
    out = np.ones(pred.shape + (3,)) if image is None else 0.35 + 0.65 * np.asarray(image)
    out[pred & ref] = AGREEMENT["tp"]; out[pred & ~ref] = AGREEMENT["fp"]; out[~pred & ref] = AGREEMENT["fn"]
    return out


def equal_accuracy_pair(scores, tolerance=0.005, low=0.90, high=0.99):
    """From the chip scores of a block, the pair of chips with a roof whose shares of pixels right differ by less than
    `tolerance` and whose IoUs differ most, the first the lower IoU; the lower chip's accuracy between `low` and `high`."""
    with_roof = [s for s in scores if s["iou"] is not None and s["tp"] + s["fn"] > 0]
    best = None
    for a in with_roof:
        if not (low <= a["accuracy"] <= high):
            continue
        for b in with_roof:
            if abs(a["accuracy"] - b["accuracy"]) < tolerance and b["iou"] > a["iou"] and (best is None or b["iou"] - a["iou"] > best[0]):
                best = (b["iou"] - a["iou"], a, b)
    assert best is not None, "no two chips with the same share of pixels right and a different IoU"
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# the chip head: a logistic regression on the 512 frozen numbers, scikit-learn's
# --------------------------------------------------------------------------- #

def fit_head(features, labels, train, C=0.01, seed=20261012):
    """A logistic regression on standardised features, fitted on the rows where `train` is True: a head of 513
    weights on the 512 numbers, with a penalty (`C`, smaller is stronger) that keeps it from memorising a thousand
    chips in 512 dimensions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C, random_state=seed))
    return head.fit(np.asarray(features)[train], np.asarray(labels)[train])


def head_accuracy(head, features, labels, idx):
    """The share of the chips at `idx` the head gets right."""
    return float((head.predict(np.asarray(features)[idx]) == np.asarray(labels)[idx]).mean())


def fisher_plane(features, labels, train):
    """Two coordinates per chip for drawing a feature space: the direction that best separates the training chips
    (Fisher's discriminant on standardised features) and the first principal component orthogonal to it."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = np.asarray(features, dtype="float64"); y = np.asarray(labels)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xn = (X - mu) / sd
    w = LinearDiscriminantAnalysis(n_components=1).fit(Xn[train], y[train]).scalings_[:, 0]
    w = w / np.linalg.norm(w)
    a = Xn @ w
    R = Xn - np.outer(a, w)
    v = np.linalg.svd(R[train] - R[train].mean(0), full_matrices=False)[2][0]
    return a, R @ v


# --------------------------------------------------------------------------- #
# the model, continued: the kit's weights, a block, a short training, the maps
# --------------------------------------------------------------------------- #

def load_model(path="data/segmenter_frozen.npz"):
    """The U-Net with the shipped weights, in evaluation mode, and the provenance the file carries."""
    return set_weights(build_unet(pretrained=False), load_weights(path)), weights_provenance(path)


def predict_block(model, image, stride=128, dev=None, quiet=False):
    """The probability map of a block, chip by chip with the chip centres kept, printing how long it took."""
    t0 = time.time()
    prob = predict_tiled(model, image, 256, stride, dev=dev)
    if not quiet:
        print(f"{image.shape[1] // 256 * (image.shape[2] // 256)} chips predicted in {time.time() - t0:.0f} s on {dev or device()}")
    return prob


def train_decoder(model, chips, masks, epochs=6, lr=1e-3, batch=8, seed=20261012, eval_chips=None, eval_masks=None,
                  dev=None, augment=True, quiet=False):
    """Train a fresh decoder on a few chips with the encoder frozen, the recipe of the lecture's frozen run in
    miniature: Adam, the per-pixel cross-entropy, the eight symmetries, `epochs` passes in batches of `batch`.
    After every epoch the IoU on the evaluation chips is measured. Returns the model and the curve as a list of
    (epoch, training loss, evaluation IoU, seconds)."""
    import torch
    dev = dev or device()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = fresh_decoder(model, seed).to(dev)
    for m in model.encoder_modules():
        for p in m.parameters():
            p.requires_grad_(False)
    params = [p for m in model.decoder_modules() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    curve = []
    for epoch in range(1, epochs + 1):
        t0 = time.time(); losses = []
        model.train()
        for m in model.encoder_modules():
            m.eval()                                   # the encoder's batch statistics stay as learned
        order = rng.permutation(len(chips))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            imgs, msks = [], []
            for j in idx:
                k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
                im, mk = dihedral(chips[j], masks[j], k, flip) if augment else (chips[j], masks[j])
                imgs.append(im); msks.append(mk)
            x = normalise(np.stack(imgs)).to(dev); y = torch.from_numpy(np.stack(msks)).float().unsqueeze(1).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        score = float("nan")
        if eval_chips is not None:
            p = predict(model, eval_chips, dev=dev)
            score = iou(p >= 0.5, np.asarray(eval_masks) > 0)
        curve.append((epoch, float(np.mean(losses)), score, time.time() - t0))
        if not quiet:
            print(f"epoch {epoch:2d}   training loss {curve[-1][1]:.3f}   IoU on the evaluation chips {score:.3f}   {curve[-1][3]:.1f} s")
    return model.eval(), curve


def encoder_maps(model, chip, dev=None):
    """One chip through the encoder: the five maps (128, 64, 32, 16 and 8 pixels a side) as their mean over channels,
    and the 512 pooled numbers."""
    import torch
    dev = dev or device()
    model = model.to(dev).eval()
    with torch.no_grad():
        maps = model.encode(normalise(np.asarray(chip)[None]).to(dev))
        pooled = maps[-1].mean(dim=(2, 3))[0].float().cpu().numpy()
    return [m[0].float().mean(0).cpu().numpy() for m in maps], pooled


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

ROLE_COLOUR = {"train": None, "val": "#F6BD60", "test": "#DB5461"}
PROB_COLOURS = ["#f5f0e6", "#e77500", "#7a2e00"]


def _prob_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("prob", PROB_COLOURS)


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


def show_window(ax, picture, blocks=None, title=None):
    """The window with the blocks drawn over it: a square per block, tinted by role (test red, validation yellow)
    and numbered. `blocks` is the table of the blocks layer in the picture's pixels, or None."""
    from matplotlib.patches import Rectangle
    ax.imshow(picture, interpolation="nearest"); ax.set_axis_off()
    if blocks is not None:
        n = int(np.sqrt(len(blocks))); size = picture.shape[0] / n
        for _, b in blocks.iterrows():
            br, bc = divmod(int(b["block"]), n)
            col = ROLE_COLOUR.get(str(b["role"]))
            ax.add_patch(Rectangle((bc * size, br * size), size, size, facecolor=col or "none", alpha=0.35 if col else 1,
                                   edgecolor="white", linewidth=0.6))
            ax.text(bc * size + 0.05 * size, br * size + 0.18 * size, str(int(b["block"])), color="white", fontsize=8, path_effects=None)
    if title:
        ax.set_title(title)


def show_mask(ax, picture, mask, title=None, colour=(0.90, 0.25, 0.20), alpha=0.7):
    """A boolean mask drawn over a picture in one colour."""
    ax.imshow(picture, interpolation="nearest")
    over = np.zeros(np.asarray(mask).shape + (4,)); over[..., :3] = colour; over[..., 3] = np.asarray(mask, dtype=bool) * alpha
    ax.imshow(over, interpolation="nearest"); ax.set_axis_off()
    if title:
        ax.set_title(title)


def show_chips(images, masks=None, labels=None, ncols=8, size=1.6, title=None, probs=None):
    """A mosaic of chips: the pictures in one row of panels, their masks below when given; a chip framed red when its
    label is a building and grey otherwise; a number printed in the corner when `probs` is given."""
    import matplotlib.pyplot as plt
    n = len(images); nrows = (1 if masks is None else 2) * int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.set_axis_off()
    for k in range(n):
        block, pos = divmod(k, ncols)
        r = block * (2 if masks is not None else 1)
        ax = axes[r][pos]; ax.imshow(stretch(images[k]), interpolation="nearest"); ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
        if labels is not None:
            for sp in ax.spines.values():
                sp.set_edgecolor("#DB5461" if labels[k] else "#bbbbbb"); sp.set_linewidth(2.5 if labels[k] else 1)
        if probs is not None:
            ax.text(0.04, 0.96, f"{probs[k]:.2f}", transform=ax.transAxes, va="top", color="white", fontsize=9,
                    bbox=dict(facecolor="black", alpha=0.5, pad=1))
        if masks is not None:
            axes[r + 1][pos].imshow(masks[k], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    if title:
        fig.suptitle(title)
    return fig


def show_batch(images, labels, title="one batch, framed red where the label says building"):
    """The chips of a batch as a 4 by 8 mosaic framed by label."""
    return show_chips(images, labels=labels, ncols=8, title=title)


def show_feature_space(spaces, labels, train, test, titles):
    """One panel per feature space: the test chips as points placed by the Fisher direction fitted on the training
    chips and the orthogonal principal component, coloured by label."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(spaces), 1, figsize=(9, 2.6 * len(spaces)))
    for ax, X, title in zip(np.atleast_1d(axes), spaces, titles):
        a, b = fisher_plane(X, labels, train)
        for lab, col, name in ((0, "#8a8a8a", "no building"), (1, "#DB5461", "building")):
            m = test[np.asarray(labels)[test] == lab]
            ax.scatter(a[m], b[m], s=10, color=col, alpha=0.7, linewidths=0, label=name)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("the direction that separates the training chips")
    fig.tight_layout()
    return fig


def show_feature_maps(chip, maps, pooled, title="one chip through the encoder, each map as its mean over channels"):
    """The chip, its five encoder maps from 128 to 8 pixels, and the 512 pooled numbers as a strip."""
    import matplotlib.pyplot as plt
    widths = [1] * (len(maps) + 1) + [0.25]
    fig, axes = plt.subplots(1, len(maps) + 2, figsize=(2.2 * (len(maps) + 2), 2.6), gridspec_kw=dict(width_ratios=widths))
    axes[0].imshow(stretch(chip), interpolation="nearest"); axes[0].set_title("the chip, 256")
    for ax, m in zip(axes[1:-1], maps):
        ax.imshow(m, cmap="magma", interpolation="nearest"); ax.set_title(f"{m.shape[0]} by {m.shape[1]}")
    axes[-1].imshow(pooled.reshape(-1, 1), cmap="magma", aspect="auto", interpolation="nearest"); axes[-1].set_title("512")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(title)
    return fig


def show_evolution(image, mask, probs, ious, epochs, best, title="one validation chip over the epochs"):
    """The chip and its label, then its probability map after the given epochs with the IoU under each, the kept epoch framed."""
    import matplotlib.pyplot as plt
    n = len(epochs) + 2
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    axes[0].imshow(stretch(image), interpolation="nearest"); axes[0].set_title("the chip")
    axes[1].imshow(mask, cmap="Greys", vmin=0, vmax=1, interpolation="nearest"); axes[1].set_title("the label")
    for ax, e in zip(axes[2:], epochs):
        ax.imshow(probs[e - 1] / 255.0, cmap=_prob_cmap(), vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"epoch {e}{', kept' if e == best else ''}\nIoU {ious[e - 1]:.2f}")
        if e == best:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#e77500"); sp.set_linewidth(2.5)
            continue
        ax.set_axis_off()
    for ax in axes[:2]:
        ax.set_axis_off()
    fig.suptitle(title)
    return fig


def show_two_chips(pictures, masks, preds, scores, title="two chips with the same share of pixels right"):
    """Two chips side by side, each as its picture above its agreement map, with the share of pixels right and the IoU."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    for k in range(2):
        axes[0][k].imshow(pictures[k], interpolation="nearest")
        axes[0][k].set_title(f"pixels right {scores[k]['accuracy']:.1%}, IoU {scores[k]['iou']:.2f}")
        axes[1][k].imshow(agreement_rgb(preds[k], masks[k], pictures[k]), interpolation="nearest")
        for ax in (axes[0][k], axes[1][k]):
            ax.set_axis_off()
    handles = [Patch(color=AGREEMENT["tp"], label="roof found"), Patch(color=AGREEMENT["fp"], label="false alarm"),
               Patch(color=AGREEMENT["fn"], label="roof missed")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title)
    return fig


def show_curves(ax, curves, column, names=None, ylabel=None, title=None, mark_best=True):
    """The lecture's curves of one column (by name in the file's column list) for the given runs, with the kept
    epoch as a dot."""
    for run, c in curves.items():
        rows = np.asarray(c["curve"]); k = column
        ax.plot(rows[:, 0], rows[:, k], label=(names or {}).get(run, run))
        if mark_best:
            b = int(c["best_epoch"]); ax.plot([b], [rows[b - 1, k]], "o", color=ax.lines[-1].get_color())
    ax.set_xlabel("epoch"); ax.legend(frameon=False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def show_grid(ax, picture, rows, cols, size=256, origin=(0, 0), title=None, colour="white"):
    """A picture with the outline of every chip at the given window rows and columns; `origin` is the window row
    and column of the picture's first pixel."""
    from matplotlib.patches import Rectangle
    ax.imshow(picture, interpolation="nearest"); ax.set_axis_off()
    for r, c in zip(rows, cols):
        ax.add_patch(Rectangle((c - origin[1], r - origin[0]), size, size, fill=False, edgecolor=colour, linewidth=0.6))
    if title:
        ax.set_title(title)


def show_agreement(ax, pred, ref, picture=None, title=None):
    """The agreement map of a prediction and a mask over a picture: roof found in green, false alarms in red, roof missed in blue."""
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


# --------------------------------------------------------------------------- #
# street level
# --------------------------------------------------------------------------- #

def class_shares(labels, names):
    """The share of the label map's pixels in every class, as a dict ordered by share."""
    counts = np.bincount(np.asarray(labels).ravel(), minlength=len(names))
    shares = {names[i]: float(counts[i] / counts.sum()) for i in range(len(names))}
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))


def green_view_index(labels, names, green=("vegetation",)):
    """The share of the pixels in the green classes, vegetation alone unless told otherwise."""
    shares = class_shares(labels, names)
    return float(sum(shares.get(g, 0.0) for g in green))


def show_street(image, labels, palette, names, title=None, top=8):
    """The image beside its label map in the Cityscapes palette, with a legend of the largest classes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(image); axes[1].imshow(np.asarray(palette)[np.asarray(labels)], interpolation="nearest")
    for ax in axes:
        ax.set_axis_off()
    shares = class_shares(labels, names)
    handles = [Patch(color=np.asarray(palette[names.index(n)]) / 255.0, label=f"{n} {s:.0%}")
               for n, s in list(shares.items())[:top]]
    fig.legend(handles=handles, loc="lower center", ncol=min(top, 4), frameon=False)
    if title:
        fig.suptitle(title)
    return fig
