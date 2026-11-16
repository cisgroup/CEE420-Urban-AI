"""graph_tools: the toolbox of Lecture 08 and the P08 kit, graph learning in plain NumPy and PyTorch.

Every algorithm of the lecture is written here once: the adjacency matrix as an operator, message passing with
its normalisations, the graph Laplacian, the graph convolutional layer of Kipf and Welling (2017), the model
without a graph it is compared against, the training loop on the labelled nodes, and the sensor graph and the
baselines of traffic forecasting. The notation is the lecture's: A the adjacency matrix, D the degree matrix,
X the node features, H the hidden states, W the weights of a layer.

The matrices are dense. A town of a few hundred places and a network of 207 sensors need nothing else, and a
dense matrix can be printed and checked by hand.
"""

import json

import numpy as np

try:
    import torch
    from torch import nn
    torch_ok = True
    torch.set_num_threads(1)             # one thread: a training run then gives the same numbers on every laptop
except ImportError:                      # the NumPy half of the toolbox works without PyTorch
    torch, nn, torch_ok = None, None, False

NO_TORCH = "PyTorch is not installed in this environment, so this cell is skipped."
NO_PYG = "PyTorch Geometric is not installed in this environment, so this cell is skipped."
SEED = 20261116
CRS = "EPSG:26918"                       # UTM zone 18 north, metres: every map of Princeton in the kit
SPEED_NAMES = ["25 mph or less", "30 mph", "35 mph", "40 mph or more"]
SPEED_COLOURS = ["#3b6fb6", "#30AB89", "#E0B94A", "#DB5461"]
GRAPH_COLOURS = {"none": "#8a8a8a", "proximity": "#E0B94A", "streets": "#3b6fb6"}
FEATURE_SETS = ("geometry and embedding", "with the road class", "road class only", "none")


# ---------------------------------------------------------------------------------------------------------------
# graphs as matrices
# ---------------------------------------------------------------------------------------------------------------

def adjacency(edges, nodes):
    """The symmetric adjacency matrix of an undirected graph, A[i, j] = 1 where i and j share an edge.

    edges holds pairs of node names (further entries of an edge, a length for one, are ignored), nodes the
    names in the order of the rows.
    """
    index = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for e in edges:
        i, j = index[e[0]], index[e[1]]
        A[i, j] = A[j, i] = 1.0
    return A


def line_graph(edges):
    """The line graph of an undirected graph: every edge becomes a node, and two of them are joined when the
    edges share an end point. For a street network the nodes are then the street segments, joined where they meet.

    edges holds pairs of node names; returns the adjacency matrix in the order of the edges.
    """
    at = {}
    for k, e in enumerate(edges):
        for v in (e[0], e[1]):
            at.setdefault(v, []).append(k)
    A = np.zeros((len(edges), len(edges)))
    for members in at.values():
        for i in members:
            for j in members:
                if i != j:
                    A[i, j] = 1.0
    return A


def proximity_graph(xy, m):
    """A graph that knows distance and nothing else: the m closest pairs of points are joined.

    xy holds one point per node; with m the number of edges of another graph on the same nodes, the two graphs
    can be compared edge for edge.
    """
    xy = np.asarray(xy, dtype=float)
    dx = xy[:, None, 0] - xy[None, :, 0]
    dy = xy[:, None, 1] - xy[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    i, j = np.triu_indices(len(xy), 1)
    nearest = np.argsort(dist[i, j], kind="stable")[:m]
    A = np.zeros((len(xy), len(xy)))
    A[i[nearest], j[nearest]] = 1.0
    A[j[nearest], i[nearest]] = 1.0
    return A


def hops(A, source):
    """The number of edges on a shortest path from the source node to every node (infinite where there is none)."""
    dist = np.full(len(A), np.inf)
    dist[source] = 0
    frontier = np.zeros(len(A), dtype=bool)
    frontier[source] = True
    k = 0
    while frontier.any():
        k += 1
        frontier = (np.dot(A, frontier.astype(float)) > 0) & np.isinf(dist)
        dist[frontier] = k
    return dist


def degrees(A):
    """The degree of every node, the row sums of A."""
    return A.sum(axis=1)


def add_self_loops(A):
    """A with ones on the diagonal, so that a node also hears itself."""
    return A + np.eye(len(A))


def normalise(A, kind="sym", self_loops=True):
    """The propagation matrix of one round of message passing.

    kind "sum" returns A (every neighbour counts fully), "mean" D^-1 A (the average over the neighbours),
    "sym" D^-1/2 A D^-1/2 (the weight of the edge i j is 1 / sqrt(d_i d_j)), the rule of the graph
    convolutional network. With self_loops the identity is added first and the degrees count the loop.
    """
    A = add_self_loops(A) if self_loops else np.asarray(A, dtype=float)
    d = A.sum(axis=1)
    safe = np.where(d > 0, d, 1.0)
    if kind == "sum":
        return A
    if kind == "mean":
        return A / safe[:, None]
    if kind == "sym":
        s = 1.0 / np.sqrt(safe)
        return A * s[:, None] * s[None, :]
    raise ValueError(f"unknown kind {kind!r}")


def laplacian(A, normalised=False):
    """The graph Laplacian D - A, or the normalised I - D^-1/2 A D^-1/2."""
    d = A.sum(axis=1)
    if not normalised:
        return np.diag(d) - A
    s = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    return np.eye(len(A)) - A * s[:, None] * s[None, :]


def message_passing(A, h, rounds=1, kind="sum", self_loops=False, g=None):
    """The states of all nodes after every round of message passing, a list starting with h itself.

    One round replaces the state of a node by the aggregate of its neighbours' states under normalise(A, kind,
    self_loops) and then applies g, a function of one number (None leaves the aggregate as it is).
    """
    P = normalise(A, kind, self_loops)
    states = [np.asarray(h, dtype=float)]
    for _ in range(rounds):
        nxt = np.dot(P, states[-1])
        states.append(g(nxt) if g else nxt)
    return states


def k_hop(A, source, k):
    """A boolean vector, True for the nodes within k edges of the source node."""
    reach = np.zeros(len(A), dtype=bool)
    reach[source] = True
    for _ in range(k):
        reach = reach | (np.dot(A, reach.astype(float)) > 0)
    return reach


def laplacian_eigenmaps(A, k=6):
    """The k eigenvectors of the normalised Laplacian with the smallest eigenvalues, and those eigenvalues."""
    values, vectors = np.linalg.eigh(laplacian(A, normalised=True))
    return values[:k], vectors[:, :k]


def dirichlet_energy(A, H):
    """How much the rows of H differ across the edges: the mean squared difference over all edges.

    The value falls towards zero when message passing has made all neighbours alike (over-smoothing).
    """
    i, j = np.nonzero(np.triu(A, 1))
    diff = H[i] - H[j]
    return float((diff * diff).sum(axis=1).mean())


def one_hot(n):
    """The identity matrix as node features, for a graph whose nodes carry nothing but their name."""
    return np.eye(n)


def softmax(Z):
    """The softmax of every row of Z."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def gcn_layer(A, H, W, activation="relu"):
    """One graph convolutional layer in NumPy: sigma(D~^-1/2 A~ D~^-1/2 H W), with A~ = A + I."""
    Z = np.dot(np.dot(normalise(A, "sym", True), H), W)
    if activation == "relu":
        return np.maximum(Z, 0.0)
    if activation == "softmax":
        return softmax(Z)
    return Z


def cross_entropy(P, y, labelled):
    """The mean of minus log P[i, y_i] over the labelled nodes."""
    idx = np.asarray(labelled)
    return float(-np.log(P[idx, np.asarray(y)[idx]]).mean())


def accuracy(pred, y, idx=None):
    """The share of the nodes in idx (all nodes if None) whose predicted class is the true one."""
    pred, y = np.asarray(pred), np.asarray(y)
    if idx is not None:
        pred, y = pred[idx], y[idx]
    return float((pred == y).mean())


def confusion(pred, y, n_classes, idx=None):
    """The error matrix, rows the true class and columns the predicted class."""
    pred, y = np.asarray(pred), np.asarray(y)
    if idx is not None:
        pred, y = pred[idx], y[idx]
    M = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(M, (y, pred), 1)
    return M


def draw_labels(y, pool, per_class, rng):
    """per_class labelled nodes of every class drawn from the pool (all of a class that has fewer)."""
    y, pool = np.asarray(y), np.asarray(pool)
    out = []
    for c in np.unique(y[pool]):
        members = pool[y[pool] == c]
        out.extend(rng.choice(members, size=min(per_class, len(members)), replace=False).tolist())
    return np.array(sorted(out))


def draw_random(pool, n, rng):
    """n labelled nodes drawn at random from the pool, so that the classes keep their natural shares."""
    pool = np.asarray(pool)
    return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))


# ---------------------------------------------------------------------------------------------------------------
# the models (PyTorch)
# ---------------------------------------------------------------------------------------------------------------

if torch_ok:

    class GraphConvolution(nn.Module):
        """The layer of Kipf and Welling: H' = P H W + b, with P the normalised adjacency with self-loops."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.linear = nn.Linear(d_in, d_out)

        def forward(self, H, P):
            return torch.matmul(P, self.linear(H))

    class GCN(nn.Module):
        """Graph convolutional layers with ReLU and dropout between them; the last layer returns class scores."""

        def __init__(self, d_in, d_hidden, n_classes, layers=2, dropout=0.5):
            super().__init__()
            sizes = [d_in] + [d_hidden] * (layers - 1) + [n_classes]
            self.convs = nn.ModuleList(GraphConvolution(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
            self.dropout = dropout
            self.hidden = None

        def forward(self, X, P):
            H = X
            for k, conv in enumerate(self.convs):
                H = nn.functional.dropout(H, self.dropout, self.training)
                H = conv(H, P)
                if k < len(self.convs) - 1:
                    H = torch.relu(H)
                    self.hidden = H
            return H

    class MLP(nn.Module):
        """The model without a graph: the same layers with the identity in place of the adjacency."""

        def __init__(self, d_in, d_hidden, n_classes, layers=2, dropout=0.5):
            super().__init__()
            sizes = [d_in] + [d_hidden] * (layers - 1) + [n_classes]
            self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
            self.dropout = dropout
            self.hidden = None

        def forward(self, X, P=None):
            H = X
            for k, linear in enumerate(self.linears):
                H = nn.functional.dropout(H, self.dropout, self.training)
                H = linear(H)
                if k < len(self.linears) - 1:
                    H = torch.relu(H)
                    self.hidden = H
            return H

    def fit(model, X, P, y, train, val, epochs=200, lr=0.01, weight_decay=5e-4, seed=SEED, watch=()):
        """Train on the labelled nodes and keep the weights of the epoch with the lowest validation loss.

        X the features, P the propagation matrix (ignored by the MLP), y the classes of all nodes, train and val
        index arrays. The loss is the cross-entropy over the train nodes only. Returns a dict with the curves,
        the best epoch, the predicted class and the class probabilities of every node, the hidden states of the
        last hidden layer, and the predictions at the epochs listed in watch.
        """
        torch.manual_seed(seed)
        for p in model.parameters():                      # a fresh start from the seed
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        whole = P is not None and np.asarray(P).dtype.kind in "iu"     # an edge list of PyTorch Geometric stays whole numbers
        P = None if P is None else torch.as_tensor(np.asarray(P), dtype=torch.long if whole else torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        train, val = torch.as_tensor(np.asarray(train)), torch.as_tensor(np.asarray(val))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        hist = {"train_loss": [], "val_loss": [], "val_acc": [], "watch": {}}
        best, best_state = float("inf"), None
        for epoch in range(1, epochs + 1):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(X, P)[train], y[train])
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                out = model(X, P)
                vl = float(loss_fn(out[val], y[val]))
                hist["train_loss"].append(float(loss))
                hist["val_loss"].append(vl)
                hist["val_acc"].append(float((out[val].argmax(1) == y[val]).float().mean()))
                if epoch in watch:
                    hist["watch"][epoch] = out.argmax(1).numpy().copy()
                if vl < best:
                    best, hist["best_epoch"] = vl, epoch
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            out = model(X, P)
            hist["pred"] = out.argmax(1).numpy()
            hist["prob"] = torch.softmax(out, 1).numpy()
            hist["hidden"] = None if model.hidden is None else model.hidden.numpy().copy()
        return hist

    def count_parameters(model):
        """The number of learned numbers of a model."""
        return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------------------------------------------
# graphs over time: the sensor graph and the forecasting baselines
# ---------------------------------------------------------------------------------------------------------------

def kernel_adjacency(dist, threshold=0.1):
    """The weighted sensor graph of Li et al. (2018): W_ij = exp(-dist_ij^2 / sigma^2), set to zero below the threshold.

    dist holds the road distance from sensor i to sensor j (infinite where none was computed), sigma is the
    standard deviation of the finite distances.
    """
    finite = np.isfinite(dist)
    sigma = dist[finite].std()
    W = np.where(finite, np.exp(-np.square(np.where(finite, dist, 0.0) / sigma)), 0.0)
    W[W < threshold] = 0.0
    return W


def masked_mae(pred, truth):
    """The mean absolute error over the readings that exist (a reading of zero is a missing one in METR-LA)."""
    ok = truth > 0
    return float(np.abs(pred - truth)[ok].mean())


def masked_rmse(pred, truth):
    """The root mean squared error over the readings that exist."""
    ok = truth > 0
    return float(np.sqrt(np.square(pred - truth)[ok].mean()))


def masked_mape(pred, truth):
    """The mean absolute percentage error over the readings that exist, in percent."""
    ok = truth > 0
    return float(100 * (np.abs(pred - truth)[ok] / truth[ok]).mean())


def fill_missing(X, start=None):
    """Readings with every missing value (a zero) replaced by the last reading of the same sensor that exists.

    X holds time in the rows and sensors in the columns. A sensor that starts with missing readings takes start
    (one number per sensor, its mean over the training weeks for instance) until its first reading; with start None
    those stay at zero. This is what a forecaster knows at every moment: the last value it has seen.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()
    last = np.zeros(X.shape[1]) if start is None else np.asarray(start, dtype=float).copy()
    for t in range(len(X)):
        ok = X[t] > 0
        last[ok] = X[t, ok]
        out[t] = last
    return out


def lag_windows(X, times, lags=12):
    """The last `lags` readings before and including each time in `times`, an array of shape (times, lags, sensors).

    Window k of time t holds X[t - lags + 1 + k], so its last entry is the reading at t itself.
    """
    times = np.asarray(times)
    idx = times[:, None] + np.arange(-lags + 1, 1)[None, :]
    return np.asarray(X)[idx]


def time_of_week(seconds, step=300):
    """The slot of the week of every time stamp (seconds since 1970, a Thursday): 0 for Monday 0:00, in steps of `step` s."""
    seconds = np.asarray(seconds, dtype=np.int64)
    day = (seconds // 86400 + 3) % 7                     # 1 January 1970 was a Thursday, day 3 when Monday is 0
    return (day * 86400 + seconds % 86400) // step


def historical_average(X, slots, train):
    """The mean reading of every sensor in every slot of the week over the training times, missing readings left out.

    Returns an array of shape (number of slots, sensors); a slot never observed takes the sensor's training mean.
    """
    X = np.asarray(X, dtype=float)
    slots = np.asarray(slots)
    n_slots = int(slots.max()) + 1
    total = np.zeros((n_slots, X.shape[1])); count = np.zeros((n_slots, X.shape[1]))
    Xt, st = X[train], slots[train]
    np.add.at(total, st, np.where(Xt > 0, Xt, 0.0))
    np.add.at(count, st, (Xt > 0).astype(float))
    overall = np.where(Xt > 0, Xt, 0.0).sum(0) / np.maximum((Xt > 0).sum(0), 1)
    return np.where(count > 0, total / np.maximum(count, 1), overall[None, :])


def neighbour_average(X, W):
    """The weighted mean of the neighbours' readings for every sensor: X W^T with the rows of W scaled to sum to one.

    W[i, j] is the weight of sensor j for sensor i, its diagonal is ignored. A sensor without a neighbour gets its
    own reading, so the extra input carries no information for it.
    """
    W = np.array(W, dtype=float)
    np.fill_diagonal(W, 0.0)
    s = W.sum(axis=1)
    R = W / np.where(s > 0, s, 1.0)[:, None]
    out = np.tensordot(np.asarray(X, dtype=float), R, axes=([-1], [1]))
    return np.where(s > 0, out, X)


def ridge_fit(F, y, alpha=1.0):
    """Ridge regression of y on the columns of F with an intercept that is not penalised.

    Returns (w, b) minimising the sum of (y - F w - b)^2 plus alpha times the sum of w^2.
    """
    F = np.asarray(F, dtype=float); y = np.asarray(y, dtype=float)
    mu, my = F.mean(axis=0), y.mean()
    Fc = F - mu
    G = np.dot(Fc.T, Fc) + alpha * np.eye(F.shape[1])
    w = np.linalg.solve(G, np.dot(Fc.T, y - my))
    return w, float(my - np.dot(mu, w))


def forecast_origins(n_times, lags=12, horizon=12, train=0.7, val=0.1):
    """The split of a series in time and the forecast origins of each part, as in Li et al. (2018).

    The first `train` share of the times is for fitting, the next `val` share for choosing, the rest for testing.
    An origin needs a full hour of readings behind it and the furthest horizon ahead of it inside the same part.
    Returns (train times, validation times, test times) and (train origins, validation origins, test origins).
    """
    n_train, n_val = int(round(train * n_times)), int(round(val * n_times))
    times = (np.arange(n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n_times))
    origins = (np.arange(lags - 1, n_train - horizon), np.arange(n_train + lags - 1, n_train + n_val - horizon),
               np.arange(n_train + n_val + lags - 1, n_times - horizon))
    return times, origins


def forecast_inputs(Xf, NB, t, i, neighbours=False, clock=None, h=0, lags=12):
    """The inputs of sensor i at the origins t for a forecast h steps ahead.

    Its own last `lags` readings from the gap-filled series Xf, then (neighbours) the neighbours' mean NB over the
    same hour, then (clock, the historical average of every time by sensor) the historical average of the target time.
    """
    cols = [lag_windows(Xf[:, i], t, lags)]
    if neighbours:
        cols.append(lag_windows(NB[:, i], t, lags))
    if clock is not None:
        cols.append(clock[t + h, i][:, None])
    return np.hstack(cols)


def ridge_forecast(Xf, NB, Y, origins, h, neighbours=False, alphas=(1.0,), clock=None, lags=12):
    """A ridge regression per sensor for the horizon of h steps, the penalty chosen on the validation origins.

    Xf the gap-filled readings, NB the neighbours' mean, Y the readings as recorded (a zero is a missing one and
    is left out of the fit and of the error), origins the triple of forecast_origins. With one value in alphas no
    choice is made. Returns the predictions at the test origins (origins by sensors), the penalty kept and the
    validation error of every candidate.
    """
    tr, va, te = origins
    n = Xf.shape[1]
    val_err = {a: [] for a in alphas}
    models = {a: [] for a in alphas}
    for i in range(n):
        F = forecast_inputs(Xf, NB, tr, i, neighbours, clock, h, lags); y = Y[tr + h, i]; ok = y > 0
        Fv = forecast_inputs(Xf, NB, va, i, neighbours, clock, h, lags); yv = Y[va + h, i]
        for a in alphas:
            w, b = ridge_fit(F[ok], y[ok], a)
            models[a].append((w, b))
            pv = np.dot(Fv, w) + b
            m = yv > 0
            val_err[a].append((np.abs(pv - yv)[m].sum(), m.sum()))
    score = {a: sum(e for e, _ in val_err[a]) / max(sum(c for _, c in val_err[a]), 1) for a in alphas}
    best = min(score, key=score.get)
    P = np.zeros((len(te), n))
    for i in range(n):
        w, b = models[best][i]
        P[:, i] = np.dot(forecast_inputs(Xf, NB, te, i, neighbours, clock, h, lags), w) + b
    return P, best, score


def sensor_mae(pred, truth):
    """The mean absolute error of every sensor over the readings that exist, one number per column."""
    return np.array([np.abs(pred[:, i] - truth[:, i])[truth[:, i] > 0].mean() for i in range(truth.shape[1])])


def symmetric(W):
    """A weighted graph read in both directions: the larger of the two weights of every pair."""
    return np.maximum(W, W.T)


# ---------------------------------------------------------------------------------------------------------------
# the street experiment of Lecture 08: features, graphs, splits and repeated draws
# ---------------------------------------------------------------------------------------------------------------

def standardise(F):
    """Every column minus its mean and divided by its standard deviation, so no feature outweighs another by its unit."""
    return (F - F.mean(0)) / (F.std(0) + 1e-9)


def street_features(s, kind="geometry and embedding"):
    """One of the four feature sets of the lecture's table for the street segments s (see FEATURE_SETS).

    "geometry and embedding" is the logarithm of the length, the bearing as two numbers and the 64 numbers of the
    satellite embedding at the midpoint; "with the road class" adds the road class as four columns of zeros and
    ones; "road class only" keeps those four; "none" gives every segment its own column, so a model can learn only
    from the graph.
    """
    road = np.eye(4)[s["road_class"]]
    if kind == "geometry and embedding":
        return standardise(np.c_[s["geometry"], s["embedding"]])
    if kind == "with the road class":
        return standardise(np.c_[s["geometry"], s["embedding"], road])
    if kind == "road class only":
        return standardise(road)
    if kind == "none":
        return one_hot(len(s["y"]))
    raise ValueError(f"kind is one of {FEATURE_SETS}")


def street_graphs(s):
    """The three graphs of the comparison as propagation matrices: None, the proximity graph and the street graph.

    Both graphs are normalised symmetrically with self-loops. The proximity graph joins the closest pairs of
    midpoints and has exactly as many edges as the street graph, so the two differ in which pairs they join only.
    """
    return {"none": None, "proximity": normalise(s["A_near"].astype(float)), "streets": normalise(s["A_street"].astype(float))}


def street_classes(s):
    """The speed limit class of every segment as 0 to 3; an untagged segment carries 0 and is never scored."""
    return np.where(s["y"] < 0, 0, s["y"])


def split_streets(s, kind, rng, n_labelled=150, n_val=100):
    """Labelled, validation and test segments among the tagged ones, three sorted index arrays.

    kind "random" shuffles the tagged segments and cuts them in three. kind "blocks" tests on the tagged segments
    inside Lecture 04's held-out blocks and draws the labelled and the validation segments outside them, so no
    test segment has a labelled neighbour across the street.
    """
    tagged = np.nonzero(s["tagged"])[0]
    if kind == "random":
        perm = rng.permutation(tagged)
        return np.sort(perm[:n_labelled]), np.sort(perm[n_labelled:n_labelled + n_val]), np.sort(perm[n_labelled + n_val:])
    inside = tagged[s["test"][tagged]]
    perm = rng.permutation(tagged[~s["test"][tagged]])
    return np.sort(perm[:n_labelled]), np.sort(perm[n_labelled:n_labelled + n_val]), np.sort(inside)


def balanced_accuracy(pred, y, n_classes, idx=None):
    """The mean over the classes of the share of each class named right, so a rare class counts like a common one."""
    M = confusion(pred, y, n_classes, idx)
    return float(np.mean([M[c, c] / M[c].sum() for c in range(n_classes) if M[c].sum() > 0]))


def majority_accuracy(y, labelled, test):
    """The score of the rule that names every test node by the most common class among the labelled nodes."""
    return float((y[test] == np.bincount(y[labelled]).argmax()).mean())


def summary(values):
    """Mean, standard deviation, smallest and largest of a list of scores."""
    v = np.asarray(values)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def steps_to(A, marked):
    """For every node the number of edges to the nearest marked node: 0 for a marked one, -1 where none is reached."""
    reach = np.asarray(marked, dtype=bool).copy()
    steps = np.where(reach, 0, -1)
    A = np.asarray(A, dtype=float)
    for k in range(1, len(A)):
        wider = reach | (np.dot(A, reach.astype(float)) > 0)
        if (wider == reach).all():
            break
        steps[wider & ~reach] = k
        reach = wider
    return steps


def homophily(A, y, known=None):
    """The share of edges whose two ends carry the same class, over the edges with both ends in `known`."""
    i, j = np.nonzero(np.triu(np.asarray(A), 1))
    if known is not None:
        ok = np.asarray(known)[i] & np.asarray(known)[j]
        i, j = i[ok], j[ok]
    return float((np.asarray(y)[i] == np.asarray(y)[j]).mean())


def lookup_street(s, name):
    """The segments whose street name contains `name`, upper and lower case read alike, as an index array."""
    return np.nonzero(np.char.find(np.char.lower(s["name"].astype(str)), name.lower()) >= 0)[0]


if torch_ok:

    def train(F, P, y, labelled, val, seed=SEED, hidden=32, layers=2, epochs=200, watch=()):
        """One training run of the lecture: a GCN on the propagation matrix P, or the model without a graph for P None.

        Both models have the same layers and the same number of weights. Returns the result of fit and the model.
        """
        Model = MLP if P is None else GCN
        model = Model(F.shape[1], hidden, int(np.max(y)) + 1, layers=layers)
        return fit(model, F, P, y, labelled, val, epochs=epochs, seed=seed, watch=watch), model

    def run_draws(s, F, P, kind="random", draws=20, n_labelled=150, n_val=100, first=0, layers=2):
        """The test accuracy of `draws` training runs, each with its own labelled segments and its own start.

        Draw d splits with the seed SEED + 100 + d and starts the weights from SEED + d, the lecture's numbering,
        so the draws 0 to 19 are the lecture's twenty and `first=20` gives draws nobody has seen. Returns a dict
        with the accuracies, the balanced accuracies and the majority rule's score, one entry per draw.
        """
        y = street_classes(s)
        out = {"accuracy": [], "balanced": [], "majority": []}
        for d in range(first, first + draws):
            lab, val, test = split_streets(s, kind, np.random.default_rng(SEED + 100 + d), n_labelled, n_val)
            h, _ = train(F, P, y, lab, val, SEED + d, layers=layers)
            out["accuracy"].append(accuracy(h["pred"], y, test))
            out["balanced"].append(balanced_accuracy(h["pred"], y, int(y.max()) + 1, test))
            out["majority"].append(majority_accuracy(y, lab, test))
        return out


# ---------------------------------------------------------------------------------------------------------------
# the same layer from PyTorch Geometric
# ---------------------------------------------------------------------------------------------------------------

def pyg_available():
    """True when PyTorch and PyTorch Geometric can both be imported."""
    if not torch_ok:
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False


def to_edge_index(A):
    """A dense adjacency matrix as the edge list of PyTorch Geometric: two rows of node numbers, one column per
    direction of every edge. The library stores a graph this way because most pairs of nodes are not joined."""
    return np.stack(np.nonzero(np.asarray(A))).astype(np.int64)


if torch_ok:

    def pyg_gcn(d_in, d_hidden, n_classes, dropout=0.5):
        """The two-layer GCN of the lecture built from the library's GCNConv, called as model(X, edge_index).

        GCNConv adds the self-loops and normalises symmetrically itself, so it takes the bare edge list. It adds
        its bias after the neighbourhood average, where the layer written by hand adds it before; with a bias of
        zero the two layers are the same function.
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch_geometric.nn import GCNConv

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.convs = nn.ModuleList([GCNConv(d_in, d_hidden), GCNConv(d_hidden, n_classes)])
                self.dropout, self.hidden = dropout, None

            def forward(self, X, edge_index):
                H = nn.functional.dropout(X, self.dropout, self.training)
                H = torch.relu(self.convs[0](H, edge_index))
                self.hidden = H
                return self.convs[1](nn.functional.dropout(H, self.dropout, self.training), edge_index)

        return Net()

    def copy_weights(by_hand, library):
        """Give the library's GCN the weights of the GCN written by hand and set every bias of both to zero."""
        with torch.no_grad():
            for mine, theirs in zip(by_hand.convs, library.convs):
                theirs.lin.weight.copy_(mine.linear.weight)
                mine.linear.bias.zero_()
                theirs.bias.zero_()

    def scores(model, X, P):
        """The class scores of a model for every node, without dropout: model(X, P) as a NumPy array."""
        whole = np.asarray(P).dtype.kind in "iu"
        model.eval()
        with torch.no_grad():
            out = model(torch.as_tensor(np.asarray(X), dtype=torch.float32),
                        torch.as_tensor(np.asarray(P), dtype=torch.long if whole else torch.float32))
        return out.numpy()


# ---------------------------------------------------------------------------------------------------------------
# the kit's files
# ---------------------------------------------------------------------------------------------------------------

def load_json(path):
    """A json file of the kit as a dict."""
    with open(path) as f:
        return json.load(f)


def load_arrays(path):
    """An npz file of the kit as a dict of arrays; text arrays come back as text, nothing is unpickled."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_segments(path="data/streets.gpkg"):
    """The street segments as lines in metres (CRS), one row per node of the street graph, in the node order."""
    import geopandas as gpd
    return gpd.read_file(path).sort_values("segment").reset_index(drop=True)


def load_toy(path="data/toy.json"):
    """The seven places of Lecture 07 as the lecture reads them: nodes, edges, the adjacency matrix A, the feature
    x (places to eat), the zones, the classes y and the three labelled places, as NumPy arrays where it helps."""
    t = load_json(path)
    return {"nodes": t["nodes"], "edges": [tuple(e) for e in t["edges"]], "A": np.array(t["A"], dtype=float),
            "x": np.array(t["x"], dtype=float), "zones": t["zones"], "y": np.array(t["y"]),
            "labelled": np.array(t["labelled"]), "layout": {k: tuple(v) for k, v in t["layout"].items()}}


# ---------------------------------------------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------------------------------------------

def _axes(ax, size=(5, 4)):
    import matplotlib.pyplot as plt
    return ax if ax is not None else plt.subplots(figsize=size)[1]


def show_graph(A, xy, labels=None, values=None, colours=None, ax=None, title=None, size=420):
    """A small graph drawn with its nodes at xy: labels inside the nodes, values (numbers or text) beside them."""
    ax = _axes(ax)
    xy = np.asarray(xy, dtype=float)
    for i, j in zip(*np.nonzero(np.triu(np.asarray(A), 1))):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#555555", lw=1.4, zorder=1)
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c="#6495ED" if colours is None else colours, edgecolors="#222222", zorder=2)
    span = np.ptp(xy[:, 1]) + 1e-9
    for k in range(len(xy)):
        if labels is not None:
            ax.text(xy[k, 0], xy[k, 1], str(labels[k]), ha="center", va="center", color="white", weight="bold", zorder=3)
        if values is not None:
            v = values[k]
            ax.text(xy[k, 0], xy[k, 1] + 0.11 * span, v if isinstance(v, str) else f"{v:.3g}", ha="center", va="bottom", color="#DB5461")
    ax.margins(0.18)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_matrix(M, labels=None, ax=None, title=None, fmt="{:.0f}", cmap="Blues"):
    """A matrix as a grid of shaded cells with every number written in its cell."""
    M = np.asarray(M, dtype=float)
    ax = _axes(ax, (0.5 * M.shape[1] + 1.2, 0.5 * M.shape[0] + 1))
    ax.imshow(M, cmap=cmap, vmin=min(0.0, M.min()), vmax=max(M.max(), 1e-9) * 1.6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8)
    if labels is not None:
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(labels[:M.shape[1]])
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(labels[:M.shape[0]])
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_streets(segments, classes=None, values=None, mark=None, ax=None, title=None, legend=True, cmap="viridis", lw=1.6):
    """The street segments as a map. classes colours them by speed limit class (-1 drawn pale), values by a number
    per segment, mark draws the listed segments thick and dark on top."""
    from matplotlib.lines import Line2D
    ax = _axes(ax, (6, 6))
    if values is not None:
        segments.assign(v=np.asarray(values)).plot(column="v", cmap=cmap, lw=lw, ax=ax, legend=legend, legend_kwds={"shrink": 0.6})
    else:
        c = np.full(len(segments), -1) if classes is None else np.asarray(classes)
        segments[c < 0].plot(color="#cfcfcf", lw=0.8, ax=ax)
        for k in range(4):
            if (c == k).any():
                segments[c == k].plot(color=SPEED_COLOURS[k], lw=lw, ax=ax)
        if legend and (c >= 0).any():
            keys = [Line2D([], [], color=SPEED_COLOURS[k], lw=2.5, label=SPEED_NAMES[k]) for k in range(4)]
            keys.append(Line2D([], [], color="#cfcfcf", lw=1.5, label="no limit on the map"))
            ax.legend(handles=keys, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8)
    if mark is not None and len(np.atleast_1d(mark)):
        segments.iloc[np.atleast_1d(mark)].plot(color="#111111", lw=3.2, ax=ax, zorder=5)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_edges(xy, A, ax=None, colour="#3b6fb6", title=None, window=None, lw=0.8):
    """A graph over points in the plane: a line for every edge of A between the points xy, the points as dots.
    window (x0, x1, y0, y1) crops the picture."""
    ax = _axes(ax, (5, 5))
    xy = np.asarray(xy, dtype=float)
    i, j = np.nonzero(np.triu(np.asarray(A), 1))
    ax.plot(np.stack([xy[i, 0], xy[j, 0]]), np.stack([xy[i, 1], xy[j, 1]]), color=colour, lw=lw)
    ax.scatter(xy[:, 0], xy[:, 1], s=6, color="#222222", zorder=3)
    if window is not None:
        ax.set_xlim(window[0], window[1]); ax.set_ylim(window[2], window[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_curves(runs, ax=None, title=None, key="val_loss"):
    """The loss of one or more training runs against the epoch; runs maps a name to the result of fit or to a list.
    A dot marks the epoch whose weights were kept."""
    ax = _axes(ax, (5.5, 3.2))
    for name, h in runs.items():
        v = np.asarray(h[key] if isinstance(h, dict) else h, dtype=float)
        line, = ax.plot(np.arange(1, len(v) + 1), v, label=name, color=GRAPH_COLOURS.get(name))
        if isinstance(h, dict) and "best_epoch" in h:
            ax.plot(h["best_epoch"], v[h["best_epoch"] - 1], "o", color=line.get_color())
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss" if key == "val_loss" else key.replace("_", " "))
    ax.legend(frameon=False, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_bars(groups, ax=None, title=None, baseline=None, ylabel="percent correct on the test segments", names=None):
    """Bars with the spread over the draws. groups maps a label to a dict of name to a list of scores (shares, drawn
    as percent); one cluster of bars per label, one bar per name, a whisker of one standard deviation, and a
    dashed line at baseline (a share) for the majority rule."""
    ax = _axes(ax, (1.6 * len(groups) + 2.4, 3.4))
    names = names or list(next(iter(groups.values())).keys())
    width = 0.8 / len(names)
    for g, (label, row) in enumerate(groups.items()):
        for k, name in enumerate(names):
            v = 100 * np.atleast_1d(np.asarray(row[name], dtype=float))
            x = g + (k - (len(names) - 1) / 2) * width
            ax.bar(x, v.mean(), width * 0.92, color=GRAPH_COLOURS.get(name, "#6495ED"), yerr=v.std(ddof=1) if len(v) > 1 else None,
                   capsize=2, label=name if g == 0 else None)
            ax.text(x, v.mean() + (v.std(ddof=1) if len(v) > 1 else 0) + 0.8, f"{v.mean():.1f}", ha="center", fontsize=7)
    if baseline is not None:
        ax.axhline(100 * baseline, color="#222222", ls="--", lw=0.8)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(list(groups.keys()), fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8); ax.set_ylim(40, None)
    ax.legend(frameon=False, fontsize=8, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_scatter(xy, classes, ax=None, title=None, names=None, dim=None):
    """Points in two dimensions coloured by class, for a hidden space reduced to two numbers. dim marks the points
    to draw pale (the segments without a label, for instance)."""
    ax = _axes(ax, (4.2, 4.2))
    xy, c = np.asarray(xy), np.asarray(classes)
    pale = np.zeros(len(c), dtype=bool) if dim is None else np.asarray(dim, dtype=bool)
    ax.scatter(xy[pale, 0], xy[pale, 1], s=5, color="#d9d9d9")
    for k in range(int(c.max()) + 1):
        m = (c == k) & ~pale
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=SPEED_COLOURS[k % 4], label=(names or SPEED_NAMES)[k])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def two_numbers(H):
    """The two directions of largest spread of the rows of H (a principal component analysis), n by 2."""
    H = np.asarray(H, dtype=float)
    U, S, _ = np.linalg.svd(H - H.mean(0), full_matrices=False)
    return U[:, :2] * S[:2]


def show_sensors(xy, values=None, edges=None, mark=None, ax=None, title=None, label="mph", cmap="RdYlGn"):
    """The sensors as dots at xy (metres), coloured by a number per sensor, with the graph's edges as thin lines."""
    ax = _axes(ax, (6, 4.6))
    xy = np.asarray(xy, dtype=float) / 1000
    if edges is not None:
        e = np.asarray(edges)
        ax.plot(np.stack([xy[e[:, 0], 0], xy[e[:, 1], 0]]), np.stack([xy[e[:, 0], 1], xy[e[:, 1], 1]]), color="#b5b5b5", lw=0.4, zorder=1)
    dots = ax.scatter(xy[:, 0], xy[:, 1], s=14, c="#3b6fb6" if values is None else values, cmap=None if values is None else cmap, zorder=2)
    if values is not None:
        ax.figure.colorbar(dots, ax=ax, shrink=0.7, label=label)
    if mark is not None:
        m = np.atleast_1d(mark)
        ax.scatter(xy[m, 0], xy[m, 1], s=70, facecolors="none", edgecolors="#111111", lw=1.4, zorder=3)
    ax.set_aspect("equal"); ax.set_xlabel("km east"); ax.set_ylabel("km north")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def show_series(series, minutes=None, ax=None, title=None, ylabel="speed in mph"):
    """One or more series against time; series maps a name to an array, minutes gives the time of every reading
    (in days when it spans more than two days)."""
    ax = _axes(ax, (7, 2.8))
    for name, v in series.items():
        t = np.arange(len(v)) * 5 if minutes is None else np.asarray(minutes, dtype=float)
        days = t.max() - t.min() > 2 * 1440
        ax.plot(t / 1440 if days else t, np.where(np.asarray(v) > 0, v, np.nan), lw=1.0, label=name)
    ax.set_xlabel("days" if days else "minutes"); ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# --- additions for 08_01 (its writer appends below this line and nowhere else) ---

def show_toy(A, xy, labels=None, values=None, colours=None, ax=None, title=None, size=420):
    """A small graph drawn as show_graph draws it, with every value beside a node on a white patch, so that no edge
    running past the node crosses the number."""
    ax = show_graph(A, xy, labels=labels, values=values, colours=colours, ax=ax, title=title, size=size)
    for text in ax.texts:
        if text.get_color() == "#DB5461":
            text.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9})
            text.set_zorder(4)
    return ax



# --- additions for 08_02 (its writer appends below this line and nowhere else) ---


# --- additions for 08_03 (its writer appends below this line and nowhere else) ---


# --- additions for 08_04 (its writer appends below this line and nowhere else) ---


__all__ = sorted(k for k, v in list(globals().items()) if not k.startswith("_") and k not in ("json", "np", "torch", "nn")
                 and (callable(v) or k.isupper() or k == "torch_ok"))
