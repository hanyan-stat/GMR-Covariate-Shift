"""Fix the GAMA predictor and estimate the two pointwise targets and their benchmarks."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.linalg import pinvh, svd
from scipy.optimize import brentq
from scipy.stats import binom, norm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import isotonic_regression
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

from simulation import inference as inf, nuisance as nu, simulate as sim

PREDICTOR_SEED = 2026112201
SEED = 2026112203
TABLE_RATIO_POLICY = dict(
    sim.BASE_R,
    maximum_updates=3000,
    minimum_updates=80,
    patience=8,
    cap_multiplier=8,
    widths=[96],
    refit_learning_rate=0.00025,
)


@dataclass
class Study:
    xp: np.ndarray
    xq: np.ndarray
    redshift: np.ndarray
    predictor: ExtraTreesRegressor
    hp: np.ndarray
    hq: np.ndarray
    residual: np.ndarray
    error: np.ndarray
    sky_p: np.ndarray
    sky_q: np.ndarray
    folds: list


def fix_predictor(sample):
    """Train on the independent fifth of the source sample and freeze all five source folds."""
    xp, yp, xq, sky_p, sky_q = sample
    perm = np.random.default_rng(PREDICTOR_SEED).permutation(len(yp))
    train, use = perm[: len(yp) // 5], np.sort(perm[len(yp) // 5 :])
    # The original analysis stored redshifts in units of 0.1 before fitting h.
    redshift = (yp * 10.0) / 10.0
    h = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        max_features=1.0,
        n_jobs=1,
        random_state=PREDICTOR_SEED,
    ).fit(xp[train], redshift[train])
    hp, hq = h.predict(xp[use]), h.predict(xq)
    residual = (redshift[use] - hp) * 10.0 / (1.0 + hp)
    error = np.abs(residual / (10.0 + residual))
    folds = np.array_split(np.random.default_rng(PREDICTOR_SEED + 5000).permutation(len(use)), 5)
    return Study(xp[use], xq, redshift[use], h, hp, hq, residual, error, sky_p[use], sky_q, folds)


def complement(folds, index):
    """Return the training indices when the specified fold is held out."""
    return np.concatenate([fold for j, fold in enumerate(folds) if j != index])


def table_ratio(study):
    """Refit the source-selected density ratios on the five training complements."""
    weights = np.empty(len(study.xp))
    for k, held in enumerate(study.folds):
        train = complement(study.folds, k)
        policy = dict(TABLE_RATIO_POLICY, widths=[(96, 96, 48, 96, 96)[k]])
        seed = PREDICTOR_SEED + 70000 + 10000 * k
        offsets = (0, 100000, 200000) if k < 2 else (0,)
        fits = [
            nu.fit_ratio(study.xp[train], study.xq, sim.CONFIG["neural"], seed + off, policy)
            for off in offsets
        ]

        def predict(x):
            if len(fits) == 1:
                return fits[0].predict(x)
            return np.mean([fit.predict(x) for fit in fits], axis=0)

        weights[held] = predict(study.xp[held])
        if k == 1:
            cap = 8 * len(train) ** sim.CONFIG["neural"]["cap_power"]
            # Solve the capped source-mean-one constraint on the training complement.
            raw = predict(study.xp[train])
            high = 1.0
            while np.minimum(cap, high * raw).mean() < 1.0:
                high *= 2.0
            scale = brentq(lambda s: np.minimum(cap, s * raw).mean() - 1.0, 0.0, high, xtol=1e-12)
            weights[held] = np.clip(scale * weights[held], 0.0, cap)
    return weights


def interval(estimate, lower, upper):
    if not np.isfinite([estimate, lower, upper]).all() or not lower <= estimate <= upper:
        raise ArithmeticError("Invalid interval")
    return dict(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        ci_length=float(upper - lower),
    )


def wald(estimate, p, q, jacobian):
    if not np.isfinite(jacobian) or jacobian <= 0:
        raise ArithmeticError("Invalid Wald slope")
    var = p.var(ddof=1) / len(p) + q.var(ddof=1) / len(q)
    half = norm.ppf(0.975) * np.sqrt(var) / jacobian
    return interval(estimate, estimate - half, estimate + half)


def scale_interval(p, q):
    """Invert EL for the squared conditional-mean scale, then take square roots."""
    value = q.mean() + (p.mean() if p is not None else 0.0)
    if value <= 0.0:
        raise ArithmeticError("Nonpositive squared scale")

    def stat(v):
        raw = inf.one_sample(q - v) if p is None else inf.two_sample(p, q - v)
        return inf.checked(raw) - inf.CUTOFF

    lo = 0.0 if stat(0.0) <= 0 else brentq(stat, 0.0, value, xtol=1e-13)
    hi = 2 * value
    while stat(hi) < 0:
        hi *= 2
    hi = brentq(stat, value, hi, xtol=1e-13)
    return interval(np.sqrt(value), np.sqrt(lo), np.sqrt(hi))


@njit
def median_profile(grid, source_cdf, target_cdf, ratio, error):
    stats = np.empty(len(grid))
    for j, threshold in enumerate(grid):
        q = target_cdf[:, j] - 0.5
        if len(ratio):
            p = ratio * ((error <= threshold) - source_cdf[:, j])
            val, status = inf.screen_el(p, q, 15.0)
        else:
            val, status = inf.screen_one(q, 15.0)
        if status == 2:
            raise ArithmeticError("EL failed to converge")
        stats[j] = np.inf if status == 1 else max(0.0, val)
    return stats


def median_interval(grid, target_cdf, source_cdf=None, ratio=None, error=None):
    if ratio is None:
        source_cdf, ratio, error = np.empty((0, 0)), np.empty(0), np.empty(0)
    stats = median_profile(grid, source_cdf, target_cdf, ratio, error)
    best = int(np.argmin(stats))
    inside = np.flatnonzero(stats <= stats[best] + inf.CUTOFF)
    if not np.isfinite(stats[best]) or inside[-1] + 1 >= len(grid):
        raise ArithmeticError("EL endpoint outside donor grid")
    return interval(grid[best], grid[inside[0]], grid[inside[-1] + 1])


def forest_weights(x, y, new, trees, leaf, seed):
    """Request the DRF conditional weights from the shared R implementation."""
    script = Path(__file__).resolve().parents[1] / "simulation" / "drf_interface.R"
    args = [
        os.environ.get("RSCRIPT", "Rscript"),
        "--vanilla",
        str(script),
        *map(str, [len(x), len(new), x.shape[1], seed, 3]),
        "weights",
        str(trees),
        str(leaf),
    ]
    data = b"".join(np.asarray(a, dtype="<f8").tobytes(order="F") for a in (x, y, new))
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    cmd = subprocess.run(
        args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=1800
    )
    if cmd.returncode:
        raise RuntimeError(cmd.stderr.decode(errors="replace"))
    w = np.frombuffer(cmd.stdout, dtype="<f8").reshape((len(new), len(x)), order="F")
    if np.min(w) < -1e-12 or not np.allclose(w.sum(1), 1.0, atol=1e-8, rtol=0):
        raise ArithmeticError("Invalid DRF weights")
    return w


def weighted_cdf(weights, y, grid):
    order = np.argsort(y)
    ys = y[order]
    bins = np.searchsorted(ys, grid, side="right") - 1
    cdf = np.cumsum(weights[:, order], axis=1)[:, np.maximum(bins, 0)]
    cdf[:, bins < 0] = 0.0
    return cdf


def pseudoinverse(a):
    u, s, vt = svd(a, full_matrices=False, check_finite=True, lapack_driver="gesdd")
    tol = max(a.shape) * np.finfo(float).eps * s[0]
    valid = s > tol
    return (vt[valid].T / s[valid]) @ u[:, valid].T, int(valid.sum())


def spline_design(x, xeval, knots):
    a = [np.ones((len(x), 1))]
    b = [np.ones((len(xeval), 1))]
    for j in range(x.shape[1]):
        breaks = np.unique(np.quantile(x[:, j], knots))
        if len(breaks) == 1:
            continue
        if len(breaks) == 2:
            a.append(x[:, j, None])
            b.append(xeval[:, j, None])
        else:
            trans = SplineTransformer(
                degree=3, knots=breaks[:, None], include_bias=False, extrapolation="linear"
            )
            a.append(trans.fit_transform(x[:, j, None]))
            b.append(trans.transform(xeval[:, j, None]))
    return np.column_stack(a), np.column_stack(b)


def cdf_basis(x, y, knots, ridge, poly_degree=None):
    if poly_degree is None:
        z, _ = spline_design(x, x[:1], knots)
        mapper = lambda xx: spline_design(x, xx, knots)[1]
    else:
        center = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1.0e-6)
        poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
        z = poly.fit_transform((x - center) / scale)
        mapper = lambda xx: poly.transform((xx - center) / scale)
    if ridge:
        D = np.eye(z.shape[1])
        D[np.std(z, axis=0) <= 1e-12, np.std(z, axis=0) <= 1e-12] = 0.0
        inverse = (pinvh(z.T @ z / len(z) + ridge * D) @ z.T) / len(z)
        rank = np.linalg.matrix_rank(z)
    else:
        inverse, rank = pseudoinverse(z)
    i = np.argsort(y)
    prefix = np.column_stack([np.zeros(z.shape[1]), np.cumsum(inverse[:, i], axis=1)])
    return dict(z=z, inverse=inverse, mapper=mapper, prefix=prefix, sorted_y=y[i], rank=int(rank))


def evaluate_basis(fit, x, grid, clip=False):
    ix = np.searchsorted(fit["sorted_y"], grid, side="right")
    F = fit["mapper"](x) @ fit["prefix"][:, ix]
    return np.clip(F, 0.0, 1.0) if clip else F


def curve_projection(curves, grid):
    weights = np.diff(grid)
    out = np.empty_like(curves)
    for i, row in enumerate(curves):
        out[i, :-1] = isotonic_regression(
            row[:-1], sample_weight=weights, y_min=0, y_max=1, increasing=True
        )
        out[i, -1] = 1.0
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
    assert np.min(np.diff(out, axis=1)) >= -1e-12
    return out


def train_residual_network(a, b, valid, width, cap, seed, cfg, policy):
    torch = nu.torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1e-6)
    aa, bb, va, vb = [
        torch.tensor((v - center) / scale, dtype=torch.float32) for v in (a, b, *valid)
    ]
    net = nu.Network(a.shape[1], dict(hidden=[width, width], cap=cap), "ratio")
    nu.covariance_initialization(net, a, b)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    with torch.no_grad():
        best = nu.validation_loss(net, *valid, center, scale, va, vb, True)
    state = nu.copy.deepcopy(net.state_dict())
    best_step = 0
    stale = 0
    for step in range(1, 3001):
        ids = rng.integers(len(a), size=cfg["batch_size"])
        iy = rng.integers(len(a), size=cfg["batch_size"])
        negative = aa[ids].clone()
        negative[:, -1] = aa[iy, -1]
        opt.zero_grad()
        loss = net(negative).square().mean() - 2 * net(bb[ids]).mean()
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
        opt.step()
        if step % cfg["check_every"] == 0:
            with torch.no_grad():
                current = nu.validation_loss(net, *valid, center, scale, va, vb, True)
            if current < best - 1e-7:
                best = current
                best_step = step
                stale = 0
                state = nu.copy.deepcopy(net.state_dict())
            else:
                stale += 1
            if step >= 200 and stale >= 25:
                break
    net.load_state_dict(state)
    net.eval().double()
    return nu.RatioFit(net, center, scale, 0.0, 1.0, {}), dict(
        loss=best, width=width, cap=cap, updates=best_step
    )


def refit_residual_network(fit, a, b, selected, seed, cfg, opts):
    torch = nu.torch
    rng = np.random.default_rng(seed)
    net = fit.model.float()
    aa, bb = [torch.tensor((v - fit.center) / fit.scale, dtype=torch.float32) for v in (a, b)]

    def risk():
        return nu.validation_loss(net, a, b, fit.center, fit.scale, aa, bb, True)

    with torch.no_grad():
        best = float(risk())
    state = nu.copy.deepcopy(net.state_dict())
    updates = max(20, min(100, int(np.ceil(0.25 * selected["updates"]))))
    opt = torch.optim.Adam(net.parameters(), lr=0.00025)
    for step in range(updates):
        ids = rng.integers(len(a), size=cfg["batch_size"])
        iy = rng.integers(len(a), size=cfg["batch_size"])
        neg = aa[ids].clone()
        neg[:, -1] = aa[iy, -1]
        opt.zero_grad()
        loss = net(neg).square().mean() - 2 * net(bb[ids]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
        opt.step()
        if (step + 1) % 20 == 0 or step + 1 == updates:
            with torch.no_grad():
                cur = float(risk())
            if cur < best:
                best = cur
                state = nu.copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.eval().double()
    return fit, dict(updates=updates, final_loss=best)


class ResponseRegressor:
    def __init__(self, net, center, scale, loc, spread):
        self.net = net
        self.center = center
        self.scale = scale
        self.loc = loc
        self.spread = spread

    def predict(self, x):
        with nu.torch.no_grad():
            z = nu.torch.tensor((x - self.center) / self.scale, dtype=nu.torch.float32)
            value = self.net(z).numpy().astype(float) * self.spread + self.loc
        return value


def fit_regression(x, y, seed, width):
    torch = nu.torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    loc = float(y.mean())
    spread = max(float(y.std()), 1e-6)
    z = torch.tensor((x - center) / scale, dtype=torch.float32)
    yy = torch.tensor((y - loc) / spread, dtype=torch.float32)
    model = nu.Network(x.shape[1], dict(hidden=[width, width], cap=1.0), "regression")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    for step in range(4000):
        ids = rng.integers(len(x), size=512)
        optimizer.zero_grad()
        loss = (model(z[ids]) - yy[ids]).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return ResponseRegressor(model, center, scale, loc, spread)


def residual_fit(x, z, width, cap, seed):
    policy = dict(
        sim.BASE_RHO,
        width_by_n={str(len(z)): width},
        cap_multiplier=cap,
        initialization="covariance",
    )
    fits, _ = nu.fit_dependence(
        nu.Observed(x, z, x[:1]),
        policy,
        sim.CONFIG["neural"],
        seed,
        trainer=train_residual_network,
        refitter=refit_residual_network,
    )
    return fits


def oel(study):
    """Use both signed-residual networks and average the target outer map across source folds."""
    y, z = study.error, study.residual
    n, m = len(y), len(study.xq)
    ratio = table_ratio(study)
    grid = np.r_[0.0, np.unique(y)]
    ps = np.empty((n, len(grid)), order="F")
    qt = np.zeros((m, len(grid)), order="F")
    ap = np.empty(n)
    aq2 = np.zeros(m)
    for k, held in enumerate(study.folds):
        tr = complement(study.folds, k)
        seed = SEED + 5000 + 10000 * k
        both = [
            residual_fit(study.xp[tr], z[tr], width, cap, seed)
            for width, cap in [(192, 8.0), (96, 3.0)]
        ]
        donor = tr[np.argsort(y[tr])]
        last = np.searchsorted(y[donor], grid, side="right")
        x = np.r_[study.xp[held], study.xq]
        for start in range(0, len(x), 64):
            ids = np.arange(start, min(start + 64, len(x)))
            wide, narrow = [nu.dependence_weights(fit, x[ids], z[donor])[0] for fit in both]
            weight = np.maximum(0.5 * (wide + narrow), 0.0)
            mass = weight.sum(1)
            good = mass > 0
            weight[good] /= mass[good, None]
            weight[~good] = 1.0 / len(donor)
            cdf = np.c_[np.zeros(len(ids)), np.cumsum(weight, axis=1)][:, last]
            mean = weight @ y[donor]
            inside = ids < len(held)
            ps[held[ids[inside]]] = cdf[inside]
            ap[held[ids[inside]]] = mean[inside]
            qidx = ids[~inside] - len(held)
            fraction = len(held) / n
            qt[qidx] += fraction * cdf[~inside]
            aq2[qidx] += fraction * mean[~inside] ** 2
    order = np.argsort(y)
    med = median_interval(grid, qt, np.asfortranarray(ps[order]), ratio[order], y[order])
    sp = 2 * ratio * ap * (y - ap)
    scale = scale_interval(sp, aq2)
    j = np.searchsorted(grid, med["estimate"])
    scores = [
        (ratio * ((y <= grid[j]) - ps[:, j]), qt[:, j] - 0.5, inf.density(y, ratio, grid[j])),
        (sp, aq2 - scale["estimate"] ** 2, 2 * scale["estimate"]),
    ]
    return [med, scale], scores


def plugins(study):
    """Estimate both targets with the spline and forest CDFs and fit the supplementary projected spline."""
    y = study.error
    grid = np.r_[0.0, np.unique(y)]
    fit = cdf_basis(study.xp, y, sim.KNOTS, 0.001)
    cdf = np.asfortranarray(evaluate_basis(fit, study.xq, grid))
    a = (1 - cdf[:, :-1]) @ np.diff(grid)
    sieve = [median_interval(grid, cdf), scale_interval(None, a * a)]
    fit = cdf_basis(study.xp, y, np.linspace(0, 1, 9), 0.0)
    cdf = np.asfortranarray(curve_projection(evaluate_basis(fit, study.xq, grid), grid))
    a = (1 - cdf[:, :-1]) @ np.diff(grid)
    projected = [median_interval(grid, cdf), scale_interval(None, a * a)]
    weights = forest_weights(study.xp, y, study.xq, 1500, 5, SEED + 300)
    cdf = np.asfortranarray(weighted_cdf(weights, y, grid))
    a = weights @ y
    forest = [median_interval(grid, cdf), scale_interval(None, a * a)]
    return {"Plug-Sieve": sieve, "Plug-DRF": forest}, projected


def drw(study):
    """Use a weighted source median and a weighted quadratic-sieve scale, with Wald intervals."""
    x, y, xq = study.xp, study.error, study.xq
    rng = np.random.default_rng(SEED + 14000)
    ip, iq = rng.permutation(len(x)), rng.permutation(len(xq))
    p = len(x) // 5
    q = len(xq) // 5
    paths = []
    for width in (48, 96):
        policy = dict(sim.BASE_R, widths=[width], cap_multiplier=8.0, maximum_updates=3000)
        fit, record = nu.train_checkpoint(
            x[ip[p:]],
            xq[iq[q:]],
            (x[ip[:p]], xq[iq[:q]]),
            width,
            8 * len(x) ** 0.1,
            SEED + 14000 + width,
            sim.CONFIG["neural"],
            policy,
        )
        paths.append((fit, record))
    fit, record = min(paths, key=lambda v: (v[1]["loss"], v[1]["width"]))
    options = dict(
        refit_fraction=0.25,
        refit_min=20,
        refit_max=100,
        refit_learning_rate=0.00025,
        paired_batch=False,
        refit_full_batch=True,
        restore_lowest_empirical_loss=True,
    )
    fit, _ = nu.refit(fit, x, xq, record, SEED + 14100, sim.CONFIG["neural"], options)
    w = fit.predict(x)
    w = w / w.mean()
    order = np.argsort(y)
    point = y[order][np.searchsorted(np.cumsum(w[order]) / len(y), 0.5)]
    basis = cdf_basis(x, y, None, 0.0, 2)
    fp = evaluate_basis(basis, x, np.array([point]), True).ravel()
    fq = evaluate_basis(basis, xq, np.array([point]), True).ravel()
    med = wald(point, w * ((y <= point) - fp), fq - 0.5, inf.density(y, w, point))
    coef = basis["inverse"] @ y
    ap, aq = [np.clip(z @ coef, 0.0, y.max()) for z in (basis["z"], basis["mapper"](xq))]
    scale = np.sqrt(np.mean(w * ap**2))
    return [med, wald(scale, 2 * w * ap * (y - ap), aq**2, 2 * scale)]


def ldml(study):
    """Use three folds for the localized median and for the debiased conditional-mean scale."""
    x, y, xq = study.xp, study.error, study.xq
    px = np.array_split(np.random.default_rng(PREDICTOR_SEED + 5000).permutation(len(y)), 3)
    qx = np.array_split(np.random.default_rng(PREDICTOR_SEED + 5001).permutation(len(xq)), 3)
    ratio, fp, ap = [np.empty(len(y)) for _ in range(3)]
    fq, aq = [np.empty(len(xq)) for _ in range(2)]
    for k, (held, target) in enumerate(zip(px, qx)):
        p, r = px[(k + 1) % 3], px[(k + 2) % 3]
        qp, qr = qx[(k + 1) % 3], qx[(k + 2) % 3]
        seed = SEED + 5000 + 10000 * k
        policy = dict(sim.BASE_R, widths=[96], cap_multiplier=8.0, maximum_updates=3000)
        fit = nu.fit_ratio(
            x[np.r_[p, r]], xq[np.r_[qp, qr]], sim.CONFIG["neural"], seed + 1, policy
        )
        ratio[held] = fit.predict(x[held])
        ordered = p[np.argsort(y[p])]
        wp = fit.predict(x[ordered])
        pilot = y[ordered][np.searchsorted(np.cumsum(wp) / wp.sum(), 0.5)]
        reg = fit_regression(x[r], (y[r] <= pilot).astype(float), seed + 301, 32)
        fp[held] = np.clip(reg.predict(x[held]), 0.0, 1.0)
        fq[target] = np.clip(reg.predict(xq[target]), 0.0, 1.0)
        train = complement(px, k)
        reg = fit_regression(x[train], y[train], SEED + 55700 + k, 64)
        ap[held] = np.clip(reg.predict(x[held]), 0.0, y[train].max())
        aq[target] = np.clip(reg.predict(xq[target]), 0.0, y[train].max())
    order = np.argsort(y)
    moment = np.cumsum(ratio[order]) / len(y) + (fq - 0.5).mean() - np.mean(ratio * fp)
    point = y[order][np.flatnonzero(moment >= 0)[0]]
    med = wald(point, ratio * ((y <= point) - fp), fq - 0.5, inf.density(y, ratio, point))
    p = 2 * ratio * ap * (y - ap)
    q = aq**2
    scale = np.sqrt(p.mean() + q.mean())
    return [med, wald(scale, p, q, 2 * scale)]


# Target-only references and spatial sensitivity

def extra_tree_weights(x, y, out, leaf, seed, trees=500):
    model = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=leaf, max_features=1.0, random_state=seed, n_jobs=1
    )
    model.fit(x, y)
    weights = np.zeros((len(out), len(x)))
    for tree in model.estimators_:
        train_nodes = tree.apply(x)
        out_nodes = tree.apply(out)
        order = np.argsort(train_nodes)
        nodes = train_nodes[order]
        lo = np.searchsorted(nodes, out_nodes, side="left")
        hi = np.searchsorted(nodes, out_nodes, side="right")
        for k, (a, b) in enumerate(zip(lo, hi)):
            weights[k, order[a:b]] += 1.0 / ((b - a) * len(model.estimators_))
    assert np.max(abs(weights.sum(axis=1) - 1)) < 1e-8
    return weights


def brier(pred, train_y, valid_y, knots, weights):
    obs = (valid_y[:, None] <= knots[None, :]).astype(float)
    rate = (train_y[:, None] <= knots[None, :]).mean(axis=0)
    row = np.mean((pred - obs) ** 2 / np.maximum(rate * (1 - rate), 0.01), axis=1)
    return float(np.average(row, weights=weights))


def references(study, target_error):
    """Only this fit and the final curve comparison receive target outcomes."""
    x, y = study.xq, target_error
    n = len(y)
    lo = int(binom.ppf(0.025, n, 0.5)) - 1
    hi = int(binom.isf(0.025, n, 0.5))
    med = interval(np.median(y), np.sort(y)[lo], np.sort(y)[hi])
    parts = np.array_split(np.random.default_rng(SEED + 9000).permutation(n), 3)
    means = {key: np.empty(n) for key in ("selected", "DRF", "ExtraTrees", "Ensemble")}
    for k, held in enumerate(parts):
        train = complement(parts, k)
        p = np.random.default_rng(SEED + 19000 + 10000 * k).permutation(train)
        v, fit = p[: len(p) // 5], p[len(p) // 5 :]
        cuts = np.quantile(y[fit], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        validation = {}
        seed = SEED + 29000 + k
        for name, leaves in (("DRF", (5, 10, 25)), ("ExtraTrees", (5, 10, 20))):

            def forecast(rows, new, leaf, final):
                if name == "DRF":
                    return forest_weights(
                        x[rows], y[rows], x[new], 1500 if final else 500, leaf, seed
                    )
                return extra_tree_weights(x[rows], y[rows], x[new], leaf, seed)

            candidates = {
                leaf: weighted_cdf(forecast(fit, v, leaf, False), y[fit], cuts) for leaf in leaves
            }
            best = min(
                candidates, key=lambda leaf: brier(candidates[leaf], y[fit], y[v], cuts, None)
            )
            validation[name] = candidates[best]
            means[name][held] = forecast(train, held, best, True) @ y[train]
        validation["Ensemble"] = 0.5 * (validation["DRF"] + validation["ExtraTrees"])
        choice = min(validation, key=lambda name: brier(validation[name], y[fit], y[v], cuts, None))
        means["Ensemble"][held] = 0.5 * (means["DRF"][held] + means["ExtraTrees"][held])
        means["selected"][held] = means[choice][held]
    scales = {key: scale_interval(None, 2 * pred * y - pred**2) for key, pred in means.items()}
    ref = scales["selected"]
    pred = means["selected"]
    scores = [
        (None, (y <= med["estimate"]).astype(float) - 0.5, inf.density(y, np.ones(n), med["estimate"])),
        (None, 2 * pred * y - pred**2 - ref["estimate"] ** 2, 2 * ref["estimate"]),
    ]
    return [med, ref], scales, scores


def spatial_scores(study, components):
    """Linearized SEs under independence and one- and two-degree sky aggregation."""
    out = []
    for p, q, jac in components:
        vq = q.var(ddof=1) / len(q)
        vp = 0.0 if p is None else p.var(ddof=1) / len(p)
        sky = study.sky_q if p is None else np.vstack([study.sky_p, study.sky_q])
        value = (q - q.mean()) / len(q)
        if p is not None:
            value = np.r_[(p - p.mean()) / len(p), value]
        row = dict(
            independent=np.sqrt(vp + vq) / jac,
            target_share=vq / (vp + vq),
            top_five=np.sort((q - q.mean()) ** 2)[-5:].sum() / np.sum((q - q.mean()) ** 2),
        )
        for degrees in (1, 2):
            cells, idx = np.unique(np.floor(sky / degrees).astype(int), axis=0, return_inverse=True)
            total = np.bincount(idx, weights=value)
            row[f"sky_{degrees}"] = np.sqrt(len(cells) / (len(cells) - 1) * (total @ total)) / jac
        out.append(row)
    return out
