"""Joint GAMA error quantiles, from nuisance estimation to simultaneous EL inference."""

import hashlib
from functools import lru_cache, partial

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from simulation import inference as inf, nuisance as nu, simulate as sim
from simulation.inference import screen_el
from simulation.simulate import gaussian_cutoff

from .pointwise_inference import TABLE_RATIO_POLICY, complement

CURVE_SEED = 2026112607
MODEL_SEED = 2026112901
LEVELS = np.arange(10, 91, 5) / 100.0
# Source-validation choices in fold order: (relative-spread weight, leaf-density weight),
# followed by the two signed-residual scales. These are refitted, never loaded as predictions.
FOLD_MODELS = ((0, 1), (1, 0), (0.5, 0.5), (0, 1), (0.5, 0.5))
FOLD_SCALES = ((0.9, 1), (1, 0.9), (1, 0.9), (0.9, 1), (1, 0.9))


# Simultaneous empirical likelihood

@njit
def donor_indices(t, hh, ys, mid):
    n = len(hh)
    ny = len(ys)
    left = np.empty(n, np.int64)
    right = np.empty(n, np.int64)
    for i in range(n):
        h = hh[i]
        a = 0
        b = mid[i]
        while a < b:
            j = (a + b) // 2
            if (h - ys[j]) / (1.0 + ys[j]) > t:
                a = j + 1
            else:
                b = j
        left[i] = a
        a = mid[i]
        b = ny
        while a < b:
            j = (a + b) // 2
            if (ys[j] - h) / (1.0 + ys[j]) <= t:
                a = j + 1
            else:
                b = j
        right[i] = a
    return left, right


@njit
def cdf_at(t, h, ys, mid, raw):
    l, r = donor_indices(t, h, ys, mid)
    out = np.empty(len(h))
    for i in range(len(h)):
        out[i] = raw[i, r[i]] - raw[i, l[i]]
    return out


@njit
def error_atoms(a, b, hh, ys, mid, raw, pt, rp):
    la, ra = donor_indices(a, hh, ys, mid)
    lb, rb = donor_indices(b, hh, ys, mid)
    sz = np.sum(la - lb + rb - ra) + len(pt)
    out = np.empty(sz)
    w = 0
    for i in range(len(hh)):
        if i < len(rp) and rp[i] == 0:
            continue
        for j in range(lb[i], la[i]):
            if raw[i, j + 1] > raw[i, j]:
                t = (hh[i] - ys[j]) / (1.0 + ys[j])
                if a < t <= b:
                    out[w] = t
                    w += 1
        for j in range(ra[i], rb[i]):
            if raw[i, j + 1] > raw[i, j]:
                t = (ys[j] - hh[i]) / (1.0 + ys[j])
                if a < t <= b:
                    out[w] = t
                    w += 1
    for i in range(len(pt)):
        if rp[i] != 0.0 and a < pt[i] <= b:
            out[w] = pt[i]
            w += 1
    return np.unique(out[:w])


def estimate_band(study):
    """Fit with source redshifts and target covariates. Target redshifts enter only in evaluate()."""
    ratio, source_error, redshifts, conditional_cdf = conditional_redshifts(study)
    predictions = np.r_[study.hp, study.hq]
    n, m = len(study.hp), len(study.hq)
    split = np.searchsorted(redshifts, predictions)
    maximum = float(
        max(
            source_error.max(),
            np.max(abs(predictions - redshifts[0]) / (1.0 + redshifts[0])),
            np.max(abs(predictions - redshifts[-1]) / (1.0 + redshifts[-1])),
        )
    )
    anchors = np.unique(
        np.r_[
            0.0,
            np.quantile(source_error, np.linspace(0, 1, 769)),
            maximum,
            np.nextafter(maximum, np.inf),
        ]
    )
    anchor_cdf = np.stack(
        [cdf_at(t, predictions, redshifts, split, conditional_cdf) for t in anchors]
    )
    anchor_moments = np.array(
        [
            [ratio @ (source_error <= t) / n, ratio @ v[:n] / n, v[n:].mean()]
            for t, v in zip(anchors, anchor_cdf)
        ]
    )
    moments = anchor_moments[:, 0] - anchor_moments[:, 1] + anchor_moments[:, 2]

    @lru_cache(maxsize=80)
    def cdf(t):
        return cdf_at(float(t), predictions, redshifts, split, conditional_cdf)

    def el(t, v, tau):
        p = ratio * ((source_error <= t) - v[:n])
        q = v[n:] - tau
        s, rc = inf.two_sample(p, q)
        if rc == 1:
            return np.inf
        if rc != 0 or not np.isfinite(s):
            raise ArithmeticError(("actual EL", t, tau, s, rc))
        return float(max(0.0, s))

    def screen(p, q, tau, cap):
        s, rc = screen_el(p, q - tau, cap)
        if rc == 1:
            return np.inf
        if rc not in (0, 3) or not np.isfinite(s):
            raise ArithmeticError(("guard EL", tau, cap, s, rc))
        return float(max(0.0, s))

    def moment_parts(t, v):
        return float(ratio @ (source_error <= t) / n), float(ratio @ v[:n] / n), float(v[n:].mean())

    def moment(t, v):
        a, b, c = moment_parts(t, v)
        return a - b + c

    def atoms(a, b):
        return error_atoms(
            float(a), float(b), predictions, redshifts, split, conditional_cdf, source_error, ratio
        )

    def score_envelopes(a, b, fa, fb):
        return (
            ratio * ((source_error <= a) - fb[:n]),
            ratio * ((source_error <= b) - fa[:n]),
            fa[n:],
            fb[n:],
        )

    def minimum_at(index):
        tau = LEVELS[index]
        target = int(np.argmin(abs(moments - tau)))
        t0 = float(anchors[target])
        best = el(t0, anchor_cdf[target], tau)
        est = t0

        def consider(t, v):
            nonlocal best, est
            z = el(t, v, tau)
            if z < best or (z == best and t < est):
                best = z
                est = float(t)

        # Bracket all initial crossings to obtain a tight upper bound on the minimum.
        for j in np.flatnonzero((moments[:-1] - tau) * (moments[1:] - tau) <= 0):
            lo, hi = float(anchors[j]), float(anchors[j + 1])
            sign = (moments[j] - tau) >= 0
            for _ in range(42):
                mid = (lo + hi) / 2.0
                if mid in (lo, hi):
                    break
                f = cdf(mid)
                moment_value = moment(mid, f)
                if (moment_value >= tau) == sign:
                    lo = mid
                else:
                    hi = mid
            consider(lo, cdf(lo))
            consider(hi, cdf(hi))

        def visit(a, b, fa, fb, ag1, ag2):
            lb = ag1[0] - ag2[1] + ag1[2] - tau
            ub = ag2[0] - ag1[1] + ag2[2] - tau
            # For any t in [a,b], signed sqrt(EL) is bounded by the
            # componentwise endpoint score envelopes. The EL weight-budget
            # feasible set is the same for every t.
            if lb > 0 or ub < 0:
                gap = lb if lb > 0 else -ub
                if 4 * max(0.0, gap - 1e-13) ** 2 > (best + 2e-10) * (
                    (2 * ratio.max()) ** 2 / n + 1 / m
                ):
                    return
                if lb > 0:
                    p = ratio * ((source_error <= a) - fb[:n])
                    q = fa[n:]
                else:
                    p = ratio * ((source_error <= b) - fa[:n])
                    q = fb[n:]
                rr = screen(p, q, tau, best + 1e-8)
                if rr > best + 5e-9:
                    return
            if b - a <= 3.0e-8 or np.nextafter(a, np.inf) >= b:
                ee = atoms(a, b)
                consider(a, fa)
                for e in ee:
                    consider(float(e), cdf(float(e)))
                return
            mid = 0.5 * (a + b)
            fm = cdf(mid)
            ag = moment_parts(mid, fm)
            consider(mid, fm)
            visit(a, mid, fa, fm, ag1, ag)
            visit(mid, b, fm, fb, ag, ag2)

        for j in range(len(anchors) - 1):
            visit(
                float(anchors[j]),
                float(anchors[j + 1]),
                anchor_cdf[j],
                anchor_cdf[j + 1],
                anchor_moments[j],
                anchor_moments[j + 1],
            )
        return est, best

    estimates, minima = np.array([minimum_at(j) for j in range(len(LEVELS))]).T
    fitted = np.column_stack([cdf(t) for t in estimates])
    p = ratio[:, None] * ((source_error[:, None] <= estimates) - fitted[:n])
    q = fitted[n:] - LEVELS
    normals = np.random.default_rng(2026112501).standard_normal((20000, len(LEVELS)))
    cutoff = float(gaussian_cutoff(p, q, normals))

    def side(p, q, tau, limit):
        m = p.mean() + q.mean() - tau
        s = screen(p, q, tau, limit + 1e-7)
        if s <= limit:
            return 0
        return 1 if m > 0 else -1

    def invert_at(index):
        tau = LEVELS[index]
        limit = cutoff + minima[index]
        ranges = []

        def append(a, b):
            if ranges and a == ranges[-1][1]:
                ranges[-1][1] = b
            else:
                ranges.append([a, b])

        def visit(a, b, fa, fb, ag1, ag2):
            if a >= b:
                return
            # Test the side with the larger separation first.
            p0, p1, q0, q1 = score_envelopes(a, b, fa, fb)
            if 0.5 * ((ag1[0] + ag2[0]) - (ag1[1] + ag2[1]) + (ag1[2] + ag2[2])) >= tau:
                low = side(p0, q0, tau, limit)
                if low > 0:
                    return
                high = side(p1, q1, tau, limit)
            else:
                high = side(p1, q1, tau, limit)
                if high < 0:
                    return
                low = side(p0, q0, tau, limit)
            assert low <= high, (index, a, b, low, high)
            if high < 0 or low > 0:
                return
            if low == 0 and high == 0:
                append(a, b)
                return
            if b - a <= 3.0e-8 or np.nextafter(a, np.inf) >= b:
                ev = atoms(a, b)
                ed = np.r_[a, ev[(ev > a) & (ev < b)], b]
                for j in range(len(ed) - 1):
                    t = float(ed[j])
                    f = fa if j == 0 else cdf(t)
                    if el(t, f, tau) <= limit:
                        append(t, float(ed[j + 1]))
                return
            mid = 0.5 * (a + b)
            fm = cdf(mid)
            ag = moment_parts(mid, fm)
            visit(a, mid, fa, fm, ag1, ag)
            visit(mid, b, fm, fb, ag, ag2)

        for j in range(len(anchors) - 1):
            visit(
                float(anchors[j]),
                float(anchors[j + 1]),
                anchor_cdf[j],
                anchor_cdf[j + 1],
                anchor_moments[j],
                anchor_moments[j + 1],
            )
        assert len(ranges) > 0 and ranges[0][0] > 0 and ranges[-1][1] < maximum
        assert any(lo <= estimates[index] < hi for lo, hi in ranges)
        for l, h in ranges:
            for t, want in [
                (np.nextafter(l, -np.inf), False),
                (l, True),
                (np.nextafter(h, -np.inf), True),
                (h, False),
            ]:
                z = el(float(t), cdf(float(t)), tau)
                assert (z <= limit) == want, (index, t, want, z, limit)
        return ranges

    bounds = [invert_at(i) for i in range(len(LEVELS))]
    pieces = [estimates.astype("<f8").tobytes(), np.array([cutoff], dtype="<f8").tobytes()]
    for bs in bounds:
        pieces.extend(
            [np.array([len(bs)], dtype="<i8").tobytes(), np.array(bs, dtype="<f8").tobytes()]
        )
    source = np.sort(source_error)[(len(source_error) * np.arange(10, 91, 5) + 99) // 100 - 1]
    rows = [
        dict(q=t, e=e, src=z, bounds=b) for t, e, z, b in zip(LEVELS, estimates, source, bounds)
    ]
    return dict(digest=hashlib.sha256(b"".join(pieces)).hexdigest(), cutoff=cutoff, rows=rows)


def evaluate(result, target_error):
    """Attach the held-out empirical reference only after the joint set is fixed."""
    ordered = np.sort(target_error)
    spans = []
    for row in result["rows"]:
        percent = int(round(100 * row["q"]))
        reference = float(ordered[(len(ordered) * percent + 99) // 100 - 1])
        intervals = row["bounds"]
        row["ref"] = reference
        row["hit"] = any(lo <= reference < hi for lo, hi in intervals)
        # For a disconnected set, use the nearer endpoint of the best component.
        row["margin"] = max(min(reference - lo, hi - reference) / (hi - lo) for lo, hi in intervals)
        spans.append(intervals[-1][1] - intervals[0][0])
    result["comparison"] = dict(
        levels=len(spans),
        inside=sum(row["hit"] for row in result["rows"]),
        inside_by_tenth=int(sum(row["hit"] and row["margin"] >= 0.1 for row in result["rows"])),
        mean_outer_width=float(np.mean(spans)),
    )
    return result


@njit
def grid_profiles(cdf, r, last, taus, cut):
    n = len(r)
    m = cdf.shape[0] - n
    stats = np.full((len(taus), cdf.shape[1]), np.inf)
    status = np.full((len(taus), cdf.shape[1]), 3, np.int8)
    p = np.empty(n)
    for col in range(cdf.shape[1]):
        for i in range(n):
            p[i] = r[i] * ((1.0 if i < last[col] else 0.0) - cdf[i, col])
        q = cdf[n:, col]
        total = p.mean() + q.mean()
        denom = (p.max() - p.min()) ** 2 / n + (q.max() - q.min()) ** 2 / m
        for j in range(len(taus)):
            # A Pinsker bound certifies pruning outside the entire raw EL range of interest.
            d = max(abs(total - taus[j]) - 1e-11, 0.0)
            if 4 * d * d > (cut + 1.0e-6) * denom:
                continue
            value, code = screen_el(p, q - taus[j], cut)
            status[j, col] = code
            if code == 0:
                stats[j, col] = max(0.0, value)
    return stats, status


def full_source_band(study):
    """Supplementary sensitivity analysis with both nuisances and EL fitted on the full source sample."""
    x, xq, z, error = study.xp, study.xq, study.residual, study.error
    n = len(x)
    seed = 771103
    cfg = sim.CONFIG["neural"]
    r = nu.fit_ratio(
        x,
        xq,
        cfg,
        seed + 500,
        dict(TABLE_RATIO_POLICY, cap_multiplier=16),
        trainer=partial(nu.train_checkpoint, initializer=initialize_tree_ratio),
    )
    policy = dict(
        sim.BASE_RHO,
        width_by_n={str(n): 96},
        cap_multiplier=3.0,
        initialization="covariance",
        learning_rate=0.001,
        learning_schedule="constant",
        maximum_updates=3000,
        minimum_updates=200,
        patience=25,
    )
    fits, _ = nu.fit_dependence(
        nu.Observed(x, z, x[:1]),
        policy,
        cfg,
        seed + 517,
        trainer=partial(nu.train_path, initializer=initialize_hinge_ratio),
    )
    order = np.argsort(error)
    y = error[order]
    ratio = r.predict(x)[order]
    grid = np.r_[0.0, np.unique(y)]
    last = np.searchsorted(y, grid, side="right")
    design = np.vstack([x[order], xq])
    cdf = np.empty((len(design), len(grid)), order="F")
    for start in range(0, len(design), 64):
        stop = min(start + 64, len(design))
        weights, _ = nu.dependence_weights(fits, design[start:stop], z[order])
        weights = np.maximum(weights, 0.0)
        mass = weights.sum(1)
        good = mass > 0.0
        weights[good] /= mass[good, None]
        weights[~good] = 1.0 / n
        cdf[start:stop] = np.c_[np.zeros(stop - start), np.cumsum(weights, axis=1)][:, last]
    screen_cut = 15.0
    values, status = grid_profiles(cdf, ratio, last, LEVELS, screen_cut)
    if np.any(status == 2):
        raise ArithmeticError("Full-source EL profile failed")
    best = values.argmin(axis=1)
    minima = values[np.arange(len(LEVELS)), best]
    estimates = grid[best]
    fitted = cdf[:, best]
    p = ratio[:, None] * ((y[:, None] <= estimates) - fitted[:n])
    q = fitted[n:] - LEVELS
    normal = np.random.default_rng(2026112501).standard_normal((20000, len(LEVELS)))
    cutoff = float(gaussian_cutoff(p, q, normal))
    if cutoff + max(minima) >= screen_cut:
        raise ArithmeticError("EL screening cutoff too small")
    source = y[(n * np.arange(10, 91, 5) + 99) // 100 - 1]
    rows = []
    for j, level in enumerate(LEVELS):
        keep = (status[j] == 0) & (values[j] - minima[j] <= cutoff)
        first = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
        end = np.flatnonzero(keep & ~np.r_[keep[1:], False])
        if not len(first) or first[0] == 0 or end[-1] + 1 >= len(grid):
            raise ArithmeticError("EL boundary outside donor grid")
        bounds = [[float(grid[lo]), float(grid[hi + 1])] for lo, hi in zip(first, end)]
        rows.append(dict(q=level, e=estimates[j], src=source[j], bounds=bounds))
    return dict(cutoff=cutoff, rows=rows)


# Conditional redshift distribution

def redshift_brackets(H, t, grid):
    t = np.asarray(t, dtype=float)
    l = H[:, None] - np.log1p(t)[None, :]
    top = np.full(len(t), np.inf)
    m = t < 1.0
    top[m] = -np.log1p(-t[m])
    hi = H[:, None] + top[None, :]
    return np.searchsorted(grid, l, "left"), np.searchsorted(grid, hi, "right")


def split_pairs(inds, rng):
    v = rng.permutation(inds)
    s = len(v) // 2
    l, r = v[:s], v[s : 2 * s]
    return [(l, rng.permutation(r)), (r, rng.permutation(l))]


def residual_scale(disp, base):
    return np.maximum(0.15 * base, 0.5 * (disp + base))


@njit
def donor_predictions(ys, wu, wd, offset, h, ss, w2, b2, w3, b3, cap):
    result = np.empty((len(h), len(ys)))
    for i in range(len(h)):
        slope = wu + wd / ss[i]
        bias = offset[i] - wd * (h[i] / ss[i])
        result[i], _ = nu.row_predictions(ys, slope, bias, w2, b2, w3, b3, cap)
    return result


def signed_weights(models, x, H, disp, ys, slope):
    raw = np.zeros((len(x), len(ys)))
    for alpha, mod in models:
        # Multiplying the signed outcome coordinate changes only the conditional
        # redshift ratio. The observed errors, donor redshifts and h remain fixed.
        ss = residual_scale(disp, mod["base"]) / slope
        for fit in mod["fits"]:
            net = fit.model.layers

            def par(t):
                return np.ascontiguousarray(t.detach().numpy(), dtype=float)

            w1, b1 = par(net[0].weight), par(net[0].bias)
            w2, b2 = par(net[2].weight), par(net[2].bias)
            w3 = par(net[4].weight).ravel()
            b3 = float(net[4].bias.detach().numpy()[0])
            wu = w1[:, -2] / fit.scale[-2]
            wd = w1[:, -1] / fit.scale[-1]
            off = (
                ((x - fit.center[:-2]) / fit.scale[:-2]) @ w1[:, :-2].T
                + b1
                - wu * fit.center[-2]
                - wd * fit.center[-1]
            )
            a = donor_predictions(
                ys,
                np.ascontiguousarray(wu),
                np.ascontiguousarray(wd),
                np.ascontiguousarray(off),
                np.ascontiguousarray(H),
                np.ascontiguousarray(ss),
                w2,
                b2,
                w3,
                b3,
                float(fit.model.cap),
            )
            for i in np.flatnonzero(a.sum(1) <= 1e-8):
                v = np.c_[np.repeat(x[i : i + 1], len(ys), axis=0), ys, (ys - H[i]) / ss[i]]
                a[i] = fit.predict(v)
            raw += 0.5 * alpha * a
    assert np.min(raw) > -1e-8 and np.isfinite(raw).all()
    return np.maximum(raw, 0.0)


def conditional_redshifts(study):
    """Return out-of-fold source ratios and conditional CDFs on all distinct donor redshifts."""
    ratio, spread_p, spread_q = curve_ratio(study)
    source_z = study.redshift
    log_z = np.log1p(source_z)
    log_prediction_p, log_prediction_q = np.log1p(study.hp), np.log1p(study.hq)
    observed_error = np.abs(study.hp - source_z) / (1.0 + source_z)
    log_support, first = np.unique(log_z, return_index=True)
    support = source_z[first]
    tree_ids = np.linspace(0, len(study.predictor.estimators_) - 1, 96, dtype=int)
    offsets = np.cumsum(
        np.r_[0, [study.predictor.estimators_[j].tree_.node_count for j in tree_ids]]
    )
    leaf = (
        np.stack([study.predictor.estimators_[j].apply(study.xp) for j in tree_ids], axis=1)
        + offsets[:-1]
    )
    leaf_membership = csr_matrix(
        (np.ones(leaf.size), (np.repeat(np.arange(len(leaf)), len(tree_ids)), leaf.ravel())),
        shape=(len(leaf), int(offsets[-1])),
    )
    importance = np.full(17, 0.4 / 17)
    importance[LEVELS >= 0.50] += 0.4 / np.sum(LEVELS >= 0.50)
    importance[LEVELS >= 0.75] += 0.1 / 4
    importance[3] += 0.1
    threshold_weights = importance / (LEVELS * (1.0 - LEVELS))

    def pair_features(ids, u, scale0):
        return np.c_[
            study.xp[ids], u, (u - log_prediction_p[ids]) / residual_scale(spread_p[ids], scale0)
        ]

    def pilot_ratio(ref, ev, obs_u, scale0, mode):
        u = log_z[ref]
        s = residual_scale(spread_p[ref], scale0)
        vs = np.arcsinh((u - log_prediction_p[ref]) / s)
        m = np.median(u)
        sd = max(1e-3, np.subtract(*np.quantile(u, [0.75, 0.25])) / 1.35)
        ub = np.linspace(min(u.min(), m - 4.0 * sd) - 0.02, max(u.max(), m + 4.0 * sd) + 0.02, 181)
        uc = 0.5 * (ub[:-1] + ub[1:])
        du = ub[1] - ub[0]
        dens = gaussian_filter1d(
            np.histogram(u, bins=ub)[0].astype(float),
            max(0.85, (0.8 * sd * len(u) ** (-0.2)) / du),
            mode="constant",
        ) / (len(u) * du)
        pu = np.interp(obs_u, uc, dens, left=0.0, right=0.0)
        pu = 0.99 * pu + 0.01 / (np.pi * sd * (1.0 + ((obs_u - m) / sd) ** 2))
        vb = np.linspace(-6.0, 6.0, 101)
        vc = 0.5 * (vb[1:] + vb[:-1])
        dv = vb[1] - vb[0]
        vbin = np.clip(np.searchsorted(vb, vs, side="right") - 1, 0, len(vc) - 1)
        gg = np.bincount(vbin, minlength=len(vc)).astype(float)
        prior = gaussian_filter1d(gg, 1.1, mode="nearest")
        prior = prior / prior.sum()
        if mode == 1:
            hc = np.quantile(log_prediction_p[ref], [0.33, 0.67])
            sc = np.median(spread_p[ref])

            def group(ids):
                return 2 * np.searchsorted(hc, log_prediction_p[ids]) + (
                    spread_p[ids] >= sc
                ).astype(int)

            codes = group(ref) * len(vc) + vbin
            hist = np.bincount(codes, minlength=6 * len(vc)).reshape(6, -1).astype(float)
            hist = gaussian_filter1d(hist, 1.15, axis=1, mode="nearest") + 30.0 * prior
            hist /= hist.sum(1)[:, None]
            dens_v = hist[group(ev)]
        else:
            A = leaf_membership[ref]
            B = leaf_membership[ev]
            one = csr_matrix(
                (np.ones(len(ref)), (np.arange(len(ref)), vbin)), shape=(len(ref), len(vc))
            )
            mat = (A.T @ one).toarray()
            mat = gaussian_filter1d(mat.astype(float), 1.2, axis=1, mode="nearest") + 15.0 * prior
            mat /= mat.sum(1)[:, None]
            dens_v = np.asarray(B @ mat) / len(tree_ids)
        ss = residual_scale(spread_p[ev], scale0)
        d = (obs_u - log_prediction_p[ev]) / ss
        v = np.arcsinh(d)
        pos = (v - vc[0]) / dv
        lo = np.floor(pos).astype(int)
        frac = pos - lo
        lo = np.clip(lo, 0, len(vc) - 1)
        up = np.clip(lo + 1, 0, len(vc) - 1)
        ix = np.arange(len(ev))
        pv = ((1.0 - frac) * dens_v[ix, lo] + frac * dens_v[ix, up]) / dv
        pv = np.maximum(pv, 1.0e-7)
        return pv / (ss * np.sqrt(1.0 + d * d) * pu)

    def fit_directional_networks(p, style, seed):
        torch = nu.torch
        rg0 = np.random.default_rng(seed)
        ap = rg0.permutation(p)
        nv = 2 * (len(p) // 10)
        tr = split_pairs(ap[nv:], rg0)
        val = split_pairs(ap[:nv], rg0)
        full = split_pairs(p, rg0)
        base = max(1.0e-6, float(np.median(spread_p[p])))
        mode = 2 if style == 2 else 1
        W = 96
        cap = 22 * (len(p) // 2) ** sim.CONFIG["neural"]["cap_power"]
        fits = []
        for k in range(2):
            own, don = tr[k]
            vid, _ = val[k]
            allown, alldon = full[k]
            ss = seed + 179 * k + 77
            torch.manual_seed(int(ss))
            rg = np.random.default_rng(ss)
            A = pair_features(own, log_z[don], base)
            B = pair_features(own, log_z[own], base)
            loc = A.mean(0)
            scale = np.maximum(A.std(0), 1e-6)
            loc[-1] = np.median(B[:, -1])
            scale[-1] = max(0.12, np.subtract(*np.quantile(B[:, -1], [0.8, 0.2])) / 1.68)

            def arr(t):
                return torch.as_tensor(t, dtype=torch.float32)

            def ft(z):
                return arr((z - loc) / scale)

            aa, bb = ft(A), ft(B)
            pp = pilot_ratio(don, np.r_[own, own], np.r_[log_z[don], log_z[own]], base, mode)
            ta, tb = map(arr, np.split(np.log1p(np.clip(pp, 0.0, cap)), 2))
            net = nu.Network(15, dict(hidden=[W, W], cap=cap), "ratio")
            with torch.no_grad():
                first = net.layers[0]
                kn = np.quantile((B[:, -1] - loc[-1]) / scale[-1], np.linspace(0.015, 0.985, 28))
                for a, knot in enumerate(kn):
                    sg = 1 if a % 2 == 0 else -1
                    first.weight[a].zero_()
                    first.weight[a, -1] = sg
                    first.bias[a] = -sg * float(knot)
            cuts = np.quantile(observed_error[own], LEVELS)
            prop = arr(threshold_weights)
            # Use fixed empirical donor quantiles for optimization only. Final inference
            # always integrates every redshift in the appropriate training complement.
            ngrid = 192
            grid = np.quantile(log_z[don], (np.arange(ngrid) + 0.5) / ngrid)
            gg = arr(grid)
            hbreak = np.quantile(spread_p[own], [1.0 / 3, 2.0 / 3])
            dcut = np.median(log_prediction_p[own])

            def groups(ids):
                if style == 1:
                    return 2 * np.searchsorted(hbreak, spread_p[ids]) + (ratio[ids] >= 1.0).astype(
                        int
                    )
                return 2 * np.searchsorted(hbreak, spread_p[ids]) + (
                    log_prediction_p[ids] >= dcut
                ).astype(int)

            def quad(ids):
                hh = log_prediction_p[ids]
                sc = residual_scale(spread_p[ids], base)
                lo, hi = redshift_brackets(hh, cuts, grid)
                l = (log_z[ids, None] < hh[:, None] - np.log1p(cuts)).astype(float)
                r = (log_z[ids, None] > hh[:, None] - np.log1p(-cuts)).astype(float)
                grp = groups(ids)
                ww = 0.45 * np.ones(len(ids)) + 0.55 * ratio[ids] / max(1.0e-8, ratio[ids].mean())
                return (
                    arr((study.xp[ids] - loc[:-2]) / scale[:-2]),
                    arr(hh),
                    arr(sc),
                    torch.as_tensor(lo),
                    torch.as_tensor(hi),
                    arr(l),
                    arr(r),
                    grp,
                    ww,
                )

            qt = quad(own)
            qv = quad(vid)
            gr = qt[-2]
            weights = qt[-1]
            active = np.unique(gr)
            pools = [np.flatnonzero(gr == g) for g in active]
            ng = len(pools)
            gw = arr([weights[ii].sum() / weights.sum() for ii in pools])
            obsL = arr(
                np.stack(
                    [np.average(qt[5][ii].numpy(), weights=weights[ii], axis=0) for ii in pools]
                )
            )
            obsR = arr(
                np.stack(
                    [np.average(qt[6][ii].numpy(), weights=weights[ii], axis=0) for ii in pools]
                )
            )

            def batch(q, ii):
                xx, hh, sc, lo, hi, yl, yr, _, _ = q
                layer = net.layers[0]
                lhs = torch.nn.functional.linear(xx[ii], layer.weight[:, :-2], layer.bias)
                u = (gg - float(loc[-2])) / float(scale[-2])
                d = ((gg[None, :] - hh[ii, None]) / sc[ii, None] - float(loc[-1])) / float(
                    scale[-1]
                )
                hh1 = torch.relu(
                    lhs[:, None, :]
                    + u[None, :, None] * layer.weight[None, None, :, -2]
                    + d[:, :, None] * layer.weight[None, None, :, -1]
                )
                raw = torch.clamp(
                    net.layers[4](torch.relu(net.layers[2](hh1))).squeeze(-1), 0.0, cap
                )
                cs = torch.cat([torch.zeros_like(raw[:, :1]), raw.cumsum(1)], dim=1)
                den = cs[:, -1:]
                safe = torch.clamp(den, min=1e-7)
                l = torch.where(den > 1e-7, cs.gather(1, lo[ii]) / safe, lo[ii].float() / ngrid)
                r = torch.where(
                    den > 1e-7, 1.0 - cs.gather(1, hi[ii]) / safe, 1.0 - hi[ii].float() / ngrid
                )
                return l, r, yl[ii], yr[ii], den / ngrid

            viv = np.arange(len(vid))

            def valid_score():
                with torch.no_grad():
                    lv = []
                    rv = []
                    for z in np.array_split(viv, max(1, len(viv) // 100)):
                        l, r, _, _, _ = batch(qv, z)
                        lv.append(l.numpy())
                        rv.append(r.numpy())
                    l, r = np.vstack(lv), np.vstack(rv)
                    yl = qv[5].numpy()
                    yr = qv[6].numpy()
                    e = l + r - yl - yr
                    wg = qv[-1] / qv[-1].sum()
                    rc = float(
                        (wg @ (e**2)) @ threshold_weights
                        + 3 * ((wg @ e) ** 2 @ threshold_weights)
                        + 0.15 * (wg @ ((l - yl) ** 2 + (r - yr) ** 2)) @ threshold_weights
                    )
                    for g in np.unique(qv[-2]):
                        ix = qv[-2] == g
                        rc += float(
                            ((wg[ix] @ e[ix]) ** 2 @ threshold_weights)
                            / (wg[ix].sum() + 8.0 / len(wg))
                        )
                    return rc

            opt = torch.optim.Adam(net.parameters(), lr=0.002)
            best = np.inf
            state = None
            for step in range(1, 661):
                ii = rg.integers(len(own), size=384)
                f0, f1 = net(aa[ii]), net(bb[ii])
                pilot = 0.5 * (
                    (torch.log1p(f0) - ta[ii]).square().mean()
                    + (torch.log1p(f1) - tb[ii]).square().mean()
                )
                m = 10
                ix = np.concatenate(
                    [
                        rg.choice(z, size=m, replace=True, p=weights[z] / weights[z].sum())
                        for z in pools
                    ]
                )
                left, right, yl, yr, mass = batch(qt, ix)
                e = left + right - yl - yr
                brier = (e.square() @ prop).reshape(ng, m).mean(1)
                le = left.reshape(ng, m, -1).mean(1) - obsL
                re = right.reshape(ng, m, -1).mean(1) - obsR
                calib = (le + re).square() @ prop + 0.35 * ((le.square() + re.square()) @ prop)
                tail = ((left - yl).square() + (right - yr).square()) @ prop
                objective = (
                    (0.5 if step <= 250 else 0.15) * pilot
                    + gw @ (brier + 0.15 * tail.reshape(ng, m).mean(1) + 5.0 * calib)
                    + 0.035 * torch.log(torch.clamp(mass, min=0.05)).square().mean()
                )
                assert torch.isfinite(objective)
                opt.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 6.0)
                opt.step()
                if step % 60 == 0:
                    vv = valid_score()
                    if vv < best:
                        best = vv
                        state = nu.copy.deepcopy(net.state_dict())
            net.load_state_dict(state)
            opt = torch.optim.Adam(net.parameters(), lr=0.00013)
            best = np.inf
            for step in range(1, 321):
                ii = rg.integers(len(own), size=512)
                ls = net(aa[ii]).square().mean() - 2.0 * net(bb[ii]).mean()
                opt.zero_grad()
                ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if step in (5, 15, 30, 60, 100, 160, 240, 320):
                    vv = valid_score()
                    if vv < best:
                        best = vv
                        bst = step
                        state = nu.copy.deepcopy(net.state_dict())
            net.load_state_dict(state)
            opt = torch.optim.Adam(net.parameters(), lr=0.000055)
            af = ft(pair_features(allown, log_z[alldon], base))
            bf = ft(pair_features(allown, log_z[allown], base))
            for step in range(max(5, min(24, bst // 8))):
                ii = rg.integers(len(allown), size=512)
                ls = net(af[ii]).square().mean() - 2 * net(bf[ii]).mean()
                opt.zero_grad()
                ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval().double()
            fits.append(nu.RatioFit(net, loc, scale, 0.0, 1.0, {}))
        return dict(fits=fits, base=base)

    n, m = len(study.xp), len(study.xq)
    cdf = np.zeros((n + m, len(log_support) + 1))
    for k, held in enumerate(study.folds):
        tr = complement(study.folds, k)
        models = [
            (weight, fit_directional_networks(tr, style, MODEL_SEED + 1109 * k + 77))
            for style, weight in enumerate(FOLD_MODELS[k], 1)
            if weight
        ]
        donor = tr[np.argsort(log_z[tr])]
        ys = log_z[donor]
        end = np.searchsorted(ys, log_support, side="right")
        x = np.r_[study.xp[held], study.xq]
        h = np.r_[log_prediction_p[held], log_prediction_q]
        disp = np.r_[spread_p[held], spread_q]
        left, right = FOLD_SCALES[k]
        for start in range(0, len(x), 64):
            ids = np.arange(start, min(start + 64, len(x)))
            first = signed_weights(models, x[ids], h[ids], disp[ids], ys, left)
            second = signed_weights(models, x[ids], h[ids], disp[ids], ys, right)
            weights = np.where(ys[None, :] <= h[ids, None], first, second)
            mass = weights.sum(1)
            good = mass > 0
            weights[good] /= mass[good, None]
            weights[~good] = 1.0 / len(ys)
            prefix = np.c_[np.zeros(len(ids)), np.cumsum(weights, axis=1)]
            prefix[:, -1] = 1.0
            values = np.c_[np.zeros(len(ids)), prefix[:, end]]
            p = ids < len(held)
            cdf[held[ids[p]]] = values[p]
            cdf[n + ids[~p] - len(held)] += len(held) / n * values[~p]
    return ratio, observed_error, support, cdf


# Covariate density ratios and network initialization

def initialize_hinge_ratio(net, a, b, *unused):
    # Express a regularized linear-spline least-squares density ratio as the
    # starting point of the same two-hidden-layer ReLU net, on raw inputs.
    # No response or domain identifiers are available at prediction time.
    d = a.shape[1]
    width = net.layers[0].weight.shape[0]
    mu = a.mean(0)
    sd = np.maximum(a.std(0), 1.0e-6)
    aa = (a - mu) / sd
    bb = (b - mu) / sd
    rng = np.random.default_rng(7081 + d)
    vv = []
    kk = []

    def add(v, t):
        if len(vv) < width:
            u = v / max(np.linalg.norm(v), 1.0e-10)
            vq = np.asarray(aa @ u)
            knot = float(np.quantile(vq, t))
            if np.max(vq) - knot < 1.0e-9:
                knot = float(np.min(vq))
            vv.append(u)
            kk.append(knot)

    eye = np.eye(d)
    for t, sg in ((0.25, 1), (0.25, -1), (0.65, 1), (0.65, -1)):
        for j in range(d):
            add(sg * eye[j], t)
    if d == 13:
        # Domain membership can depend on nearby galaxies and sky position.
        group = [0, 5, 6, 7, 8, 9, 10]
        for t in (0.15, 0.50, 0.85):
            for j in group:
                add(eye[j], t)
        while len(vv) < width:
            j, k = rng.choice(group, 2, replace=False)
            add(eye[j] + rng.choice([-1.0, 1.0]) * eye[k], rng.choice([0.25, 0.50, 0.75]))
    else:
        for t in np.arange(0.05, 0.96, 0.05):
            add(eye[-1], t)
        while len(vv) < width:
            for j in rng.permutation(d - 1):
                add(eye[-1] + rng.choice([-1.0, 1.0]) * eye[j], rng.choice([0.25, 0.50, 0.75]))
                if len(vv) >= width:
                    break
    v = np.stack(vv)
    kn = np.asarray(kk)
    za = np.maximum(aa @ v.T - kn, 0.0)
    zb = np.maximum(bb @ v.T - kn, 0.0)
    zbar = za.mean(0)
    zz = za - zbar
    penalty = 0.02
    beta = np.linalg.solve(zz.T @ zz / len(za) + penalty * np.eye(width), zb.mean(0) - zbar)

    def assign(t, x):
        t.copy_(nu.torch.as_tensor(x, dtype=t.dtype))

    with nu.torch.no_grad():
        assign(net.layers[0].weight, v)
        assign(net.layers[0].bias, -kn)
        assign(net.layers[2].weight, np.eye(width))
        net.layers[2].bias.zero_()
        assign(net.layers[4].weight, beta[None, :])
        net.layers[4].bias.fill_(float(1.0 - zbar @ beta))


def balance_features(x, xc, sc):
    v = np.clip((x - xc) / sc, -3.0, 3.0)
    ids = [0, 1, 2, 5, 6, 7, 8]
    crowd = np.clip(x[:, 8], 0.0, 1.0)
    more = np.clip(x[:, 8] - 1.0, 0.0, 1.0)
    pair = np.minimum(np.maximum(v[:, [2, 5, 6]], 0.0), crowd[:, None])
    return np.c_[
        v, np.maximum(v[:, ids], 0.0), crowd, more, pair, np.minimum(x[:, 11:13], crowd[:, None])
    ]


def domain_classifier(a, b, seed):
    x = np.r_[a, b]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    weight = np.r_[np.full(len(a), len(x) / (2 * len(a))), np.full(len(b), len(x) / (2 * len(b)))]
    m = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=2.0,
        learning_rate=0.06,
        early_stopping=False,
        random_state=seed,
    )
    return m.fit(x, y, sample_weight=weight)


def domain_score_bins(a, b, xa, xb, seed):
    model = domain_classifier(a, b, seed)
    pa = model.predict_proba(a)[:, 1]
    pb = model.predict_proba(b)[:, 1]
    split = np.unique(np.quantile(np.r_[pa, pb], np.arange(1, 10) / 10))

    def score(x):
        d = (model.predict_proba(x)[:, 1, None] > split).astype(float)
        return d

    ca, cb = score(a), score(b)
    mu = ca.mean(0)
    sd = np.maximum(0.15, np.sqrt(0.5 * (ca.var(0) + cb.var(0))))
    return (score(xa) - mu) / sd, (score(xb) - mu) / sd


def validation_risk(wp, wq, fx, fy, rx, ry, dx, dy):
    pp = wp / (1.0 + wp)
    pq = wq / (1.0 + wq)
    domain = 0.5 * (np.mean(pp**2) + np.mean((1.0 - pq) ** 2))

    def err(x, y):
        return np.mean((np.mean(wp[:, None] * x, axis=0) - y.mean(0)) ** 2)

    m = (wp.mean() - 1.0) ** 2
    v = err(fx, fy)
    s = err(rx, ry)
    d = err(dx, dy)
    return np.array(
        [
            domain + m + (v + s + d) / 3.0,
            m,
            v,
            s,
            d,
            domain,
            wp.mean(),
            np.mean(wp**2) - 2 * wq.mean(),
        ]
    )


def curve_ratio(study):
    """Five cross-fitted 96-by-96 ratios, validated on domain loss and covariate balance."""

    def prediction_summary(x, h):
        pr = np.stack([t.predict(x) for t in study.predictor.estimators_])
        disc = np.abs((pr - h[None, :]) / (1.0 + pr))
        return np.c_[
            h, np.std(pr, axis=0) / (1.0 + h), np.quantile(disc, [0.25, 0.5, 0.85, 0.95], axis=0).T
        ]

    source_summary = prediction_summary(study.xp, study.hp)
    target_summary = prediction_summary(study.xq, study.hq)
    summaries = {
        np.ascontiguousarray(x).tobytes(): v
        for x, v in zip(np.r_[study.xp, study.xq], np.r_[source_summary, target_summary])
    }

    def lookup_summary(x):
        return np.stack([summaries[np.ascontiguousarray(z).tobytes()] for z in x])

    def balance_basis(a, b):
        loc = a.mean(0)
        scale = np.maximum(a.std(0), 1.0e-6)
        h = lookup_summary(a)
        levels = [0.25, 0.5, 0.75, 0.85, 0.90]
        cut = np.quantile(h[:, 4], levels)
        hc = np.quantile(h[:, 0], [0.25, 0.5, 0.75])
        bright = np.median(a[:, 0])

        def basis(x):
            raw = balance_features(x, loc, scale)
            d = lookup_summary(x)
            g = d[:, 4, None] > cut
            c = (x[:, 8] >= 1).astype(float)
            v = x[:, 0] >= bright
            risk = np.c_[
                d,
                (d[:, 0, None] > hc).astype(float),
                g.astype(float),
                g * c[:, None],
                g * v[:, None],
            ]
            return raw, risk

        aa1, aa2 = basis(a)
        bb1, bb2 = basis(b)

        def center(v, z):
            sd = np.maximum(np.sqrt(0.5 * (v.var(0) + z.var(0))), 1.0e-5)
            return v.mean(0), sd

        mr, sr = center(aa1, bb1)
        mh, sh = center(aa2, bb2)

        def out(x):
            r, h = basis(x)
            return np.clip((r - mr) / sr, -4.0, 4.0), np.clip((h - mh) / sh, -4.0, 4.0)

        return out

    def train_balanced_ratio(a, b, valid, width, cap, seed, cfg, policy):
        torch = nu.torch
        rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed))
        av, bv = valid
        center = a.mean(0)
        scale = np.maximum(a.std(0), 1.0e-6)

        def arr(z):
            return torch.as_tensor(z, dtype=torch.float32)

        aa, bb, va, vb = [arr((z - center) / scale) for z in (a, b, av, bv)]
        basis = balance_basis(a, b)
        fa, ha = basis(a)
        fb, hb = basis(b)
        fva, hva = basis(av)
        fvb, hvb = basis(bv)
        da, db = domain_score_bins(a, b, av, bv, seed + 923)
        F = arr(fa)
        H = arr(ha)
        mf = arr(fb.mean(0))
        mh = arr(hb.mean(0))
        ixp = np.array_split(rng.permutation(len(a)), 2)
        ixq = np.array_split(rng.permutation(len(b)), 2)
        ra = np.empty(len(a))
        rb = np.empty(len(b))
        for t in range(2):
            pilot = domain_classifier(a[ixp[t]], b[ixq[t]], seed + 20 + t)
            for z, ii, r in [(a, ixp[1 - t], ra), (b, ixq[1 - t], rb)]:
                probs = pilot.predict_proba(z[ii])[:, 1]
                r[ii] = np.minimum(probs / np.maximum(1.0 - probs, 1.0e-8), cap)
        labA = arr(np.log1p(ra))
        labB = arr(np.log1p(rb))
        net = nu.Network(a.shape[1], dict(hidden=[width, width], cap=cap), "ratio")
        initialize_hinge_ratio(net, a, b)
        with torch.no_grad():
            net.layers[-1].weight.mul_(0.2)
            net.layers[-1].bias.mul_(0.2).add_(0.8)
        opt = torch.optim.Adam(net.parameters(), lr=0.0015)
        for step in range(400):
            i = rng.integers(len(a), size=512)
            j = rng.integers(len(b), size=256)
            wp = net(aa[i])
            wq = net(bb[j])
            fit = 0.5 * (
                (torch.log1p(wp) - labA[i]).square().mean()
                + (torch.log1p(wq) - labB[j]).square().mean()
            )
            imbalance = (
                (wp.mean() - 1.0).square()
                + 0.5 * ((wp[:, None] * F[i]).mean(0) - mf).square().mean()
                + 0.5 * ((wp[:, None] * H[i]).mean(0) - mh).square().mean()
            )
            val = fit + imbalance
            assert torch.isfinite(val)
            opt.zero_grad()
            val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        opt = torch.optim.Adam(net.parameters(), lr=0.0006)
        best = np.inf
        best_step = 0
        for step in range(1, 801):
            i = rng.integers(len(a), size=cfg["batch_size"])
            j = rng.integers(len(b), size=cfg["batch_size"])
            loss = net(aa[i]).square().mean() - 2 * net(bb[j]).mean()
            assert torch.isfinite(loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if step % 40 == 0:
                with torch.no_grad():
                    pv = net(va).detach().numpy().astype(float)
                    qv = net(vb).detach().numpy().astype(float)
                score = validation_risk(pv, qv, fva, fvb, hva, hvb, da, db)[0]
                if score < best:
                    best = score
                    best_step = step
                    vl = float(np.mean(pv * pv) - 2 * qv.mean())
                    state = nu.copy.deepcopy(net.state_dict())
        assert best_step >= 40
        net.load_state_dict(state)
        net.eval().double()
        return nu.RatioFit(net, center, scale, 0.0, 1.0, {}), dict(
            width=width, cap=cap, loss=vl, updates=best_step
        )

    weights = np.empty(len(study.xp))
    policy = dict(TABLE_RATIO_POLICY, widths=[96], cap_multiplier=16)
    for k, held in enumerate(study.folds):
        train = complement(study.folds, k)
        fit = nu.fit_ratio(
            study.xp[train],
            study.xq,
            sim.CONFIG["neural"],
            CURVE_SEED + 170 * k + 515,
            policy,
            trainer=train_balanced_ratio,
        )
        weights[held] = fit.predict(study.xp[held])
    return weights, source_summary[:, 1], target_summary[:, 1]


def initialize_tree_ratio(net, a, b, *unused):
    """Encode splits of a source-versus-target classification tree as initial ReLU ramps."""
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1.0e-6)
    x, z = (a - center) / scale, (b - center) / scale
    classifier = DecisionTreeClassifier(
        max_leaf_nodes=24, min_samples_leaf=max(50, len(a) // 100), random_state=439
    )
    classifier.fit(
        np.vstack([x, z]),
        np.r_[np.zeros(len(a)), np.ones(len(b))],
        sample_weight=np.r_[np.ones(len(a)) / len(a), np.ones(len(b)) / len(b)],
    )
    tree = classifier.tree_
    nodes = np.flatnonzero(tree.children_left != -1)
    position = {node: j for j, node in enumerate(nodes)}

    def leaves(node=0, path=()):
        if tree.children_left[node] < 0:
            yield node, path
        else:
            j = position[node]
            yield from leaves(tree.children_left[node], path + ((j, -1),))
            yield from leaves(tree.children_right[node], path + ((j, 1),))

    source_leaf, target_leaf = classifier.apply(x), classifier.apply(z)
    first, second, output = net.layers[0], net.layers[2], net.layers[4]
    spread = 0.08
    with nu.torch.no_grad():
        for j, node in enumerate(nodes):
            for i, offset in [(2 * j, spread), (2 * j + 1, -spread)]:
                first.weight[i].zero_()
                first.weight[i, tree.feature[node]] = 1.0
                first.bias[i] = -tree.threshold[node] + offset
        output.weight.zero_()
        output.bias.fill_(1.0)
        for row, (leaf, path) in enumerate(leaves()):
            n0 = np.sum(source_leaf == leaf)
            n1 = np.sum(target_leaf == leaf)
            odds = (n1 * len(a) / len(b) + 20.0) / (n0 + 20.0)
            second.weight[row].zero_()
            bias = 1 - len(path)
            for k, sign in path:
                second.weight[row, 2 * k] += sign / (2 * spread)
                second.weight[row, 2 * k + 1] -= sign / (2 * spread)
                if sign < 0:
                    bias += 1
            second.bias[row] = bias
            output.weight[0, row] = odds - 1.0
