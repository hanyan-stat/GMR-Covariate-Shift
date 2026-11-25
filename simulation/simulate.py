#!/usr/bin/env python3
"""Reproduce the pointwise and seven-quantile simulation tables.

The prescribed settings below are shared by all workers. N, M and TAU are
worker-local context, reset before every replication. Replications use processes,
not shared Python threads.
"""
import sys
sys.dont_write_bytecode = True
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import gc
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, expit, roots_hermitenorm
from scipy.linalg import svd
from sklearn.preprocessing import SplineTransformer
from .nuisance import Observed, initialize, stamp, elapsed, data_hash, json_compatible, fit_ratio, fit_dependence, dependence_weights, _weights, fixed_network, fit_localized, weighted_quantile, fit_sieve, max10_point, cdf_at, RATE, BATCH
from .inference import (
    CUTOFF, checked, density, wald, two_sample, one_sample, family_step_profile,
    certified_minimum, maximum_profile, Profile, dual
)
METHODS = ("OEL-oracle", "OEL-est", "Plug-Sieve", "Plug-DRF", "LDML", "DRW")
SIZES = (1000, 2000, 5000, 10000)
MASTER, REPS, DRAW, R_THREADS = 2026111907, 500, 2000, 3
JOINT_TAU = np.linspace(.2, .8, 7)
POINT_TAU = np.array([.5])
TRUTH = np.array([-4.546430737105005, -4.1187303372080315, -3.711556677182649,
                  -3.2587762042534734, -2.6059876323705753, 3.215678695168975,
                  4.24400493612259])
PRIMARY_TRUTH = np.array([-3.258776204253473, -1.9860445772774868])
POINT = CUTOFF
N, M, TAU = 10000, 5000, JOINT_TAU
KNOTS = [0, .05, .15, .30, .50, .70, .85, .90, .95, 1]
CONFIG = {'neural': {'ratio_updates': 600,
                     'check_every': 20,
                     'batch_size': 512,
                     'learning_rate': 0.002,
                     'gradient_clip': 5,
                     'cap_power': 0.1}}
BASE_R = {'maximum_updates': 600,
          'minimum_updates': 80,
          'patience': 8,
          'refit_fraction': 0.25,
          'refit_min': 20,
          'refit_max': 100,
          'refit_learning_rate': 0.00025,
          'width_by_n': {'2000': 16, '5000': 32, '10000': 48, '1000': 16},
          'cap_multiplier': 4.0,
          'learning_schedule': 'decay',
          'initial_checkpoint': True,
          'initialization': 'tilt'}
BASE_RHO = {'width_by_n': {'1000': 24, '2000': 24, '5000': 24, '10000': 24},
            'initialization': 'product',
            'cap_multiplier': 1.5,
            'patience': 12,
            'minimum_updates': 80,
            'maximum_updates': 2000,
            'learning_schedule': 'decay',
            'initial_checkpoint': True,
            'refit_fraction': 0.25,
            'refit_min': 20,
            'refit_max': 100,
            'refit_full_batch': True,
            'refit_learning_rate': 0.00025}


def generate(seed, n=10000):
    rng = np.random.default_rng(seed)
    xp = rng.standard_normal((10000, 10))
    xq = rng.standard_normal((5000, 10))
    hp = rng.random(10000) < .1
    hq = rng.random(5000) < 1 / 3
    xp[:, 0] += np.where(hp, 4., -4.)
    mu = mean_response(xp)
    yp = mu + rng.standard_normal(10000)
    xq[:, 0] += np.where(hq, 5., -4.)
    return Observed(xp[:n].copy(), yp[:n].copy(), xq[:n // 2].copy())


def ratio(xp, xq, cfg, seed):
    policy = dict(BASE_R, widths=[BASE_R["width_by_n"][str(N)]])
    return fit_ratio(xp, xq, cfg, seed, policy)


def mean_response(x):
    return 4 * np.tanh(x[:, 0]) + .5 * np.tanh(x[:, 0] + 4 + x[:, 1:].sum(1) / 3)


def true_ratio(x):
    x1 = x[:, 0]
    post = expit(8 * x1 - np.log(9.))
    return (1 - post) * (20 / 27) + post * (10 / 3) * np.exp(x1 - 4.5)


def population_truth(size):
    (z, w) = roots_hermitenorm(size)
    mass = np.outer(w, w) / (2 * np.pi)
    low = z[:, None] - 4
    high = z[:, None] + 5
    mu_low = 4 * np.tanh(low) + 0.5 * np.tanh(low + 4 + z[None, :])
    mu_high = 4 * np.tanh(high) + 0.5 * np.tanh(high + 4 + z[None, :])

    def F(t):
        return np.sum(mass * (2 / 3 * ndtr(t - mu_low) + 1 / 3 * ndtr(t - mu_high)))
    return np.array([brentq(lambda x: F(x) - tau, -10, 10, xtol=1e-12) for tau in JOINT_TAU])


def primary_truth(size):
    z, w = roots_hermitenorm(size)
    mass = np.outer(w, w) / (2 * np.pi)
    lo = z[:, None] - 4
    hi = z[:, None] + 5
    ml = 4 * np.tanh(lo) + .5 * np.tanh(lo + 4 + z[None, :])
    mh = 4 * np.tanh(hi) + .5 * np.tanh(hi + 4 + z[None, :])

    def F(t, k):
        return float(np.sum(mass * ((2 / 3) * ndtr(t - ml)**k + (1 / 3) * ndtr(t - mh)**k)))
    return np.array([brentq(lambda t:F(t, k) - .5, -10, 10, xtol=1e-12) for k in (1, 10)])


def gaussian_cutoff(p, q, normal):
    covariance = np.zeros((q.shape[1], q.shape[1]))
    for group in (p, q):
        if group is not None:
            centered = group - group.mean(0)
            covariance += centered.T @ centered / len(group)**2
    sd = np.sqrt(np.diag(covariance))
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ArithmeticError("invalid multiplier variance")
    values, basis = np.linalg.eigh(covariance / np.outer(sd, sd))
    if values[0] < -1e-10:
        raise ArithmeticError("non-positive-semidefinite score covariance")
    z = normal @ (basis * np.sqrt(np.maximum(values, 0))).T
    return float(np.quantile(np.max(z * z, axis=1), .95, interpolation="higher"))


def joint(obs, parts, normal):
    r = np.mean([fit["r"] for fit in parts], axis=0)
    rp = np.mean([fit["r"][:, None] * fit["cp"] for fit in parts], axis=0)
    cq = np.mean([fit["cq"] for fit in parts], axis=0)
    q = cq - TAU
    offset = q.mean(0) - rp.mean(0)
    order = np.argsort(obs.yp)
    prefix = np.cumsum(r[order]) / N
    est = []
    for j in range(len(TAU)):
        ix = np.flatnonzero(offset[j] + prefix >= 0)
        if not len(ix) or offset[j] >= 0:
            raise ArithmeticError("no localized equation root")
        est.append(float(obs.yp[order[ix[0]]]))
    est = np.array(est)
    p = r[:, None] * (obs.yp[:, None] <= est) - rp
    jac = np.array([density(obs.yp, r, e) for e in est])
    se = np.sqrt(p.var(0, ddof=1) / N + q.var(0, ddof=1) / M) / abs(jac)
    crit = gaussian_cutoff(p, q, normal)
    cover = abs(est - TRUTH) <= np.sqrt(crit) * se
    if not np.isfinite(se).all() or min(se) <= 0:
        raise ArithmeticError("invalid standard errors")
    return dict(estimate=est, covered=cover, joint=bool(cover.all()), cutoff=crit)


def pack(estimate, covered, cut, **fields):
    estimate = np.asarray(estimate)
    covered = np.asarray(covered, dtype=bool)
    if estimate.shape != (7,) or covered.shape != (7,) or not np.isfinite(estimate).all():
        raise ArithmeticError("incomplete seven-quantile output")
    return dict(estimate=estimate, covered=covered, joint=bool(covered.all()), cutoff=cut, **fields)


def cell_stat(r, cdf, k, tau):
    p = -r * cdf[:N, k]
    p[:k + 1] += r[:k + 1]
    return checked(two_sample(p, cdf[N:, k] - tau))


def fast_minimum(r, cdf):
    means = cdf.T @ np.r_[-r / N, np.full(M, 1 / M)] + np.cumsum(r) / N
    denominator = ((2 * np.max(abs(r)))**2 / N + 1 / M) * (1 + 1e-8)
    ids, minimum = [], []
    for tau in TAU:
        residual = abs(means - tau)
        lower = 4 * np.maximum(residual - 1e-10, 0)**2 / denominator
        kbest = int(np.argmin(residual))
        best = cell_stat(r, cdf, kbest, tau)
        if not np.isfinite(best):
            _, _, j, v, _ = certified_minimum(lambda c: family_step_profile(cdf, r, TAU, c))
            return j, v
        for k in np.flatnonzero(lower <= best + 1e-10):
            if k != kbest and lower[k] <= best + 1e-10:
                value = cell_stat(r, cdf, k, tau)
                if value < best or (value == best and k < kbest):
                    kbest, best = int(k), value
        ids.append(kbest)
        minimum.append(best)
    return np.asarray(ids), np.asarray(minimum)


def oracle(obs, seed, normal):
    r = true_ratio(obs.xp)
    mu = mean_response(np.vstack((obs.xp, obs.xq)))
    cdf = SimpleNamespace(obs=obs, n=N, m=M, r=r, cdf=lambda t: ndtr(t - mu))
    radius = np.sqrt(np.log(M) / (2 * M))
    ps, qs, estimates, low_stats, high_stats = [], [], [], [], []
    for tau, truth in zip(TAU, TRUTH):
        profile = Profile(cdf, 1, tau)
        moment = lambda t: float(ndtr(t - mu[N:]).mean() - tau)
        bracket = profile.lo, profile.hi
        profile.lo = float(brentq(lambda t: moment(t) + radius, *bracket, xtol=1e-11))
        profile.hi = float(brentq(lambda t: moment(t) - radius, *bracket, xtol=1e-11))
        profile.pilot = float(brentq(moment, profile.lo, profile.hi, xtol=1e-11))
        estimates.append(profile.minimize())
        p, q = profile.groups(estimates[-1])
        ps.append(p)
        qs.append(q)
        value = profile.at(truth)[0] if profile.lo <= truth <= profile.hi else np.inf
        low_stats.append(value - profile.minimum_upper)
        high_stats.append(value - profile.minimum_lower)
    crit = gaussian_cutoff(np.column_stack(ps), np.column_stack(qs), normal)
    cover = np.asarray(high_stats) <= crit
    if np.any(~cover & (np.asarray(low_stats) <= crit)):
        raise ArithmeticError("oracle coverage unresolved within minimum tolerance")
    return pack(estimates, cover, crit)


def estimated(obs, seed, normal):
    y, r, cdf = fit_oel(obs, seed)
    ids, minimum = fast_minimum(r, cdf)
    estimates = y[ids]
    fe = cdf[:, ids]
    p = r[:, None] * ((y[:, None] <= estimates) - fe[:N])
    q = fe[N:] - TAU
    crit = gaussian_cutoff(p, q, normal)
    where = np.searchsorted(y, TRUTH, side="right") - 1
    ft = cdf[:, np.maximum(where, 0)].copy()
    ft[:, where < 0] = 0
    raw = [checked(two_sample(r * ((y <= truth) - ft[:N, j]), ft[N:, j] - TAU[j]))
           for j, truth in enumerate(TRUTH)]
    return pack(estimates, np.asarray(raw) - minimum <= crit, crit, rho_width=24)


def sieve(obs, seed, normal):
    target, beta, y = fit_spline(obs)
    curve = target.mean(0) @ beta
    differences = curve[None, :] - TAU[:, None]
    ids = np.argmin(abs(differences), axis=1)
    truth_ids = np.searchsorted(y, TRUTH, side="right") - 1

    def at(k):
        return target @ beta[:, k] if k >= 0 else np.zeros(M)
    q = target @ beta[:, ids] - TAU
    minimum = np.asarray([checked(one_sample(q[:, j])) for j in range(7)])
    if not np.isfinite(minimum).all():
        raise ArithmeticError("invalid spline EL minimum")
    ranges = (np.ptp(target, axis=0) @ abs(beta)) * (1 + 1e-10)
    lower = 4 * M * np.maximum(abs(differences) - 1e-10, 0)**2 / np.maximum(ranges, 1e-9)**2
    for j, tau in enumerate(TAU):
        for k in np.flatnonzero(lower[j] <= minimum[j] + 1e-8):
            if k != ids[j] and lower[j, k] <= minimum[j] + 1e-8:
                minimum[j] = min(minimum[j], checked(one_sample(at(k) - tau, minimum[j])))
    crit = gaussian_cutoff(None, q, normal)
    raw = np.asarray([checked(one_sample(at(k) - tau)) for k, tau in zip(truth_ids, TAU)])
    return pack(y[ids], raw - minimum <= crit, crit, sieve_dimension=111, clipping=False, calibration="target_only")


def plugin(y, curve, fetch, normal):
    differences = curve[None, :] - TAU[:, None]
    ids = np.argmin(abs(differences), axis=1)
    truth_ids = np.searchsorted(y, TRUTH, side="right") - 1
    cache = fetch(np.r_[ids, truth_ids])
    q = np.column_stack([cache[k] for k in ids]) - TAU
    minimum = np.asarray([checked(one_sample(q[:, j])) for j in range(7)])
    if not np.isfinite(minimum).all():
        raise ArithmeticError("invalid forest EL minimum")
    lower = M * differences**2 / (.25 + .5 * abs(differences))
    candidates = [np.flatnonzero(lower[j] <= minimum[j] + 1e-8) for j in range(7)]
    extra = np.unique(np.concatenate(candidates))
    if len(extra):
        cache.update(fetch(extra))
    for j, tau in enumerate(TAU):
        for k in candidates[j]:
            if k != ids[j] and lower[j, k] <= minimum[j] + 1e-8:
                minimum[j] = min(minimum[j], checked(one_sample(cache[k] - tau, minimum[j])))
    crit = gaussian_cutoff(None, q, normal)
    raw = np.asarray([checked(one_sample(cache[k] - tau)) for k, tau in zip(truth_ids, TAU)])
    return pack(y[ids], raw - minimum <= crit, crit, trees=1500, ci_group_size=100)


def localized(obs, seed, normal):
    fit = fit_localized(obs, CONFIG, seed + 5000, TAU, fit_ratio=ratio)
    assert len(fit['records']) == 21
    assert all((not r['model_selection'] and r['selected']
               ['updates'] == 4000 for r in fit['records']))
    return dict(joint(obs, [fit], normal), architecture=[32, 32], updates=4000, rate=RATE, batch=BATCH, outer_partitions=1)


def primary_pack(est, covered, **kw):
    est = np.asarray(est, float)
    covered = np.asarray(covered, bool)
    if est.shape != (2,) or covered.shape != (2,) or not np.all(np.isfinite(est)):
        raise ArithmeticError("invalid primary output")
    return dict(estimate=est, covered=covered, cutoff=POINT, **kw)


def oracle_primary(obs, seed):
    r = true_ratio(obs.xp)
    mu = mean_response(np.vstack((obs.xp, obs.xq)))
    cdf = SimpleNamespace(obs=obs, n=N, m=M, r=r, cdf=lambda t: ndtr(t - mu))
    radius = np.sqrt(np.log(M) / (2 * M))
    est = []
    cover = []
    bounds = []
    lengths = []
    uncertainties = []
    components = []
    for k, true in zip((1, 10), PRIMARY_TRUTH):
        profile = Profile(cdf, k, .5)
        moment = lambda t: float((ndtr(t - mu[N:])**k).mean() - .5)
        bracket = profile.lo, profile.hi
        profile.lo = float(brentq(lambda t: moment(t) + radius, *bracket, xtol=1e-11))
        profile.hi = float(brentq(lambda t: moment(t) - radius, *bracket, xtol=1e-11))
        profile.pilot = float(brentq(moment, profile.lo, profile.hi, xtol=1e-11))
        est.append(profile.minimize())
        raw = profile.at(true)[0] if profile.lo <= true <= profile.hi else np.inf
        low = raw - profile.minimum_upper
        high = raw - profile.minimum_lower
        if low <= POINT < high:
            raise ArithmeticError("unresolved oracle membership")
        cover.append(high <= POINT)
        bounds.append([profile.lo, profile.hi])
        inv = profile.invert(POINT, float(true))
        if profile.lo <= true <= profile.hi:
            assert inv['covered'] == bool(cover[-1])
        lengths.append(inv['confidence_set_length'])
        uncertainties.append(inv['boundary_total_width'])
        components.append(len(inv['components']))
    return primary_pack(est, cover, search_bounds=bounds, ci_length=lengths, length_uncertainty=uncertainties, components=components)


def estimated_primary(obs, seed, fit=None):
    global TAU
    TAU = POINT_TAU
    y, r, cdf = fit_oel(obs, seed) if fit is None else fit
    ids, minimum = fast_minimum(r, cdf)

    def scan(c):
        a, b, _ = maximum_profile(cdf, r, c)
        return a[None, :], b[None, :]
    mv, ms, mi, mm, mt = certified_minimum(scan)
    indices = [int(ids[0]), int(mi[0])]
    minima = [float(minimum[0]), float(mm[0])]
    est = y[indices]
    cover = []
    truth_stats = []
    for k, true, mn in zip((1, 10), PRIMARY_TRUTH, minima):
        ix = np.searchsorted(y, true, side="right") - 1
        f = cdf[:, ix] if ix >= 0 else np.zeros(N + M)
        fp = np.clip(f[:N], 0, 1)
        fq = np.clip(f[N:], 0, 1)
        p = r * k * fp**(k - 1) * ((y <= true) - fp)
        q = fq**k - .5
        stat = checked(two_sample(p, q)) - mn
        cover.append(stat <= POINT)
        truth_stats.append(stat)
    qv, qs = family_step_profile(cdf, r, POINT_TAU, POINT + minima[0] + 1e-5)
    if POINT + minima[1] > mt:
        mv, ms = scan(POINT + minima[1] + 1e-5)
    masks = []
    for a, b, mn in ((qv[0], qs[0], minima[0]), (mv[0], ms[0], minima[1])):
        if np.any(b == 2):
            raise ArithmeticError("CI profile nonconvergence")
        masks.append((b == 0) & (a - mn <= POINT))
    lengths = []
    components = []
    for mask in masks:
        lengths.append(float(np.diff(y) @ mask[:-1]))
        components.append(int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]])))
        assert not mask[-1]
    return primary_pack(est, cover, profile_minimum=minima, truth_statistic=truth_stats, rho_width=24,
                        ci_length=lengths, components=components)


def plugin_primary(y, curves, fetch, kind, ranges=None):
    differences = curves - .5
    ids = np.argmin(abs(differences), axis=1)
    truth_ids = np.searchsorted(y, PRIMARY_TRUTH, side="right") - 1
    cache = fetch(np.r_[ids, truth_ids])
    minimum = []
    for a, k in enumerate((1, 10)):
        minimum.append(checked(one_sample(cache[int(ids[a])]**k - .5)))
    minimum = np.asarray(minimum)
    if not np.isfinite(minimum).all():
        raise ArithmeticError("nonfinite primary plugin initial EL")
    if kind == "forest":
        lower = M * differences**2 / (.25 + .5 * abs(differences))
    else:
        lower = 4 * M * np.maximum(abs(differences) - 1e-10, 0)**2 / np.maximum(ranges, 1e-12)**2
    candidates = [np.flatnonzero(lower[a] <= minimum[a] + 1e-8) for a in range(2)]
    extra = np.unique(np.concatenate(candidates))
    if len(extra):
        cache.update(fetch(extra))
    for a, k in enumerate((1, 10)):
        for i in candidates[a]:
            if i != ids[a] and lower[a, i] <= minimum[a] + 1e-8:
                minimum[a] = min(minimum[a], checked(
                    one_sample(cache[int(i)]**k - .5, minimum[a])))
    raw = np.array([checked(one_sample(cache[int(i)]**k - .5))
                   for i, k in zip(truth_ids, (1, 10))])
    masks = np.zeros((2, N), bool)
    candidates_ci = [np.flatnonzero(lower[a] <= minimum[a] + POINT + 1e-8) for a in range(2)]
    all_ci = np.unique(np.concatenate(candidates_ci))
    eligible = [set(map(int, c)) for c in candidates_ci]
    for begin in range(0, len(all_ci), 512):
        batch = all_ci[begin:begin + 512]
        missing = [int(i) for i in batch if int(i) not in cache]
        block = fetch(missing) if missing else {}
        for i in batch:
            f = cache[int(i)] if int(i) in cache else block[int(i)]
            for a, k in enumerate((1, 10)):
                if int(i) in eligible[a]:
                    val, status = one_sample(f**k - .5, minimum[a] + POINT)
                    if status == 2:
                        raise ArithmeticError("plugin CI numerical failure")
                    masks[a, i] = (status == 0 and val - minimum[a] <= POINT)
    lengths = []
    components = []
    for mask in masks:
        lengths.append(float(np.diff(y) @ mask[:-1]))
        components.append(int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]])))
        assert not mask[-1]
    return primary_pack(y[ids], raw - minimum <= POINT, profile_minimum=minimum,
                        truth_statistic=raw - minimum, ci_length=lengths, components=components)


def spline_primary(obs, seed):
    target, beta, y = fit_spline(obs)
    curves = np.empty((2, N))
    ranges = np.empty((2, N))
    for begin in range(0, N, 128):
        end = min(begin + 128, N)
        f = target @ beta[:, begin:end]
        for a, k in enumerate((1, 10)):
            g = f**k
            curves[a, begin:end] = g.mean(0)
            ranges[a, begin:end] = np.ptp(g, axis=0) * (1 + 1e-10)

    def fetch(ids):
        out = {}
        for i in np.unique(ids):
            out[int(i)] = target @ beta[:, i] if i >= 0 else np.zeros(M)
        return out
    out = plugin_primary(y, curves, fetch, "spline", ranges)
    out.update(sieve_dimension=111, clipping=False, calibration="target_only")
    return out


def forest_primary(obs, seed):
    with forest_prediction(obs, seed + 3) as (curves, fetch):
        out = plugin_primary(np.sort(obs.yp), curves, fetch, "forest")
    out.update(trees=1500, min_node=10, ci_group_size=100)
    return out


def localized_primary(obs, seed):
    xp, y, xq = obs.xp, obs.yp, obs.xq
    local_seed = seed + 5000
    rng = np.random.default_rng(local_seed)
    fp = np.array_split(rng.permutation(N), 3)
    fq = np.array_split(rng.permutation(M), 3)
    r = np.empty(N)
    cp = np.empty((N, 2))
    cq = np.empty((M, 2))
    pilots = []
    for fold in range(3):
        pilot = (fold + 1) % 3
        reg = (fold + 2) % 3
        ii = fp[fold]
        jj = fq[fold]
        h1 = fp[pilot]
        h2 = fp[reg]
        tp = np.r_[h1, h2]
        tq = np.r_[fq[pilot], fq[reg]]
        sk = local_seed + 10000 * fold
        rf = ratio(xp[tp], xq[tq], CONFIG["neural"], sk + 1)
        r[ii] = rf.predict(xp[ii])
        hp = np.array_split(rng.permutation(len(h1)), 2)
        hq = np.array_split(rng.permutation(len(fq[pilot])), 2)
        rw = np.empty(len(h1))
        for h in range(2):
            fit = ratio(xp[h1[hp[1 - h]]], xq[fq[pilot][hq[1 - h]]], CONFIG["neural"], sk + 10 + h)
            rw[hp[h]] = fit.predict(xp[h1[hp[h]]])
        initial = weighted_quantile(y[h1], rw, .5)
        # The Max10 pilot retains the forest pilot, with the migrated DRF budget.
        pilotobs = Observed(xp[h1], y[h1], xq[fq[pilot]])
        with forest_prediction(pilotobs, sk + 20) as (curves, fetch):
            ids = np.flatnonzero(curves[1] >= .5)
            if not len(ids):
                raise ArithmeticError("no finite LDML Max10 pilot")
            initial_max = float(np.sort(y[h1])[ids[0]])
        for a, t, fitseed in [(0, initial, sk + 150), (1, initial_max, sk + 700)]:
            fit, record = fixed_network(xp[h2], (y[h2] <= t).astype(float), fitseed, "binary")
            cp[ii, a] = fit.predict(xp[ii])
            cq[jj, a] = fit.predict(xq[jj])
        pilots.append([float(initial), initial_max])
    order = np.argsort(y)
    est = []
    cover = []
    ses = []
    pointwise = []
    for a, k in enumerate((1, 10)):
        weights = r * k * cp[:, a]**(k - 1)
        q = cq[:, a]**k - .5
        offset = q.mean() - np.mean(weights * cp[:, a])
        curve = offset + np.cumsum(weights[order]) / N
        ix = np.flatnonzero(curve >= 0)
        if not len(ix) or offset >= 0:
            raise ArithmeticError("no finite localized primary root")
        e = float(y[order[ix[0]]])
        p = weights * ((y <= e) - cp[:, a])
        point = wald(e, p, q, density(y, weights, e), PRIMARY_TRUTH[a])
        est.append(e)
        cover.append(point["covered"])
        ses.append(point["se"])
        pointwise.append(point)
    return primary_pack(est, cover, se=ses, wald_points=pointwise, pilots=pilots,
                        architecture=[32, 32], updates=4000, rate=RATE, outer_partitions=1)


def weighted_primary(obs, seed):
    y = obs.yp
    fit = fit_ratio(obs.xp, obs.xq, CONFIG["neural"], seed + 1, dict(BASE_R, cap_multiplier=8))
    raw = fit.predict(obs.xp)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ArithmeticError("invalid DRW ratio")
    weights = raw / raw.mean()
    order = np.argsort(y)
    qi = int(np.searchsorted(np.cumsum(weights[order]) / weights.sum(), .5))
    median = float(y[order[qi]])
    basis, source, inverse, record = fit_sieve(obs.xp, y)
    max10, upcrossings = max10_point(y, order, source, inverse, weights)
    target = basis(obs.xq)
    est = [median, max10]
    cover = []
    ses = []
    points = []
    for a, (k, e) in enumerate(zip((1, 10), est)):
        event, fp, fq = cdf_at(y, e, source, target, inverse)
        fp = np.clip(fp, 0, 1)
        scale = raw * k * fp**(k - 1)
        point = wald(e, scale * (event - fp), fq**k - .5, density(y, scale, e), PRIMARY_TRUTH[a])
        cover.append(point["covered"])
        ses.append(point["se"])
        points.append(point)
    return primary_pack(est, cover, se=ses, wald_points=points, upcrossings=upcrossings,
                        ratio_cap_multiplier=8, ci_type="efficiency_bound_wald")


def donor_cdf(obs, seed):
    """Fit the source dependence model once and evaluate all source and target rows."""
    n = len(obs.yp)
    order = np.argsort(obs.yp)
    x = np.vstack((obs.xp[order], obs.xq))
    fits, meta = fit_dependence(obs, BASE_RHO, CONFIG["neural"], seed + 2)
    assert meta["selected"]["width"] == 24
    cdf = np.empty((len(x), n), order="F")
    for begin in range(0, len(x), 256):
        end = min(begin + 256, len(x))
        w, _ = dependence_weights(fits, x[begin:end], obs.yp)
        if not np.isfinite(w).all() or w.min() < -1e-14:
            raise ArithmeticError("Invalid dependence weights")
        total = w.sum(1)
        zero = total == 0
        w[zero] = 1
        total[zero] = n
        w /= total[:, None]
        cdf[begin:end] = np.cumsum(w[:, order], axis=1)
    return order, obs.yp[order], cdf


def fit_oel(obs, seed):
    """Fit r and rho once, then reuse their donor CDF for every requested target."""
    fit = fit_ratio(obs.xp, obs.xq, CONFIG["neural"], seed + 1, BASE_R)
    order, y, cdf = donor_cdf(obs, seed)
    return y, fit.predict(obs.xp)[order], cdf


def fit_spline(obs):
    """Unpenalized additive cubic B-spline; retain raw predictions."""
    knots = np.quantile(obs.xp, KNOTS, axis=0)
    basis = SplineTransformer(
        knots=knots, degree=3, include_bias=False, extrapolation="constant"
    )
    source = np.column_stack((np.ones(N), basis.fit_transform(obs.xp)))
    target = np.column_stack((np.ones(M), basis.transform(obs.xq)))
    u, s, vt = svd(source, full_matrices=False, lapack_driver="gesdd")
    rank = int(np.sum(s > max(source.shape) * np.finfo(float).eps * s[0]))
    if rank != 111:
        raise ArithmeticError("Rank-deficient additive spline")
    inverse = (vt.T / s) @ u.T
    order = np.argsort(obs.yp)
    beta = np.cumsum(inverse[:, order], axis=1)
    return target, beta, obs.yp[order]


@contextmanager
def forest_prediction(obs, seed, mode="primary"):
    """Fit one R forest and request CDF columns through binary pipes, without files."""
    n, m = len(obs.yp), len(obs.xq)
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    args = [
        "timeout", "1200", os.environ.get("RSCRIPT", "Rscript"), "--vanilla",
        str(Path(__file__).with_name("drf_interface.R")),
        str(n), str(m), "10", str(seed), str(R_THREADS), mode
    ]
    with subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env
    ) as proc:
        def receive(size):
            block = proc.stdout.read(size * 8)
            if len(block) != size * 8:
                raise RuntimeError("Incomplete DRF response")
            a = np.frombuffer(block, dtype="<f8")
            if not np.isfinite(a).all():
                raise ArithmeticError("Nonfinite DRF response")
            return a

        def fetch(ids):
            ids = np.unique(ids).astype("<i4")
            proc.stdin.write(np.asarray([len(ids)], dtype="<i4").tobytes())
            proc.stdin.write((ids + 1).tobytes())
            proc.stdin.flush()
            a = receive(m * len(ids)).reshape(len(ids), m)
            return dict(zip(map(int, ids), np.clip(a, 0, 1)))

        try:
            for block in (obs.xp, obs.yp, obs.xq):
                proc.stdin.write(np.asarray(block, dtype="<f8").tobytes(order="F"))
            proc.stdin.flush()
            rows = 2 if mode == "primary" else 1
            curves = np.clip(receive(rows * n).reshape(rows, n), 0, 1)
            yield (curves if rows == 2 else curves[0]), fetch
            proc.stdin.write(b"\0" * 4)
            proc.stdin.close()
            message = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError(message[-1500:])
        except Exception as error:
            if proc.poll() is None:
                proc.kill()
            message = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(message[-1500:] + " " + str(error)) from error


def forest(obs, seed, normal):
    with forest_prediction(obs, seed + 3, "joint") as (curve, fetch):
        return plugin(np.sort(obs.yp), curve, fetch, normal)


def weighted(obs, seed, normal):
    """Density-ratio weighted quantiles with the prescribed working Wald variance."""
    y = obs.yp
    fit = fit_ratio(
        obs.xp, obs.xq, CONFIG["neural"], seed + 1, dict(BASE_R, cap_multiplier=8)
    )
    raw = fit.predict(obs.xp)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ArithmeticError("Invalid DRW ratio")
    weights = raw / raw.mean()
    order = np.argsort(y)
    ids = np.searchsorted(np.cumsum(weights[order]) / weights.sum(), TAU)
    estimates = y[order[ids]]
    basis, source, inverse, _ = fit_sieve(obs.xp, y)
    target = basis(obs.xq)
    ps, qs, points = [], [], []
    for tau, estimate, true in zip(TAU, estimates, TRUTH):
        event, fp, fq = cdf_at(y, estimate, source, target, inverse)
        fp = np.clip(fp, 0, 1)
        p, q = raw * (event - fp), fq - tau
        point = wald(estimate, p, q, density(y, raw, estimate), true)
        ps.append(p)
        qs.append(q)
        points.append(point)
    crit = gaussian_cutoff(np.column_stack(ps), np.column_stack(qs), normal)
    se = np.asarray([point["se"] for point in points])
    return pack(estimates, abs(estimates - TRUTH) <= np.sqrt(crit) * se, crit, ci_type="Wald")


def warm():
    """Compile numerical kernels before timing or starting worker processes."""
    initialize()
    f = np.asfortranarray(np.broadcast_to(np.linspace(.05, .98, 12), (20, 12)))
    family_step_profile(f, np.ones(12), JOINT_TAU, 10.)
    one_sample(np.array([-.3, .8]))
    dual(np.array([-.1, .7]), np.array([-.3, -.2, -.1]))
    _weights(
        np.array([-1., 1.]), np.arange(2), np.ones((2, 2)), np.ones((2, 2, 2)),
        np.ones((2, 2, 2)), np.zeros((2, 2)), np.ones((2, 2)), np.zeros(2),
        np.ones(2)
    )
    maximum_profile(np.asfortranarray(np.tile(np.linspace(.01, .99, 20), (30, 1))),
                    np.ones(20), 10.)


REFERENCE = [{'experiment': 'pointwise',
              'n': 1000,
              'rep': 0,
              'seed': 580332976,
              'data_hash': 'e9c56c6fe2c1385c52d0c675efa196f949a052f44ac7d8d024e18166590c0e39',
              'methods': {'OEL-oracle': {'estimate': [-3.233866405669294, -1.921798364906102],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'ci_length': [0.3524495830564556, 0.4315171579046164]},
                          'OEL-est': {'estimate': [-3.2436287527960315, -1.884892896602221],
                                      'covered': [True, True],
                                      'cutoff': 3.841458820694124,
                                      'profile_minimum': [5.840063550820976e-05, 0.000194609446120469],
                                      'truth_statistic': [0.021382548694278684, 0.6260287367616326],
                                      'ci_length': [0.3338449143932736, 0.52072414709682]},
                          'Plug-Sieve': {'estimate': [-3.203056025172383, -2.197050524469472],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'profile_minimum': [0.00016156801233140256, 0.004836233586407423],
                                         'truth_statistic': [0.5449155049896176, 0.17800281702632373],
                                         'ci_length': [0.3905272890171112, 0.6747656211682407]},
                          'Plug-DRF': {'estimate': [-3.2575280379959137, -1.884892896602221],
                                       'covered': [True, True],
                                       'cutoff': 3.841458820694124,
                                       'profile_minimum': [5.459140553174544e-07, 0.03950701357970793],
                                       'truth_statistic': [0.012075821722859738, 3.7425424783993644],
                                       'ci_length': [0.27784162285505243, 0.22281973447123837]},
                          'LDML': {'estimate': [-3.2120173469035835, -1.8539976325094307],
                                   'covered': [True, True],
                                   'cutoff': 3.841458820694124,
                                   'ci_length': [0.455424941124809, 0.44539200995731076]},
                          'DRW': {'estimate': [-3.3880473952517454, -1.9848558270894916],
                                  'covered': [True, True],
                                  'cutoff': 3.841458820694124,
                                  'ci_length': [0.32770400010922973, 0.3078453559801142]}}},
             {'experiment': 'pointwise',
              'n': 2000,
              'rep': 0,
              'seed': 580332976,
              'data_hash': '6abca5939a772879d7848e1284a2360773a887b488091c91ddccc915044c7958',
              'methods': {'OEL-oracle': {'estimate': [-3.2039837185586055, -1.884892896602221],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'ci_length': [0.28618968806871115, 0.30059869684800833]},
                          'OEL-est': {'estimate': [-3.208791459420355, -1.882413664958394],
                                      'covered': [True, False],
                                      'cutoff': 3.841458820694124,
                                      'profile_minimum': [5.0941320096593856e-05, 0.0034582249457123224],
                                      'truth_statistic': [0.39205967588830126, 4.066551693671781],
                                      'ci_length': [0.2838410155092932, 0.38063626382090066]},
                          'Plug-Sieve': {'estimate': [-3.1879055271085774, -1.9627962001948331],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'profile_minimum': [1.4222249967868548e-06, 0.00034216522186114357],
                                         'truth_statistic': [0.6721190684189191, 0.2297119858500011],
                                         'ci_length': [0.26447599401762867, 0.24664290443220072]},
                          'Plug-DRF': {'estimate': [-3.2120173469035835, -1.8794749536958362],
                                       'covered': [True, False],
                                       'cutoff': 3.841458820694124,
                                       'profile_minimum': [0.0002261569536364449, 0.0010083983165832028],
                                       'truth_statistic': [0.39763001885308036, 16.127282790308165],
                                       'ci_length': [0.22606418514936388, 0.18403393228388176]},
                          'LDML': {'estimate': [-3.2531881145599737, -1.8536229378144853],
                                   'covered': [True, True],
                                   'cutoff': 3.841458820694124,
                                   'ci_length': [0.3052862506698837, 0.2659641968591697]},
                          'DRW': {'estimate': [-3.1869253532997974, -1.7138763955235305],
                                  'covered': [True, False],
                                  'cutoff': 3.841458820694124,
                                  'ci_length': [0.2969956479780418, 0.3348219289554111]}}},
             {'experiment': 'pointwise',
              'n': 5000,
              'rep': 0,
              'seed': 580332976,
              'data_hash': '721323dfc735bf5cc68a00b8db2f8175c45ebc2442eb19ec8caedfe8d090c13a',
              'methods': {'OEL-oracle': {'estimate': [-3.2868393976320855, -1.9616269850004555],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'ci_length': [0.17590456003746935, 0.18323879168091772]},
                          'OEL-est': {'estimate': [-3.287166953927742, -1.9533101869428098],
                                      'covered': [True, True],
                                      'cutoff': 3.841458820694124,
                                      'profile_minimum': [1.0417019849537899e-05, 0.0005140077321089307],
                                      'truth_statistic': [0.43164905004760934, 0.7090073644459497],
                                      'ci_length': [0.1822795463784228, 0.19645041232872407]},
                          'Plug-Sieve': {'estimate': [-3.27370462952936, -1.999321077127335],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'profile_minimum': [9.213568041536795e-05, 0.004469307818251241],
                                         'truth_statistic': [0.10165544969378279, 0.017430126090769237],
                                         'ci_length': [0.16859305225578458, 0.13688825997938991]},
                          'Plug-DRF': {'estimate': [-3.2824042012894368, -1.9533101869428098],
                                       'covered': [True, False],
                                       'cutoff': 3.841458820694124,
                                       'profile_minimum': [8.72017797785356e-05, 0.0019874588427884524],
                                       'truth_statistic': [0.3167770776400516, 4.5288765976791385],
                                       'ci_length': [0.1536677515749898, 0.0945387954348953]},
                          'LDML': {'estimate': [-3.2963742445773043, -1.8828493348750865],
                                   'covered': [True, False],
                                   'cutoff': 3.841458820694124,
                                   'ci_length': [0.18112923492003574, 0.19701508162399062]},
                          'DRW': {'estimate': [-3.3295243067056552, -1.9453371490461584],
                                  'covered': [True, True],
                                  'cutoff': 3.841458820694124,
                                  'ci_length': [0.162269942648833, 0.1652113338312935]}}},
             {'experiment': 'pointwise',
              'n': 10000,
              'rep': 0,
              'seed': 580332976,
              'data_hash': '2bb9903d8f005f98d631a3fed9f1afe5f0ce9a0531184c812fcc444f45785d84',
              'methods': {'OEL-oracle': {'estimate': [-3.264327435340193, -1.975174253767309],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'ci_length': [0.12121089520427608, 0.16346454188210768]},
                          'OEL-est': {'estimate': [-3.264813288277256, -1.975174253767309],
                                      'covered': [True, True],
                                      'cutoff': 3.841458820694124,
                                      'profile_minimum': [1.734729435195814e-05, 0.0002922315690960614],
                                      'truth_statistic': [0.04999852439113124, 0.10324764616339066],
                                      'ci_length': [0.12220066781062977, 0.17066808960591162]},
                          'Plug-Sieve': {'estimate': [-3.2575280379959137, -1.9979006819904412],
                                         'covered': [True, True],
                                         'cutoff': 3.841458820694124,
                                         'profile_minimum': [1.3064446724058809e-05, 0.0005742063372293827],
                                         'truth_statistic': [0.010985416487376243, 0.0012022280303168617],
                                         'ci_length': [0.11227559617492044, 0.10880885949361341]},
                          'Plug-DRF': {'estimate': [-3.266243896074296, -1.9561611112164161],
                                       'covered': [True, True],
                                       'cutoff': 3.841458820694124,
                                       'profile_minimum': [1.3415765376805248e-05, 0.000342729986770299],
                                       'truth_statistic': [0.14177566331537972, 3.436064471041505],
                                       'ci_length': [0.10673532899496774, 0.09544743550528212]},
                          'LDML': {'estimate': [-3.2626628315192776, -1.908108595808142],
                                   'covered': [True, True],
                                   'cutoff': 3.841458820694124,
                                   'ci_length': [0.12523258342516197, 0.15670183278764283]},
                          'DRW': {'estimate': [-3.241056574860693, -1.9024532464851591],
                                  'covered': [True, False],
                                  'cutoff': 3.841458820694124,
                                  'ci_length': [0.1260594306682581, 0.15151041122711595]}}},
             {'experiment': 'joint',
              'n': 10000,
              'rep': 0,
              'seed': 580332976,
              'data_hash': '2bb9903d8f005f98d631a3fed9f1afe5f0ce9a0531184c812fcc444f45785d84',
              'methods': {'OEL-oracle': {'estimate': [-4.55072869616015,
                                                      -4.129675488388628,
                                                      -3.7202243827573804,
                                                      -3.264327435340193,
                                                      -2.5949970535093683,
                                                      3.098552908115479,
                                                      4.324124259726563],
                                         'covered': [True, True, True, True, True, True, True],
                                         'cutoff': 6.779223950685661},
                          'OEL-est': {'estimate': [-4.5520076944646215,
                                                   -4.127286445716652,
                                                   -3.72063083631049,
                                                   -3.264813288277256,
                                                   -2.596611717492965,
                                                   3.0985529081154795,
                                                   4.308961442785356],
                                      'covered': [True, True, True, True, True, True, True],
                                      'cutoff': 6.635469621200734},
                          'Plug-Sieve': {'estimate': [-4.5567704601825785,
                                                      -4.124257066899311,
                                                      -3.720158618280021,
                                                      -3.2575280379959137,
                                                      -2.590360134772685,
                                                      3.0985529081154795,
                                                      4.321588636333367],
                                         'covered': [True, True, True, True, True, True, False],
                                         'cutoff': 4.823877491407047},
                          'Plug-DRF': {'estimate': [-4.562305857177975,
                                                    -4.139665922863725,
                                                    -3.730974726612889,
                                                    -3.266243896074296,
                                                    -2.5966049166679066,
                                                    3.1196349249954434,
                                                    4.278882165004099],
                                       'covered': [True, True, True, True, True, True, True],
                                       'cutoff': 4.464466990673962},
                          'LDML': {'estimate': [-4.5542785262400916,
                                                -4.127286445716652,
                                                -3.730974726612889,
                                                -3.2626628315192776,
                                                -2.6066930372007597,
                                                3.106092085412838,
                                                4.313062870409401],
                                   'covered': [True, True, True, True, True, True, True],
                                   'cutoff': 6.917515625340196},
                          'DRW': {'estimate': [-4.534640085507333,
                                               -4.114252110952997,
                                               -3.7053981850720583,
                                               -3.241056574860693,
                                               -2.5522488654258875,
                                               3.178748991629825,
                                               4.321588636333367],
                                  'covered': [True, True, True, True, True, True, True],
                                  'cutoff': 6.803935766025207}}},
             {'experiment': 'pointwise',
              'n': 1000,
              'rep': 151,
              'seed': 644904463,
              'data_hash': '06fd6306397b9c318f39b31a5ec73b51e6360beb7889aeb09a2ca052ce634e61',
              'methods': {'Plug-Sieve': {'estimate': [-3.3904220653667587, -4.639404802255292],
                                         'covered': [False, True],
                                         'cutoff': 3.841458820694124,
                                         'profile_minimum': [0.0001727462675391331, 8.651149518768639e-06],
                                         'truth_statistic': [4.630273299871728, 2.4075563597812573],
                                         'ci_length': [0.2680251396628761, 0.8473398680678539]}}}]

FUNCTIONS = (oracle, estimated, sieve, forest, localized, weighted)
PRIMARY_FUNCTIONS = (
    oracle_primary, estimated_primary, spline_primary,
    forest_primary, localized_primary, weighted_primary
)
MEMORY = None


def run_one(experiment, n, rep, seed, methods=METHODS):
    """Evaluate all requested methods on one shared, untouched sample."""
    global N, M, TAU
    N, M = int(n), int(n) // 2
    TAU = POINT_TAU if experiment == "pointwise" else JOINT_TAU
    initialize()
    obs = generate(seed, N)
    signature = data_hash(obs)
    normal = np.random.default_rng([seed, 92651]).standard_normal((DRAW, 7))
    functions = PRIMARY_FUNCTIONS if experiment == "pointwise" else FUNCTIONS
    output = {}
    for name, function in zip(METHODS, functions):
        if name not in methods:
            continue
        semaphore = MEMORY if name == "OEL-est" and N >= 5000 else None
        if semaphore is not None:
            semaphore.acquire()
        timer = stamp()
        try:
            result = function(obs, seed) if experiment == "pointwise" else function(
                obs, seed, normal)
        except Exception as error:
            raise RuntimeError(
                f"{experiment}, nP={N}, replication={rep}, method={name}, seed={seed}"
            ) from error
        finally:
            duration = elapsed(timer)
            if semaphore is not None:
                semaphore.release()
        result["seconds"] = duration
        if experiment == "pointwise" and "wald_points" in result:
            result["ci_length"] = [
                point["ci_hi"] - point["ci_lo"] for point in result["wald_points"]
            ]
        if data_hash(obs) != signature:
            raise RuntimeError("A method mutated the paired observations")
        output[name] = json_compatible(result)
        gc.collect()
    return dict(
        experiment=experiment, n=N, nQ=M, rep=int(rep), seed=int(seed),
        data_hash=signature, methods=output
    )


def seeds():
    return np.random.SeedSequence(MASTER).generate_state(REPS).astype(np.int64) % 1800000000 + 1


def environment():
    """Report the installed software and source versions without fitting."""
    import importlib.metadata
    packages = {name: importlib.metadata.version(name) for name in (
        "numpy", "scipy", "torch", "numba", "llvmlite", "scikit-learn",
        "threadpoolctl", "joblib"
    )}
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    session = subprocess.run(
        [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "-e",
         'library(drf);stopifnot(as.character(packageVersion("drf"))=="1.3.0");'
         'print(sessionInfo())'],
        capture_output=True, text=True, check=True, env=env
    )
    root = Path(__file__).resolve().parent
    return dict(
        python=sys.version, platform=platform.platform(), cpus=os.cpu_count(),
        packages=packages, r_session=session.stdout,
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ("simulate.py", "nuisance.py", "inference.py", "drf_interface.R")}
    )


def sample_sd(values, axis=None):
    return np.std(values, axis=axis, ddof=1) if len(values) > 1 else None


def summarize(records):
    """Include every replication and reject duplicate or incomplete comparisons."""
    seen = set()
    output = {"pointwise": {}, "joint": {}}
    for record in records:
        key = (record["experiment"], record["n"], record["rep"])
        if key in seen or set(record["methods"]) != set(METHODS):
            raise ValueError(f"Duplicate or incomplete replication: {key}")
        seen.add(key)
        for value in record["methods"].values():
            if value.get("error") or not np.isfinite(value["estimate"]).all():
                raise ValueError(f"Invalid replication: {key}")
    for n in SIZES:
        group = sorted(
            [r for r in records if r["experiment"] == "pointwise" and r["n"] == n],
            key=lambda r: r["rep"]
        )
        if not group:
            continue
        output["pointwise"][n] = {}
        for name in METHODS:
            error = np.asarray([r["methods"][name]["estimate"] for r in group]) - PRIMARY_TRUTH
            absolute = abs(error)
            output["pointwise"][n][name] = dict(
                replications=len(group), mean_abs=absolute.mean(0),
                sd_abs=sample_sd(absolute, axis=0), rmse=np.sqrt((error**2).mean(0)),
                coverage=100 * np.asarray([r["methods"][name]["covered"] for r in group]).mean(0),
                ci_length=np.asarray([r["methods"][name]["ci_length"] for r in group]).mean(0)
            )
    group = sorted([r for r in records if r["experiment"] == "joint"], key=lambda r: r["rep"])
    if group:
        for name in METHODS:
            error = np.asarray([r["methods"][name]["estimate"] for r in group]) - TRUTH
            rms = np.sqrt((error**2).mean(1))
            seconds = np.asarray([r["methods"][name]["seconds"]["wall"] for r in group])
            output["joint"][name] = dict(
                replications=len(group), mean_rms=rms.mean(), sd_rms=sample_sd(rms),
                coverage=100 * np.mean([r["methods"][name]["joint"] for r in group]),
                time_mean=seconds.mean(), time_sd=sample_sd(seconds)
            )
    return json_compatible(output)


def significant(value):
    if value is None:
        return "--"
    if value == 0:
        return "0.00"
    rounded = float(f"{value:.3g}")
    return f"{rounded:.{max(0,2-int(np.floor(np.log10(abs(rounded)))))}f}"


def emit(kind, payload):
    print(json.dumps(json_compatible(dict(type=kind, **payload)), allow_nan=False), flush=True)


def execute(jobs, workers):
    global MEMORY
    context = get_context("fork")
    MEMORY = context.BoundedSemaphore(8)
    warm()
    if workers == 1:
        for job in jobs:
            yield run_one(*job)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = {pool.submit(run_one, *job): job for job in jobs}
        for future in as_completed(pending):
            yield future.result()


def check(workers):
    """Compare reference cases at every sample size, plus joint and tail cases."""
    data_seeds = seeds()
    for case in REFERENCE:
        if int(data_seeds[case['rep']]) != case['seed']:
            raise AssertionError('Reference seed mapping changed')
    if max(abs(primary_truth(512) - PRIMARY_TRUTH)) > 2e-07:
        raise AssertionError('Pointwise population truth check failed')
    if max(abs(population_truth(1024) - TRUTH)) > 2e-07:
        raise AssertionError('Joint population truth check failed')
    jobs = [(case['experiment'], case['n'], case['rep'], case['seed'], tuple(case['methods']))
            for case in REFERENCE]
    reference = {(r['experiment'], r['n'], r['rep']): r for r in REFERENCE}
    (comparisons, maximum_error) = (0, 0.0)
    for result in execute(jobs, workers):
        case = reference[result['experiment'], result['n'], result['rep']]
        if result['data_hash'] != case['data_hash']:
            raise AssertionError('Reference data hash changed')
        for (name, expected) in case['methods'].items():
            observed = result['methods'][name]
            if observed['covered'] != expected['covered']:
                raise AssertionError(f'Coverage changed: {name}')
            for field in ('estimate', 'cutoff', 'ci_length', 'profile_minimum', 'truth_statistic'):
                if field not in expected:
                    continue
                difference = float(
                    np.max(abs(np.asarray(observed[field]) - np.asarray(expected[field]))))
                maximum_error = max(maximum_error, difference)
                if difference > (1e-06 if field == 'ci_length' else 1e-08):
                    raise AssertionError(f'Reference mismatch: {name}, {field}, {difference}')
                comparisons += int(np.size(expected[field]))
        emit('check', dict(experiment=result['experiment'],
             n=result['n'], rep=result['rep'], passed=True))
    emit('check_summary', dict(cases=len(jobs), comparisons=comparisons,
         maximum_absolute_difference=maximum_error))


def latex_tables(summary):
    """Typeset Table 1 (absolute errors), Table 2 (coverage/length), Table 3 (joint)."""
    point = {int(n): group for n, group in summary["pointwise"].items()}
    sizes = [n for n in SIZES if n in point]
    names = {"OEL-oracle": "OEL-Oracle", "OEL-est": "OEL-DW"}
    row = r" \\"
    tables = []
    if sizes:
        counts = {point[n][name]["replications"] for n in sizes for name in METHODS}
        if len(counts) != 1:
            raise ValueError("Comparative tables require equal replication counts")
        repetitions = counts.pop()
        k = len(sizes)
        labels = ["$" + f"{n:,}".replace(",", "{,}") + "$" for n in sizes]
        t = [
            r"\begin{table}[t]", r"\centering",
            r"\caption{Empirical means of $100|\hat\theta-\theta_0|$ over " +
            str(repetitions)+r" replications. Sample SDs are below the means. " +
            r"Column headings give $n_\P$, with $n_\Q = n_\P/2$.}",
            r"\label{tab:sim-q50}", r"\small\linespread{1}\selectfont",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l" + "r" * k + r"@{\hspace{12pt}}" + "r" * k + r"@{}}",
            r"\toprule",
            rf" & \multicolumn{{{k}}}{{c}}{{$\theta_1$}} & \multicolumn{{{k}}}{{c}}{{$\theta_2$}}" + row,
            rf"\cmidrule(lr){{2-{k+1}}}\cmidrule(l){{{k+2}-{2*k+1}}}",
            "Method & " + " & ".join(labels * 2) + row, r"\midrule"
        ]
        for name in METHODS:
            means = [point[n][name]["mean_abs"][j] * 100 for j in range(2) for n in sizes]
            deviations = [
                point[n][name]["sd_abs"][j] * 100 if point[n][name]["sd_abs"] is not None else None
                for j in range(2) for n in sizes
            ]
            t.append(names.get(name, name) + " & " + " & ".join(map(significant, means)) + row)
            t.append(" & " + " & ".join("(" + significant(x) + ")" for x in deviations) + row)
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tables.append("\n".join(t))
        t = [
            r"\begin{table}[t]", r"\centering",
            r"\caption{Coverage (\%) and mean confidence-set length (multiplied by 100) over " +
            str(repetitions)+r" paired replications. Disconnected EL components " +
            r"contribute their total length.}",
            r"\label{tab:sim-v}", r"\small\linespread{1}\selectfont",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}l" + "rr" * k + r"@{}}", r"\toprule",
            " & " + " & ".join(
                r"\multicolumn{2}{c}{$n_\P = " + f"{n:,}".replace(",", "{,}") + "$}"
                for n in sizes
            ) + row,
            "Method & " + " & ".join(["Cov. & Len."] * k) + row
        ]
        for j in range(2):
            t += [r"\midrule",
                  rf"\multicolumn{{{2*k+1}}}{{c}}{{$\theta_{j+1}$}}" + row, r"\midrule"]
            for name in METHODS:
                t.append(names.get(name, name) + " & " + " & ".join(
                    f'{point[n][name]["coverage"][j]:.1f} & {100 * point[n][name]["ci_length"][j]:.1f}'
                    for n in sizes
                ) + row)
        t += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tables.append("\n".join(t))
    if summary["joint"]:
        group = summary["joint"]
        repetitions = group[METHODS[0]]["replications"]
        t = [r"\begin{table}[t]", r"\centering",
             r"\caption{Joint inference for seven target quantiles over " + str(repetitions) +
             r" paired replications. Time is in seconds, RMSE is multiplied by 100, and joint coverage is in percent.}",
             r"\label{tab:sim-joint}", r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             " & " + " & ".join(names.get(m, m) for m in METHODS) + row, r"\midrule"]
        for title, key, scale, spread in [
            ("Time (s)", "time_mean", 1, "time_sd"),
            ("RMSE", "mean_rms", 100, "sd_rms"),
            ("Joint coverage", "coverage", 1, None),
        ]:
            digits = 1 if spread is None else 2
            values = [scale * group[m][key] for m in METHODS]
            t.append(title + " & " + " & ".join(f"{x:.{digits}f}" for x in values) + row)
            if spread:
                values = [group[m][spread] for m in METHODS]
                t.append(" & " + " & ".join(
                    "--" if x is None else f"({scale * x:.2f})" for x in values) + row)
        t.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        tables.append("\n".join(t))
    return "\n\n".join(tables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help="Print installed software and source versions")
    verify = commands.add_parser("check", help="Compare six archived reference cases")
    verify.add_argument("--workers", type=int, default=2)
    run = commands.add_parser("run", help="Stream paired replications as JSONL")
    run.add_argument("--experiment", choices=("pointwise", "joint", "all"), default="all")
    run.add_argument("--replications", type=int, default=REPS)
    run.add_argument("--workers", type=int, default=24)
    run.add_argument("--sizes", type=int, nargs="+", choices=SIZES, default=list(SIZES))
    summary = commands.add_parser("summarize", help="Summarize records without refitting")
    summary.add_argument("files", nargs="*")
    summary.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    if args.command == "environment":
        emit("environment", environment())
    elif args.command == "check":
        if args.workers < 1:
            parser.error("Use a positive worker count")
        check(args.workers)
    elif args.command == "run":
        if not 1 <= args.replications <= REPS or args.workers < 1:
            parser.error("Use 1..500 replications and a positive worker count")
        emit("metadata", dict(
            environment=environment(), master_seed=MASTER,
            replications=args.replications, workers=args.workers,
            r_threads=R_THREADS, multiplier_draws=DRAW,
            seeds=seeds()[:args.replications], sizes=args.sizes, taus=JOINT_TAU,
            truth_pointwise=PRIMARY_TRUTH, truth_joint=TRUTH,
            ratio_policy=BASE_R, dependence_policy=BASE_RHO, neural=CONFIG["neural"]
        ))
        experiments = ("pointwise", "joint") if args.experiment == "all" else (args.experiment,)
        records = []
        total = args.replications * (
            (len(args.sizes) if "pointwise" in experiments else 0)
            + (1 if "joint" in experiments else 0)
        )
        for experiment in experiments:
            jobs = [
                (experiment, n, rep, int(seed))
                for rep, seed in enumerate(seeds()[:args.replications])
                for n in (args.sizes if experiment == "pointwise" else [10000])
            ]
            for record in execute(jobs, args.workers):
                records.append(record)
                emit("replication", record)
                if len(records) % 25 == 0 or len(records) == total:
                    print(f"Completed {len(records)}/{total} paired datasets", file=sys.stderr)
        emit("summary", summarize(records))
    else:
        records = []
        streams = [open(path, encoding="utf-8")
                   for path in args.files] if args.files else [sys.stdin]
        try:
            for stream in streams:
                for line in stream:
                    if line.strip():
                        item = json.loads(line)
                        if item.get("type") == "replication":
                            records.append(item)
        finally:
            for stream in streams:
                if stream is not sys.stdin:
                    stream.close()
        if not records:
            parser.error("No replication records found")
        output = summarize(records)
        print(latex_tables(output)) if args.latex else emit("summary", output)


if __name__ == "__main__":
    main()
