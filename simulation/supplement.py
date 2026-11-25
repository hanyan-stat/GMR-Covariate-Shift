"""Reproduce the three supplementary simulation tables from paired Monte Carlo records."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import chi2

from . import simulate as sim, inference as inf, nuisance as nu
from .inference import one_sample

TARGET_SIZES = (2000, 5000, 20000)
BONFERRONI = float(chi2.ppf(1 - 0.05 / 7, 1))


def extended_sample(seed, nq):
    obs = sim.generate(seed, 10000)
    if nq <= len(obs.xq):
        return nu.Observed(obs.xp, obs.yp, obs.xq[:nq].copy())
    rng = np.random.default_rng([seed, 20261125, 20000])
    remainder = nq - len(obs.xq)
    x = rng.standard_normal((remainder, 10))
    x[:, 0] += np.where(rng.random(remainder) < 1 / 3, 5.0, -4.0)
    return nu.Observed(obs.xp, obs.yp, np.vstack([obs.xq, x]))


@njit
def plugin_stat(F, k, y, true, cut):
    n = len(y)
    m = F.shape[0]
    loss = np.empty(n)
    minval = np.inf
    best = -1
    distance = np.empty(n)
    for a in range(n):
        v = F[:, a] ** k - 0.5
        distance[a] = abs(v.mean())
    nearest = np.argmin(distance)
    for j in range(n + 1):
        a = nearest if j == 0 else j - 1
        delta = distance[a]
        bound = m * delta * delta / (0.25 + 0.5 * delta)
        if bound > minval + cut + 1e-7:
            loss[a] = np.inf
        else:
            value, status = one_sample(F[:, a] ** k - 0.5, minval + cut)
            if status == 2:
                raise ValueError("Plug-DW solver failure")
            loss[a] = value if status == 0 else np.inf
            if loss[a] < minval or (loss[a] == minval and a < best):
                minval = loss[a]
                best = a
    if not np.isfinite(minval) or best < 0:
        raise ValueError("No feasible Plug-DW point")
    a = np.searchsorted(y, true, side="right") - 1
    f = F[:, a] if a >= 0 else np.zeros(m)
    value, status = one_sample(f**k - 0.5)
    if status == 2:
        raise ValueError("Plug-DW truth-statistic failure")
    width = 0.0
    for a in range(n - 1):
        if loss[a] - minval <= cut:
            width += y[a + 1] - y[a]
    return y[best], status == 0 and value - minval <= cut, width


def score_details(y, r, F, estimates):
    n = len(y)
    m = len(F) - n
    effect = []
    wald = []
    for k, e, true in zip((1, 10), estimates, sim.PRIMARY_TRUTH):
        j = int(np.searchsorted(y, e, side="right") - 1)
        f = F[:, j] if j >= 0 else np.zeros(n + m)
        f = np.clip(f, 0, 1)
        a = r * k * f[:n] ** (k - 1)
        p = a * ((y <= e) - f[:n])
        q = f[n:] ** k - 0.5
        vp = p.var(ddof=1) / n
        vq = q.var(ddof=1) / m
        jac = inf.density(y, a, e)
        val = sim.wald(e, p, q, jac, true)
        effect.append(vq / (vp + vq))
        wald.append((e, bool(val["covered"]), val["ci_hi"] - val["ci_lo"]))
    return effect, wald


def joint_lengths(y, r, F, seed):
    n = len(y)
    m = len(F) - n
    sim.N = n
    sim.M = m
    sim.TAU = sim.JOINT_TAU
    ids, minimum = sim.fast_minimum(r, F)
    est = y[ids]
    fe = F[:, ids]
    p = r[:, None] * ((y[:, None] <= est) - fe[:n])
    q = fe[n:] - sim.TAU
    normal = np.random.default_rng([seed, 92651]).standard_normal((sim.DRAW, 7))
    cutoff = sim.gaussian_cutoff(p, q, normal)
    where = np.searchsorted(y, sim.TRUTH, side="right") - 1
    f = F[:, np.maximum(where, 0)].copy()
    f[:, where < 0] = 0
    stat = (
        np.array(
            [
                inf.checked(inf.two_sample(r * ((y <= t) - f[:n, j]), f[n:, j] - sim.TAU[j]))
                for j, t in enumerate(sim.TRUTH)
            ]
        )
        - minimum
    )
    raw, status = inf.family_step_profile(
        F, r, sim.TAU, max(cutoff, BONFERRONI) + max(minimum) + 1.0e-5
    )
    assert np.all(status != 2)
    report = {}
    for name, crit in [("Multiplier", cutoff), ("Bonferroni", BONFERRONI)]:
        mask = (status == 0) & (raw - minimum[:, None] <= crit)
        assert not mask[:, -1].any()
        width = mask[:, :-1] @ np.diff(y)
        report[name] = dict(
            joint=bool(np.all(stat <= crit)),
            avg=float(width.mean()),
            maximum=float(width.max()),
            cutoff=crit,
        )
    return report


def estimated_pointwise(y, r, cdf, obs, seed):
    sim.N, sim.M = len(y), len(obs.xq)
    fit = sim.estimated_primary(obs, seed, (y, r, cdf))
    share, wal = score_details(y, r, cdf, fit["estimate"])
    return (
        dict(
            estimate=fit["estimate"],
            covered=fit["covered"],
            length=fit["ci_length"],
            target_share=share,
        ),
        wal,
    )


def estimated(rep, seed):
    result = {}
    for n in (2000, 10000):
        obs = sim.generate(seed, n) if n == 2000 or rep >= 200 else extended_sample(seed, 20000)
        order, y, F = sim.donor_cdf(obs, seed)
        sizes = (1000,) if n == 2000 else ((5000, 2000, 20000) if rep < 200 else (5000,))
        for m in sizes:
            xq = obs.xq[:m].copy()
            sub = nu.Observed(obs.xp, obs.yp, xq)
            r = nu.fit_ratio(obs.xp, xq, sim.CONFIG["neural"], seed + 1, sim.BASE_R).predict(
                obs.xp
            )[order]
            f = F[: n + m]
            result[(str(n) + "_" + str(m))], wald = estimated_pointwise(y, r, f, sub, seed)
            if m * 2 == n:
                plugin = [
                    plugin_stat(f[n:], k, y, t, sim.POINT)
                    for k, t in zip((1, 10), sim.PRIMARY_TRUTH)
                ]
                result[str(n) + "_ablation"] = dict(plugin=plugin, wald=wald)
            if m == 5000:
                result["joint"] = joint_lengths(y, r, f, seed)
        del F, f
    return result


def oracle(seed):
    obs = extended_sample(seed, 20000)
    out = {}
    for m in TARGET_SIZES:
        sim.N = len(obs.yp)
        sim.M = m
        xq = obs.xq[:m]
        sub = nu.Observed(obs.xp, obs.yp, xq)
        a = sim.oracle_primary(sub, seed)
        muP = sim.mean_response(obs.xp)
        muQ = sim.mean_response(xq)
        r = sim.true_ratio(obs.xp)
        v = []
        for k, t in zip((1, 10), a["estimate"]):
            pp = sim.ndtr(t - muP)
            qq = sim.ndtr(t - muQ)
            p = r * k * pp ** (k - 1) * ((obs.yp <= t) - pp)
            q = qq**k
            vp = p.var(ddof=1) / len(p)
            vq = q.var(ddof=1) / len(q)
            v.append(vq / (vp + vq))
        out[str(m)] = dict(
            estimate=a["estimate"],
            covered=a["covered"],
            length=a["ci_length"],
            target_share=v,
            components=a["components"],
        )
    return out


def localized(seed):
    obs = extended_sample(seed, 20000)
    out = {}
    for m in TARGET_SIZES:
        sim.N, sim.M = len(obs.yp), m
        sim.TAU = sim.POINT_TAU
        sub = nu.Observed(obs.xp, obs.yp, obs.xq[:m])
        fit = sim.localized_primary(sub, seed)
        intervals = fit["wald_points"]
        out[str(m)] = dict(
            estimate=fit["estimate"],
            covered=fit["covered"],
            length=[w["ci_hi"] - w["ci_lo"] for w in intervals],
            target_share=[w["target_variance_share"] for w in intervals],
        )
    return out


def replicate(item):
    rep, seed = item
    row = dict(rep=rep, seed=seed, OEL=estimated(rep, seed))
    if rep < 200:
        row.update(Oracle=oracle(seed), LDML=localized(seed))
    return nu.json_compatible(row)


def report(rows):
    """All table columns use the paper's display units; interval lengths are multiplied by 100."""
    rows = sorted(rows, key=lambda r: r["rep"])
    if not rows or [r["rep"] for r in rows] != list(range(len(rows))):
        raise ValueError("Expected consecutive paired replications starting at zero")
    if [r["seed"] for r in rows] != list(sim.seeds()[: len(rows)]):
        raise ValueError("Seed sequence differs from the paired design")

    def show(values):
        print("\t".join(str(v) for v in values))

    def summary(values, truth):
        values = np.asarray(values, dtype=float)
        error = values[:, 0] - truth
        sd = "--" if len(values) < 2 else sim.significant(100 * np.std(error, ddof=1))
        return [
            sim.significant(100 * error.mean()),
            sd,
            f"{100*values[:,1].mean():.1f}",
            sim.significant(100 * values[:, 2].mean()),
        ]

    def point(row, j):
        return [row[k][j] for k in ("estimate", "covered", "length")]

    print(f"Source correction and calibration ({len(rows)} replications)")
    show(["Parameter", "nP", "Method", "Bias", "SD", "Coverage (%)", "CI length"])
    for j, truth in enumerate(sim.PRIMARY_TRUTH):
        for n in (2000, 10000):
            for name in ("Plug-DW", "OEL-DW (EL)", "OEL-DW (Wald)"):
                if name == "OEL-DW (EL)":
                    values = [point(r["OEL"][f"{n}_{n//2}"], j) for r in rows]
                else:
                    kind = "plugin" if name == "Plug-DW" else "wald"
                    values = [r["OEL"][f"{n}_ablation"][kind][j] for r in rows]
                show([f"theta{j+1}", n, name, *summary(values, truth)])
    ratios = rows[:200]
    print(f"\nTarget sample size ({len(ratios)} paired replications; nP=10000)")
    show(["Parameter", "nQ", "Method", "SD", "Coverage (%)", "CI length", "Target share (%)"])
    for j, truth in enumerate(sim.PRIMARY_TRUTH):
        for m in TARGET_SIZES:
            for name, group, key in [
                ("OEL-Oracle", "Oracle", str(m)),
                ("OEL-DW", "OEL", f"10000_{m}"),
                ("LDML", "LDML", str(m)),
            ]:
                cells = [r[group][key] for r in ratios]
                share = np.mean([c["target_share"][j] for c in cells])
                show(
                    [
                        f"theta{j+1}",
                        m,
                        name,
                        *summary([point(c, j) for c in cells], truth)[1:],
                        f"{100*share:.1f}",
                    ]
                )
    print(f"\nSeven-quantile joint calibration ({len(rows)} replications)")
    show(
        [
            "Calibration",
            "Critical value",
            "Joint coverage (%)",
            "Mean CI length",
            "Maximum CI length",
        ]
    )
    for kind in ("Multiplier", "Bonferroni"):
        cells = [r["OEL"]["joint"][kind] for r in rows]
        cutoff, coverage, avg, maximum = np.mean(
            [[c[k] for k in ("cutoff", "joint", "avg", "maximum")] for c in cells], axis=0
        )
        show(
            [
                kind,
                sim.significant(cutoff),
                f"{100*coverage:.1f}",
                sim.significant(100 * avg),
                sim.significant(100 * maximum),
            ]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("run")
    run.add_argument("--replications", type=int, default=500)
    run.add_argument("--workers", type=int, default=1)
    summary = commands.add_parser("tables")
    summary.add_argument("records", type=Path)
    args = parser.parse_args()
    if args.action == "tables":
        with args.records.open(encoding="utf-8") as stream:
            report([json.loads(s) for s in stream if s.strip()])
        return
    if not 1 <= args.replications <= 500 or args.workers < 1:
        parser.error("Use 1..500 replications and a positive number of workers")
    jobs = enumerate(map(int, sim.seeds()[: args.replications]))
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"), initializer=nu.initialize
    ) as pool:
        for row in pool.map(replicate, jobs):
            print(json.dumps(row, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
