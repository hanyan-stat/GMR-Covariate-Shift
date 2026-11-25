"""Empirical likelihood, safe profile screens, and Wald intervals.

EL statuses: 0 exact, 1 infeasible, 2 numerical failure, 3 certified rejection.
"""
import collections
import heapq
import numpy as np
from numba import njit
from scipy.optimize import brentq
from scipy.stats import chi2
CUTOFF = float(chi2.ppf(.95, 1))
MIN_TOL, THETA_TOL, MAX_NODES = 1e-5, 2e-7, 50000


@njit
def _terms(a, b, x, y):
    val = g0 = g1 = h00 = h01 = h11 = 0.0
    for i in range(len(a)):
        d = 1 + x * a[i] + y * b[i]
        if d <= 0:
            return -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0
        val += np.log(d)
        va, vb = a[i] / d, b[i] / d
        g0 += va
        g1 += vb
        h00 += va * va
        h01 += va * vb
        h11 += vb * vb
    return val, g0, g1, h00, h01, h11


@njit
def _two_newton(p, q, stop_above=np.inf):
    """Return statistic, status (0 success, 1 infeasible, 2 solver failure)."""
    n, m = len(p), len(q)
    if p.min() + q.min() >= 0 or p.max() + q.max() <= 0:
        if abs(p.mean() + q.mean()) < 1e-13:
            return 0.0, 0
        return np.inf, 1
    shift = p.mean()
    scale = np.sqrt(p.var() + q.var() + (p.mean() + q.mean()) ** 2)
    if scale < 1e-14:
        return np.inf, 1
    a = np.empty(n + m)
    b = np.empty(n + m)
    a[:n] = (p - shift) / scale * (n + m) / n
    a[n:] = (q + shift) / scale * (n + m) / m
    b[:n] = (n + m) / n
    b[n:] = -(n + m) / m
    x = y = 0.0
    for it in range(100):
        val, g0, g1, h00, h01, h11 = _terms(a, b, x, y)
        # Every feasible dual iterate is a lower bound on the maximized EL
        # statistic. Status 3 certifies rejection; it is NOT an exact statistic.
        if 2 * val > stop_above + 1e-7:
            return 2 * val, 3
        det = h00 * h11 - h01 * h01
        if det <= 1e-14 * h00 * h11 or not np.isfinite(det):
            return np.nan, 2
        dx = (h11 * g0 - h01 * g1) / det
        dy = (h00 * g1 - h01 * g0) / det
        decrement = dx * g0 + dy * g1
        if 0 <= decrement < 2e-12:
            return max(0.0, 2 * val), 0
        if decrement < 0:
            return np.nan, 2
        step = 1.0
        for i in range(n + m):
            change = dx * a[i] + dy * b[i]
            if change < 0:
                step = min(step, -0.99 * (1 + x * a[i] + y * b[i]) / change)
        accepted = False
        for _ in range(60):
            newval = _terms(a, b, x + step * dx, y + step * dy)[0]
            if np.isfinite(newval) and newval >= val + 1e-4 * step * decrement - 1e-12:
                x += step * dx
                y += step * dy
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return np.nan, 2
    return np.nan, 2


@njit
def _group_normalizer(g, lam):
    """Solve sum 1/(alpha+lambda*g)=1 without assuming alpha positive.

    Reparameterization u=alpha+min(lambda*g) gives the fixed bracket [1,n].
    It stays well conditioned when the mass-contrast Hessian approaches singularity.
    """
    minimum = np.min(lam * g)
    offset = lam * g - minimum
    low, high = 1.0, float(len(g))
    u = (low + high) / 2
    for _ in range(100):
        weights = 1 / (u + offset)
        residual = np.sum(weights) - 1
        if abs(residual) < 2e-13:
            break
        if residual > 0:
            low = u
        else:
            high = u
        proposed = u + residual / np.sum(weights * weights)
        if proposed <= low or proposed >= high:
            proposed = (low + high) / 2
        u = proposed
    den = u + offset
    weights = 1 / den
    squared = weights * weights
    center = np.sum(squared * g) / np.sum(squared)
    derivative = -np.sum(squared * (g - center) ** 2)
    value = np.sum(np.log(den / len(g)))
    return np.sum(weights * g), derivative, value


@njit
def _profile_terms(p, q, lam):
    a, da, va = _group_normalizer(p, lam)
    b, db, vb = _group_normalizer(q, lam)
    return a + b, da + db, va + vb


@njit
def _two_profile(p, q):
    scale = np.sqrt(p.var() + q.var() + (p.mean() + q.mean()) ** 2)
    if scale < 1e-15:
        return np.nan, 2
    # Avoid artificial cancellation of opposite group means at a hull boundary.
    pp = p / scale
    qq = q / scale
    initial, derivative, _ = _profile_terms(pp, qq, 0.0)
    if abs(initial) < 1e-12:
        return 0.0, 0
    direction = 1.0 if initial > 0 else -1.0
    endpoint = direction * max(1.0, abs(initial) / max(-derivative, 1e-15))
    for _ in range(100):
        value, _, _ = _profile_terms(pp, qq, endpoint)
        if value * initial <= 0:
            break
        endpoint *= 2
    else:
        return np.nan, 2
    low, high = min(0.0, endpoint), max(0.0, endpoint)
    lam = (low + high) / 2
    for _ in range(120):
        score, derivative, objective = _profile_terms(pp, qq, lam)
        if abs(score) < 2e-10 and score * score / max(-derivative, 1e-300) < 1e-10:
            return max(0.0, 2 * objective), 0
        if score > 0:
            low = lam
        else:
            high = lam
        proposed = lam - score / derivative if derivative < 0 else np.nan
        if not np.isfinite(proposed) or proposed <= low or proposed >= high:
            proposed = (low + high) / 2
        if abs(proposed - lam) <= 1e-13 * (1 + abs(lam)):
            if score * score / max(-derivative, 1e-300) < 1e-8:
                return max(0.0, 2 * objective), 0
            return np.nan, 2
        lam = proposed
    return np.nan, 2


@njit
def two_sample(p, q):
    out = _two_newton(p, q)
    if out[1] != 2:
        return out
    return _two_profile(p, q)


@njit
def two_sample_classify(p, q, largest_cutoff):
    out = _two_newton(p, q, largest_cutoff)
    return _two_profile(p, q) if out[1] == 2 else out


@njit
def one_sample(g, stop_above=np.inf):
    if abs(g.mean()) < 1e-13:
        return 0.0, 0
    if g.min() >= 0 or g.max() <= 0:
        return np.inf, 1
    low = -1 / g.max()
    high = -1 / g.min()
    lam = 0.0
    for _ in range(100):
        den = 1 + lam * g
        if np.isfinite(stop_above):
            val = 2 * np.sum(np.log(den))
            if val > stop_above + 1e-7:
                return val, 3
        grad = np.sum(g / den)
        if abs(grad) / len(g) < 1e-10:
            return max(0.0, 2 * np.sum(np.log(den))), 0
        if grad > 0:
            low = lam
        else:
            high = lam
        proposal = lam + grad / np.sum((g / den) ** 2)
        if proposal <= low or proposal >= high:
            proposal = (low + high) / 2
        if abs(proposal - lam) < 1e-13:
            return max(0.0, 2 * np.sum(np.log(1 + proposal * g))), 0
        lam = proposal
    return np.nan, 2


@njit
def dual_lower_bound(total_mean, variance_of_mean, negative_extreme, total_size):
    """Rigorous log-dual bound; see EL_SCREEN_PROOF.md for derivation.

    g_i=(p_i-pbar)/n+S/N and g_j=(q_j-qbar)/m+S/N.
    A=abs(S), B=sum(g*g), M=max(0,-min(sign(S)*g)).
    log(1+t)>=t-t*t/(2*(1-u)) for t>=-u, u<1 gives
    R >= 4*A*A/(sqrt(B+2*A*M)+sqrt(B))**2.
    Conservative inflation and a strict margin protect floating-point tests.
    """
    a = abs(total_mean)
    b = variance_of_mean + total_mean * total_mean / total_size
    m = max(0.0, negative_extreme)
    if b <= 0.0 or not np.isfinite(a + b + m):
        return 0.0
    b *= 1.0 + 1e-10
    m *= 1.0 + 1e-10
    den = np.sqrt(b + 2 * a * m) + np.sqrt(b)
    return 4 * a * a / (den * den)


@njit
def family_step_profile(cdf, sorted_ratio, taus, largest_cutoff):
    n = len(sorted_ratio)
    m = len(cdf) - n
    total = n + m
    stats = np.empty((len(taus), n))
    statuses = np.empty((len(taus), n), dtype=np.int64)
    for k in range(n):
        indicator = np.zeros(n)
        indicator[: k + 1] = 1.0
        p = sorted_ratio * (indicator - cdf[:n, k])
        f = cdf[n:, k]
        pm = p.mean()
        fm = f.mean()
        pmin = p.min()
        pmax = p.max()
        fmin = f.min()
        fmax = f.max()
        vm = p.var() / n + f.var() / m
        lower = min((pmin - pm) / n, (fmin - fm) / m)
        upper = max((pmax - pm) / n, (fmax - fm) / m)
        for j in range(len(taus)):
            s = pm + fm - taus[j]
            # Preserve the original exact solver's hull handling. Subtraction
            # is ordered as in q=f-tau to avoid endpoint cancellation changes.
            if pmin + (fmin - taus[j]) >= 0.0 or pmax + (fmax - taus[j]) <= 0.0:
                stats[j, k], statuses[j, k] = two_sample_classify(
                    p, f - taus[j], largest_cutoff
                )
                continue
            extreme = -s / total - lower if s >= 0.0 else s / total + upper
            bound = dual_lower_bound(s, vm, extreme, total)
            if bound > largest_cutoff + 1e-6 * (1.0 + largest_cutoff):
                stats[j, k] = bound
                statuses[j, k] = 3
            else:
                stats[j, k], statuses[j, k] = two_sample_classify(
                    p, f - taus[j], largest_cutoff
                )
    return stats, statuses


@njit
def classify(p, q, cutoff):
    pm = p.mean()
    qm = q.mean()
    s = pm + qm
    n = len(p)
    m = len(q)
    total = n + m
    if p.min() + q.min() >= 0.0 or p.max() + q.max() <= 0.0:
        return two_sample_classify(p, q, cutoff)
    low = min((p.min() - pm) / n, (q.min() - qm) / m)
    high = max((p.max() - pm) / n, (q.max() - qm) / m)
    extreme = -s / total - low if s >= 0.0 else s / total + high
    bound = dual_lower_bound(s, p.var() / n + q.var() / m, extreme, total)
    if bound > cutoff + 1e-6 * (1 + cutoff):
        return bound, 3
    return two_sample_classify(p, q, cutoff)


@njit
def maximum_profile(cdf, sorted_ratio, cutoff):
    n = len(sorted_ratio)
    stats = np.empty(n)
    statuses = np.empty(n, np.int64)
    moments = np.empty(n)
    for k in range(n):
        fp = np.minimum(1.0, np.maximum(0.0, cdf[:n, k]))
        fq = np.minimum(1.0, np.maximum(0.0, cdf[n:, k]))
        indicator = np.zeros(n)
        indicator[: k + 1] = 1.0
        p = sorted_ratio * 10 * fp**9 * (indicator - fp)
        q = fq**10 - 0.5
        moments[k] = p.mean() + q.mean()
        stats[k], statuses[k] = classify(p, q, cutoff)
    return stats, statuses, moments


def checked(out):
    value, status = out
    if status == 2 or np.isnan(value):
        raise ArithmeticError("EL numerical non-convergence")
    return float(value)


def base_result(est, lo, hi, true, statistic=None, kind="EL"):
    return dict(
        est=float(est),
        ci_lo=float(lo),
        ci_hi=float(hi),
        ci_type=kind,
        R_at_truth=statistic,
        covered=bool(statistic <= CUTOFF) if statistic is not None else bool(lo <= true <= hi),
        hull_covered=bool(lo <= true <= hi),
        converged=True,
        ci_unbounded=False,
    )


def density(y, r, theta):
    scale = min(np.std(y, ddof=1), (np.quantile(y, 0.75) - np.quantile(y, 0.25)) / 1.34)
    h = max(0.9 * scale * len(y) ** (-0.2), 1e-5)
    return float(np.mean(r * np.exp(-0.5 * ((y - theta) / h) ** 2) / np.sqrt(2 * np.pi) / h))


def wald(est, p, q, derivative, true):
    source_variance = np.var(p, ddof=1) / len(p)
    target_variance = np.var(q, ddof=1) / len(q)
    se2 = source_variance + target_variance
    se = np.sqrt(se2) / abs(derivative)
    if not np.isfinite(se) or se <= 0:
        raise ArithmeticError("nonfinite/nonpositive Wald standard error")
    out = base_result(
        est, est - 1.95996398454 * se, est + 1.95996398454 * se, true, kind="Wald"
    )
    out["se"] = float(se)
    out["target_variance_share"] = float(target_variance / se2)
    return out


def certified_minimum(scan):
    threshold = 10.0
    for attempt in range(8):
        values, status = scan(threshold)
        if np.any(status == 2):
            raise ArithmeticError("EL numerical nonconvergence")
        exact = np.where(status == 0, values, np.inf)
        k = np.argmin(exact, axis=-1)
        minimum = np.take_along_axis(exact, k[..., None], axis=-1)[..., 0]
        if np.all(np.isfinite(minimum)) and np.all(minimum <= threshold):
            return values, status, k, minimum, threshold
        threshold *= 4
    raise ArithmeticError("no certified finite global profile minimum")


@njit
def dual(p, q):
    """Same two-dimensional Newton dual as src/el.py, retaining multipliers."""
    n, m = len(p), len(q)
    total = n + m
    if p.min() + q.min() >= 0 or p.max() + q.max() <= 0:
        return (
            (0.0, 0, 0.0, 0.0) if abs(p.mean() + q.mean()) < 1e-13 else (np.inf, 1, 0.0, 0.0)
        )
    shift = p.mean()
    scale = np.sqrt(p.var() + q.var() + (p.mean() + q.mean()) ** 2)
    if scale < 1e-14:
        return np.nan, 2, 0.0, 0.0
    a = np.empty(total)
    b = np.empty(total)
    a[:n] = (p - shift) / scale * total / n
    a[n:] = (q + shift) / scale * total / m
    b[:n] = total / n
    b[n:] = -total / m
    x = y = 0.0
    for it in range(100):
        val, g0, g1, h00, h01, h11 = _terms(a, b, x, y)
        det = h00 * h11 - h01 * h01
        if det <= 1e-14 * h00 * h11 or not np.isfinite(det):
            return np.nan, 2, 0.0, 0.0
        dx = (h11 * g0 - h01 * g1) / det
        dy = (h00 * g1 - h01 * g0) / det
        decrement = dx * g0 + dy * g1
        if 0 <= decrement < 2e-12:
            return max(0.0, 2 * val), 0, x / scale, y - x * shift / scale
        if decrement < 0:
            return np.nan, 2, 0.0, 0.0
        step = 1.0
        for i in range(total):
            change = dx * a[i] + dy * b[i]
            if change < 0:
                step = min(step, -0.99 * (1 + x * a[i] + y * b[i]) / change)
        accepted = False
        for unused in range(60):
            new = _terms(a, b, x + step * dx, y + step * dy)[0]
            if np.isfinite(new) and new >= val + 1e-4 * step * decrement - 1e-12:
                x += step * dx
                y += step * dy
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return np.nan, 2, 0.0, 0.0
    return np.nan, 2, 0.0, 0.0


class Profile:
    def __init__(self, cdf, power, tau):
        self.cdf = cdf
        self.obs = cdf.obs
        self.n = cdf.n
        self.m = cdf.m
        self.r = cdf.r
        self.k = power
        self.c = tau if power == 1 else 0.5
        self.ys = np.unique(np.sort(self.obs.yp))
        self.memo = collections.OrderedDict()
        self.evaluations = 0
        self.nodes = 0
        width = max(2.0, 3 * np.std(self.obs.yp))
        self.lo = float(self.ys[0] - width)
        self.hi = float(self.ys[-1] + width)
        for i in range(30):
            if (
                np.max(self.cdf.cdf(self.lo)[self.n:] ** self.k) < self.c and
                np.min(self.cdf.cdf(self.hi)[self.n :] ** self.k) > self.c
            ):
                break
            w = self.hi - self.lo
            self.lo -= w
            self.hi += w
        else:
            raise ArithmeticError("oracle finite domain certification failed")
        self.best = np.inf
        self.est = None
        self.minimum_lower = 0.0
        self.minimum_upper = np.inf
        self.optimizing = True

    def groups(self, t):
        f = self.cdf.cdf(t)
        fp = f[: self.n]
        fq = f[self.n:]
        return (
            self.r * self.k * fp ** (self.k - 1) * ((self.obs.yp <= t) - fp),
            fq**self.k - self.c,
        )

    def at(self, t):
        t = float(t)
        if t in self.memo:
            self.memo.move_to_end(t)
            return self.memo[t]
        p, q = self.groups(t)
        value, status, lam, nu = dual(p, q)
        if status == 2:
            value = checked(two_sample(p, q))
            lam = nu = 0.0
        out = (float(value), float(lam), float(nu), p, q)
        self.memo[t] = out
        self.evaluations += 1
        if len(self.memo) > 24:
            self.memo.popitem(last=False)
        if self.optimizing and value < self.best:
            self.best = float(value)
            self.est = t
        return out

    def bounds(self, a, b):
        fa = self.cdf.cdf(a)
        fb = self.cdf.cdf(b)
        pa = fa[: self.n]
        pb = fb[: self.n]
        k = self.k
        ha = k * pa ** (k - 1) * (1 - pa)
        hb = k * pb ** (k - 1) * (1 - pb)
        top = np.maximum(ha, hb)
        if k > 1:
            peak = (k - 1) / k
            top = np.where((pa <= peak) & (pb >= peak), k * peak ** (k - 1) * (1 - peak), top)
        lower = -self.r * k * pb**k
        upper = -self.r * k * pa**k
        one = self.obs.yp <= a
        cross = (self.obs.yp > a) & (self.obs.yp <= b)
        lower[one] = self.r[one] * np.minimum(ha, hb)[one]
        upper[one] = self.r[one] * top[one]
        upper[cross] = self.r[cross] * top[cross]
        return lower, upper, fa[self.n:] ** k - self.c, fb[self.n:] ** k - self.c

    def interval(self, a, b, upper=False):
        pl, ph, ql, qh = self.bounds(a, b)
        slo = pl.mean() + ql.mean()
        shi = ph.mean() + qh.mean()
        distance = max(0.0, slo, -shi)
        denominator = (ph.max() - pl.min()) ** 2 / self.n + (qh.max() - ql.min()) ** 2 / self.m
        lower = 4 * distance * distance / max(denominator, 1e-300)
        mid = float(a + (b - a) / 2)
        val, lam, nu, p, q = self.at(mid)
        total = self.n + self.m
        ep = pl if lam >= 0 else ph
        eq = ql if lam >= 0 else qh
        dp = 1 + total / self.n * (lam * ep + nu)
        dq = 1 + total / self.m * (lam * eq - nu)
        if dp.min() > 0 and dq.min() > 0:
            lower = max(lower, 2 * (np.log(dp).sum() + np.log(dq).sum()))
        lower = max(0.0, float(lower) - 1e-7 * (1 + abs(lower)))
        if not upper or not np.isfinite(val):
            return lower, np.inf
        dp = 1 + total / self.n * (lam * p + nu)
        dq = 1 + total / self.m * (lam * q - nu)
        if dp.min() <= 0 or dq.min() <= 0:
            return lower, np.inf
        wp = (1 / self.n) / dp
        wq = (1 / self.m) / dq
        wp /= wp.sum()
        wq /= wq.sum()
        hp = p - wp @ p
        hq = q - wq @ q
        sp0 = wp @ pl + wq @ ql
        sp1 = wp @ ph + wq @ qh
        ap = wp * hp
        aq = wq * hq
        b0 = (
            np.maximum(ap, 0) @ pl +
            np.minimum(ap, 0) @ ph +
            np.maximum(aq, 0) @ ql +
            np.minimum(aq, 0) @ qh
        )
        b1 = (
            np.maximum(ap, 0) @ ph +
            np.minimum(ap, 0) @ pl +
            np.maximum(aq, 0) @ qh +
            np.minimum(aq, 0) @ ql
        )
        if b0 <= 0 <= b1:
            return lower, np.inf
        cs = [-s / h for s in [sp0, sp1] for h in [b0, b1]]
        cl, ch = min(cs), max(cs)
        costs = []
        base = -2 * (np.log(self.n * wp).sum() + np.log(self.m * wq).sum())
        for c in [cl, ch]:
            ep = 1 + c * hp
            eq = 1 + c * hq
            if ep.min() <= 0 or eq.min() <= 0:
                return lower, np.inf
            costs.append(base - 2 * (np.log(ep).sum() + np.log(eq).sum()))
        high = max(costs)
        return lower, max(0.0, float(high) + 1e-7 * (1 + abs(high)))

    def split(self, a, b):
        first = np.searchsorted(self.ys, a, side="right")
        last = np.searchsorted(self.ys, b, side="left")
        c = float(self.ys[(first + last - 1) // 2]) if last > first else float(a + (b - a) / 2)
        left = float(np.nextafter(c, -np.inf))
        if c <= a or left < a or c >= b:
            return []
        return [(a, left), (c, b)]

    def minimize(self):
        equation = lambda t: sum(z.mean() for z in self.groups(t))
        left, right = equation(self.lo), equation(self.hi)
        self.at(self.lo)
        self.at(self.hi)
        self.at(self.pilot)
        if left * right < 0:
            root = float(brentq(equation, self.lo, self.hi, xtol=1e-11))
            self.at(root)
        else:
            root = self.pilot
        j = np.searchsorted(self.ys, root)
        for y in self.ys[max(0, j - 2): min(len(self.ys), j + 2)]:
            candidate = float(y)
            below = float(np.nextafter(y, -np.inf))
            if self.lo <= candidate <= self.hi:
                self.at(candidate)
            if self.lo <= below <= self.hi:
                self.at(below)
        lower, _ = self.interval(self.lo, self.hi)
        queue = [(lower, self.lo, self.hi)]
        while queue and self.best - queue[0][0] > MIN_TOL:
            lb, a, b = heapq.heappop(queue)
            if lb >= self.best - MIN_TOL:
                continue
            children = self.split(a, b)
            if not children:
                self.at(a)
                self.at(b)
                if min(self.at(a)[0], self.at(b)[0]) - lb > MIN_TOL:
                    raise ArithmeticError("oracle minimum resolution exhausted")
            for ca, cb in children:
                lower, _ = self.interval(ca, cb)
                if lower < self.best - MIN_TOL:
                    heapq.heappush(queue, (lower, ca, cb))
            self.nodes += 1
            if self.nodes > MAX_NODES:
                raise ArithmeticError("oracle minimum node limit")
        self.minimum_upper = self.best
        self.minimum_lower = max(0.0, self.best - MIN_TOL)
        self.optimizing = False
        return self.est

    def invert(self, cutoff, truth):
        accepted = []
        uncertain = []
        todo = [(self.lo, self.hi)]
        nodes = 0
        while todo:
            a, b = todo.pop()
            lower, upper = self.interval(a, b, True)
            if lower > cutoff + self.minimum_upper:
                continue
            if upper <= cutoff + self.minimum_lower:
                accepted.append((a, b))
                continue
            children = self.split(a, b)
            if b - a <= THETA_TOL * (1 + abs(a) + abs(b)) or not children:
                uncertain.append((a, b))
                mid = a + (b - a) / 2
                if self.at(mid)[0] - (self.minimum_lower + self.minimum_upper) / 2 <= cutoff:
                    accepted.append((a, b))
            else:
                todo.extend(children)
            nodes += 1
            if nodes > MAX_NODES:
                raise ArithmeticError("oracle inversion node limit")
        # The exact coverage decision uses the exact truth score and certified minimum bounds.
        raw = self.at(truth)[0]
        low = max(0.0, raw - self.minimum_upper)
        high = max(0.0, raw - self.minimum_lower)
        covered = True if high <= cutoff else False if low > cutoff else None
        components = []
        for a, b in sorted(accepted):
            if components and a <= np.nextafter(components[-1][1], np.inf):
                components[-1][1] = max(components[-1][1], b)
            else:
                components.append([a, b])
        lo = components[0][0] if components else None
        hi = components[-1][1] if components else None
        return dict(
            est=float(self.est),
            ci_lo=lo,
            ci_hi=hi,
            length=None if lo is None else hi - lo,
            confidence_set_length=sum(b - a for a, b in components),
            components=components,
            ci_type="EL",
            cutoff=float(cutoff),
            covered=covered,
            R_at_truth=(low + high) / 2,
            R_at_truth_bounds=[low, high],
            profile_minimum=self.minimum_upper,
            profile_minimum_bounds=[self.minimum_lower, self.minimum_upper],
            R_at_est=0.0,
            converged=bool(components),
            empty_confidence_set=not bool(components),
            ci_unbounded=False,
            ci_disconnected=len(components) > 1,
            boundary_uncertainty=uncertain,
            boundary_total_width=sum(b - a for a, b in uncertain),
            oracle_continuous_CDF=True,
            profile_minimum_certified_tolerance=MIN_TOL,
            endpoint_relative_tolerance=THETA_TOL,
            profile_evaluations=self.evaluations,
            inversion_nodes=nodes,
        )


@njit
def screen_el(p,q,cutoff):
    n=len(p);m=len(q)
    pm=p.mean();qm=q.mean();s=pm+qm
    pmin=p.min();pmax=p.max();qmin=q.min();qmax=q.max()
    if pmin+qmin>=0 or pmax+qmax<=0:return two_sample(p,q)
    lo=min((pmin-pm)/n,(qmin-qm)/m);hi=max((pmax-pm)/n,(qmax-qm)/m)
    extreme=-s/(n+m)-lo if s>=0 else s/(n+m)+hi
    bound=dual_lower_bound(s,p.var()/n+q.var()/m,extreme,n+m)
    if bound>cutoff+1e-6*(1+cutoff):return bound,3
    return two_sample(p,q)


@njit
def screen_one(q,cutoff):
    m=len(q);lo=q.min();hi=q.max();mean=q.mean()
    if lo>=0 or hi<=0:return one_sample(q)
    bound=4*m*mean*mean/max((hi-lo)**2,1e-24)
    if bound>cutoff+1e-6*(1+cutoff):return bound,3
    return one_sample(q)
