"""Neural nuisances, donor weights, and fixed polynomial baselines."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import copy
import hashlib
import resource
import time
from dataclasses import dataclass
import numpy as np
import torch
from numba import njit
from scipy.special import expit
from scipy.linalg import solve, svd
from sklearn.preprocessing import PolynomialFeatures
_THREAD_LIMIT = None
WIDTH, STEPS, RATE, BATCH = 32, 4000, 0.0001, 512


@dataclass(frozen=True)
class Observed:
    xp: np.ndarray
    yp: np.ndarray
    xq: np.ndarray


def initialize():
    """Use deterministic CPU operations and one numerical thread per worker."""
    global _THREAD_LIMIT
    if _THREAD_LIMIT is not None:
        return

    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _THREAD_LIMIT = threadpool_limits(1)


def cpu_seconds():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def stamp():
    return time.perf_counter(), cpu_seconds()


def elapsed(start):
    return {"wall": time.perf_counter() - start[0], "cpu": cpu_seconds() - start[1]}


def data_hash(obs):
    return hashlib.sha256(obs.xp.tobytes() + obs.yp.tobytes() + obs.xq.tobytes()).hexdigest()


def json_compatible(value):
    """Convert NumPy scalars and nonfinite endpoints to portable JSON values."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_compatible(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


class Network(torch.nn.Module):
    def __init__(self, dim, cfg, kind):
        super().__init__()
        layers = []
        for width in cfg["hidden"]:
            layers.extend([torch.nn.Linear(dim, width), torch.nn.ReLU()])
            dim = width
        layers.append(torch.nn.Linear(dim, 1))
        self.layers = torch.nn.Sequential(*layers)
        self.kind, self.cap = kind, cfg["cap"]
        with torch.no_grad():
            self.layers[-1].weight.mul_(0.1)
            self.layers[-1].bias.fill_(1.0 if kind == "ratio" else 0.0)

    def forward(self, x):
        value = self.layers(x).flatten()
        return torch.clamp(value, 0.0, self.cap) if self.kind == "ratio" else value


class RatioFit:

    def __init__(self, model, center, scale, loc, spread, record):
        (self.model, self.center, self.scale) = (model, center, scale)
        (self.loc, self.spread, self.record) = (loc, spread, record)

    def predict(self, x):
        with torch.no_grad():
            return self.model(torch.as_tensor((x - self.center) / self.scale, dtype=torch.float64)).numpy()


def _constant(dim, center, scale, cap, width=16):
    model = Network(dim, dict(hidden=[width, width], cap=cap), "ratio").double()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.layers[-1].bias.fill_(1.0)
    return RatioFit(model, center, scale, 0.0, 1.0, {})


def _pairs(x, y, ids, rng):
    ids = rng.permutation(ids)
    h = len(ids) // 2
    a, b = ids[:h], ids[h: 2 * h]
    return [
        (
            np.column_stack((x[own], y[rng.permutation(other)])),
            np.column_stack((x[own], y[own])),
        )
        for own, other in [(a, b), (b, a)]
    ]


def validation_loss(net, a, b, center, scale, va, vb, rao_blackwell=False):
    if not rao_blackwell:
        return float(net(va).square().mean() - 2 * net(vb).mean())
    layers = net.layers
    w1 = layers[0].weight.detach().numpy().astype(np.float64)
    bias = layers[0].bias.detach().numpy().astype(np.float64)
    (x, y) = (a[:, :-1], a[:, -1])
    order = np.argsort(y)
    slope = (w1[:, -1] / scale[-1])[None, :]
    intercept = ((x - center[:-1]) / scale[:-1] @ w1[:, :-1].T +
                 bias - w1[:, -1] * center[-1] / scale[-1])[None, :, :]
    args = [slope, intercept, layers[2].weight.detach().numpy().astype(np.float64)[None, :, :], layers[2].bias.detach().numpy().astype(
        np.float64)[None, :], layers[4].weight.detach().numpy().astype(np.float64), np.array([float(layers[4].bias.detach()[0])]), np.array([net.cap])]
    (weights, _) = _weights(y[order], order, *[np.ascontiguousarray(v) for v in args])
    return float(np.mean(weights ** 2) - 2 * float(net(vb).mean()))


def refit(fit, a, b, selected, seed, cfg, optcfg):
    net = fit.model
    dtype = torch.float32
    net.to(dtype=dtype)
    (aa, bb) = [torch.tensor((v - fit.center) / fit.scale, dtype=dtype) for v in (a, b)]

    def risk():
        return net(aa).square().mean() - 2 * net(bb).mean()
    with torch.no_grad():
        initial = float(risk())
    updates = max(optcfg['refit_min'], min(optcfg['refit_max'], int(
        np.ceil(optcfg['refit_fraction'] * selected['updates']))))
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=optcfg['refit_learning_rate'])
    (best_risk, best_state) = (initial, copy.deepcopy(net.state_dict()))
    for _ in range(updates):
        if optcfg.get('refit_full_batch', False):
            (ia, ib) = (np.arange(len(a)), np.arange(len(b)))
        else:
            ia = rng.integers(len(a), size=cfg['batch_size'])
            ib = ia if optcfg.get('paired_batch', False) else rng.integers(
                len(b), size=cfg['batch_size'])
        optimizer.zero_grad()
        value = net(aa[ia]).square().mean() - 2 * net(bb[ib]).mean()
        if not torch.isfinite(value):
            raise ArithmeticError('Nonfinite original LS warm-refit loss')
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'])
        optimizer.step()
        if optcfg.get('restore_lowest_empirical_loss', False):
            with torch.no_grad():
                current = float(risk())
            if current < best_risk:
                (best_risk, best_state) = (current, copy.deepcopy(net.state_dict()))
    if optcfg.get('restore_lowest_empirical_loss', False):
        net.load_state_dict(best_state)
    with torch.no_grad():
        final = float(risk())
    if not np.isfinite(final):
        raise ArithmeticError('Nonfinite full-sample final loss')
    net.eval().double()
    return (fit, dict(updates=updates, initial_loss=initial, final_loss=final))


def tilt_initialization(net, a, b, center, scale):
    # Domain logit only supplies network parameters. No true direction, outcome,
    # fitted DRW weights, or parametric restriction enters the final estimator.
    beta = fit_logit(a, b)["beta"]
    vector = beta[1:] * scale
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return dict(
            kind="domain_tilt_initialization", fallback="original_random_initialization"
        )
    direction = vector / norm
    z = np.vstack(((a - center) / scale, (b - center) / scale)) @ direction
    width = net.layers[0].weight.shape[0]
    knots = np.linspace(float(z.min() - 0.25), float(z.max() + 0.25), width)
    intercept = float(beta[0] + np.log(len(a) / len(b)) + beta[1:] @ center)
    values = np.minimum(net.cap, np.exp(np.clip(intercept + norm * knots, -30, 30)))
    slopes = np.diff(values) / np.diff(knots)
    coef = np.r_[slopes[0], np.diff(slopes), -slopes[-1]]
    with torch.no_grad():
        l1, l2, l3 = net.layers[0], net.layers[2], net.layers[4]
        l1.weight.copy_(torch.tensor(np.tile(direction, (width, 1)), dtype=l1.weight.dtype))
        l1.bias.copy_(torch.tensor(-knots, dtype=l1.bias.dtype))
        l2.weight.copy_(torch.eye(width, dtype=l2.weight.dtype))
        l2.bias.zero_()
        l3.weight.copy_(torch.tensor(coef[None, :], dtype=l3.weight.dtype))
        l3.bias.fill_(float(values[0]))
    assert all(p.requires_grad for p in net.parameters())
    return dict(
        kind="domain_tilt_parameter_initialization",
        no_outcomes_used=True,
        no_known_direction=True,
        all_parameters_remain_free=True,
        original_NN_LS_training_follows=True,
        logit_not_used_as_final_estimator=True,
    )


def train_checkpoint(a, b, valid, width, cap, seed, cfg, policy, *, initializer=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1e-06)
    (aa, bb, va, vb) = [torch.tensor((v - center) / scale, dtype=torch.float32)
                        for v in (a, b, *valid)]
    net = Network(a.shape[1], dict(hidden=[width, width], cap=cap), 'ratio')
    if policy.get('initialization') == 'tilt':
        (tilt_initialization if initializer is None else initializer)(net, a, b, center, scale)
    opt = torch.optim.Adam(net.parameters(), lr=policy.get('learning_rate', cfg['learning_rate']))
    with torch.no_grad():
        initial_loss = float(net(va).square().mean() - 2 * net(vb).mean())
    best = initial_loss if policy.get('initial_checkpoint', False) else np.inf
    state = copy.deepcopy(net.state_dict()) if np.isfinite(best) else None
    best_step = 0
    stale = 0
    for step in range(1, policy.get('maximum_updates', cfg['ratio_updates']) + 1):
        if policy.get('learning_schedule') == 'decay':
            for group in opt.param_groups:
                group['lr'] = policy.get('learning_rate', cfg['learning_rate']
                                         ) / (1 + max(0, step - 80) / 120)
        ia = rng.integers(len(a), size=cfg['batch_size'])
        ib = rng.integers(len(b), size=cfg['batch_size'])
        opt.zero_grad()
        loss = net(aa[ia]).square().mean() - 2 * net(bb[ib]).mean()
        if not torch.isfinite(loss):
            raise ArithmeticError('Nonfinite original NN-LS loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'])
        opt.step()
        if step % cfg['check_every'] == 0:
            with torch.no_grad():
                value = float(net(va).square().mean() - 2 * net(vb).mean())
            if value < best - 1e-07:
                (best, best_step, stale, state) = (value, step, 0, copy.deepcopy(net.state_dict()))
            else:
                stale += 1
            if step >= policy.get('minimum_updates', 80) and stale >= policy.get('patience', 8):
                break
    assert state is not None
    net.load_state_dict(state)
    net.eval().double()
    return (RatioFit(net, center, scale, 0.0, 1.0, {}), dict(loss=best, updates=best_step, width=width))


def fit_ratio(xp, xq, cfg, seed, policy, *, trainer=None):
    """One prescribed width; validation chooses a training checkpoint only."""
    rng = np.random.default_rng(seed)
    p, q = rng.permutation(len(xp)), rng.permutation(len(xq))
    npv, nqv = len(p) // 5, len(q) // 5
    cap = policy.get("cap_multiplier", 2.0) * len(xp)**cfg["cap_power"]
    widths = policy.get("widths") or [policy["width_by_n"][str(len(xp))]]
    if len(widths) != 1:
        raise ValueError("This experiment prescribes one density-ratio width")
    width = widths[0]
    train = train_checkpoint if trainer is None else trainer
    fit, record = train(
        xp[p[npv:]], xq[q[nqv:]], (xp[p[:npv]], xq[q[:nqv]]),
        width, cap, seed + width, cfg, policy
    )
    options = dict(
        refit_fraction=policy.get("refit_fraction", .25),
        refit_min=policy.get("refit_min", 20),
        refit_max=policy.get("refit_max", 100),
        refit_learning_rate=policy.get("refit_learning_rate", .00025),
        paired_batch=False, refit_full_batch=True,
        restore_lowest_empirical_loss=True
    )
    fit, _ = refit(fit, xp, xq, record, seed + 100, cfg, options)
    fit.record = dict(cap=cap, width=width, updates=record["updates"])
    return fit


def fit_logit(xp, xq, source_weights=None, target_weights=None):
    (n, m) = (len(xp), len(xq))
    zp = np.column_stack((np.ones(n), xp))
    zq = np.column_stack((np.ones(m), xq))
    z = np.vstack((zp, zq))
    label = np.r_[np.zeros(n), np.ones(m)]
    wp = np.ones(n) if source_weights is None else source_weights
    wq = np.ones(m) if target_weights is None else target_weights
    w = np.r_[wp, wq]
    beta = np.r_[np.log(m / n), np.zeros(xp.shape[1])]

    def objective(b):
        eta = z @ b
        return float(w @ (np.logaddexp(0, eta) - label * eta) / (n + m))
    converged = False
    for iteration in range(50):
        pr = expit(z @ beta)
        score = z.T @ (w * (label - pr)) / (n + m)
        hessian = z.T @ ((w * pr * (1 - pr))[:, None] * z) / (n + m)
        if np.max(abs(score)) < 1e-11:
            converged = True
            break
        step = solve(hessian, score, assume_a='pos')
        old = objective(beta)
        scale = 1.0
        for _ in range(30):
            trial = beta + scale * step
            if objective(trial) <= old + 1e-14:
                beta = trial
                break
            scale *= 0.5
        else:
            raise ArithmeticError('DRW logit line search failed')
    if not converged:
        raise ArithmeticError('DRW logit did not converge')
    raw = np.exp(zp @ beta) * n / m
    if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
        raise ArithmeticError('DRW has invalid density ratios')
    return dict(beta=beta)


def spline_coefficients(knots, values):
    slopes = np.diff(values) / np.diff(knots)
    return np.r_[slopes[0], np.diff(slopes), -slopes[-1]], float(values[0])


def product_initialization(net, a, b):
    """Training-only NN parameter initialization, followed by unrestricted original-LS training.

    Polarization of two piecewise-linear scores is encoded in the existing two
    hidden layers. Scalar output amplitude minimizes the original training LS
    along that initialization direction. No nuisance truth or outcome regression.
    """
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1e-6)
    za, zb = (a - center) / scale, (b - center) / scale
    x = zb[:, :-1] - zb[:, :-1].mean(0)
    y = zb[:, -1] - zb[:, -1].mean()
    cov = np.mean(x * y[:, None], axis=0)
    norm = float(np.linalg.norm(cov))
    if norm < 1e-8:
        return dict(
            kind="product", fallback="covariance", detail=covariance_initialization(net, a, b)
        )
    direction = cov / norm
    w = net.layers[0].weight.shape[0]
    k = w // 2
    levels = np.linspace(0.02, 0.98, k)
    knots_x = np.quantile(zb[:, :-1] @ direction, levels)
    knots_y = np.quantile(np.r_[za[:, -1], zb[:, -1]], levels)
    if min(np.min(np.diff(knots_x)), np.min(np.diff(knots_y))) < 1e-8:
        return dict(
            kind="product",
            fallback="degenerate_ranks",
            detail=covariance_initialization(net, a, b),
        )
    qknots = np.linspace(-2, 2, k)
    qcoef, _ = spline_coefficients(qknots, qknots * qknots)
    aa, bb = [torch.tensor(z, dtype=net.layers[0].weight.dtype) for z in [za, zb]]
    best = None
    candidates = []
    starts = [("linear", 0.0, 0.0)] + [
        ("rank", sx, sy) for sx in [1.0, 2.0, 4.0, 8.0] for sy in [1.0, 2.0, 4.0, 8.0]
    ]
    for family, sx, sy in starts:
        kx, ky = (
            (np.linspace(-8, 8, k), np.linspace(-8, 8, k))
            if family == "linear"
            else (knots_x, knots_y)
        )
        fx, f0 = spline_coefficients(
            kx, kx / 3 if family == "linear" else np.clip(sx * (2 * levels - 1), -1, 1)
        )
        gy, g0 = spline_coefficients(
            ky, ky / 3 if family == "linear" else np.clip(sy * (2 * levels - 1), -1, 1)
        )
        with torch.no_grad():
            l1, l2, l3 = net.layers[0], net.layers[2], net.layers[4]
            l1.weight.zero_()
            l1.bias.zero_()
            l2.weight.zero_()
            l2.bias.zero_()
            l3.weight.zero_()
            l1.weight[:k, :-1] = torch.tensor(
                np.tile(direction, (k, 1)), dtype=l1.weight.dtype
            )
            l1.weight[k: 2 * k, -1] = 1.0
            l1.bias[:k] = torch.tensor(-kx, dtype=l1.weight.dtype)
            l1.bias[k: 2 * k] = torch.tensor(-ky, dtype=l1.weight.dtype)
            for j in range(k):
                l2.weight[j, :k] = torch.tensor(fx, dtype=l2.weight.dtype)
                l2.weight[j, k: 2 * k] = torch.tensor(gy, dtype=l2.weight.dtype)
                l2.weight[k + j, :k] = torch.tensor(fx, dtype=l2.weight.dtype)
                l2.weight[k + j, k: 2 * k] = torch.tensor(-gy, dtype=l2.weight.dtype)
                l2.bias[j] = f0 + g0 - qknots[j]
                l2.bias[k + j] = f0 - g0 - qknots[j]
            l3.weight[0, :k] = torch.tensor(qcoef / 4, dtype=l3.weight.dtype)
            l3.weight[0, k: 2 * k] = torch.tensor(-qcoef / 4, dtype=l3.weight.dtype)
            l3.bias.fill_(0.0)
            ga = net.layers(aa).flatten().numpy().astype(float)
            gb = net.layers(bb).flatten().numpy().astype(float)
            raw = float((gb.mean() - ga.mean()) / max(np.mean(ga**2), 1e-12))
            steps = np.unique(np.clip(np.array([0.0, 0.5, 1.0, 1.5]) * raw, -8, 8))

            def coordinate_risk(alpha):
                return float(
                    np.mean(np.clip(1 + alpha * ga, 0, net.cap) ** 2) -
                    2 * np.mean(np.clip(1 + alpha * gb, 0, net.cap))
                )

            amplitude = float(min(steps, key=coordinate_risk))
            l3.weight.mul_(amplitude)
            l3.bias.fill_(1.0)
            loss = float(net(aa).square().mean() - 2 * net(bb).mean())
            record = dict(
                family=family,
                scale_x=sx,
                scale_y=sy,
                amplitude=amplitude,
                original_training_LS=loss,
            )
            candidates.append(record)
            if best is None or loss < best[0]:
                best = (loss, copy.deepcopy(net.state_dict()), record)
    net.load_state_dict(best[1])
    assert all(p.requires_grad for p in net.parameters())
    return dict(
        kind="product",
        selected=best[2],
        initial_coordinate_candidates=candidates,
        training_direction=direction.tolist(),
        all_weights_unrestricted=True,
        construction="Empirical-rank NN polarization warm start; same original empirical LS scalar coordinate initialization",
        no_known_direction=True,
        no_conditional_mean_regression=True,
    )


def train_path(a, b, valid, width, cap, seed, cfg, policy, *, initializer=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1e-06)
    (aa, bb, va, vb) = [torch.tensor((v - center) / scale, dtype=torch.float32)
                        for v in (a, b, *valid)]
    net = Network(a.shape[1], dict(hidden=[width, width], cap=cap), 'ratio')
    if initializer is None:
        initializer = product_initialization if policy['initialization'] == 'product' else covariance_initialization
    initializer(net, a, b)
    optimizer = torch.optim.Adam(net.parameters(), lr=policy.get('learning_rate', 0.002))
    with torch.no_grad():
        initial_validation = validation_loss(net, *valid, center, scale, va, vb, True)
    eligible = policy.get('initial_checkpoint', True)
    best = initial_validation if eligible else np.inf
    best_step = 0
    stale = 0
    state = copy.deepcopy(net.state_dict()) if eligible else None
    maximum = policy.get('maximum_updates', 1200)
    minimum = policy.get('minimum_updates', 80)
    for step in range(1, maximum + 1):
        if policy.get('learning_schedule') == 'decay':
            rate = policy.get('learning_rate', 0.002) / (1 + max(0, step - 80) / 120)
            for group in optimizer.param_groups:
                group['lr'] = rate
        ids = rng.integers(len(a), size=cfg['batch_size'])
        optimizer.zero_grad()
        value = net(aa[ids]).square().mean() - 2 * net(bb[ids]).mean()
        if not torch.isfinite(value):
            raise ArithmeticError('Nonfinite original LS training objective')
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'])
        optimizer.step()
        if step % cfg['check_every'] == 0:
            with torch.no_grad():
                loss = validation_loss(net, *valid, center, scale, va, vb, True)
            if loss < best - 1e-07:
                (best, best_step, stale, state) = (loss, step, 0, copy.deepcopy(net.state_dict()))
            else:
                stale += 1
            if step >= minimum and stale >= policy.get('patience', 8):
                break
    assert state is not None
    net.load_state_dict(state)
    net.eval().double()
    fit = RatioFit(net, center, scale, 0.0, 1.0, {})
    record = dict(width=width, cap=cap, loss=best, updates=best_step)
    return (fit, record)


def fit_dependence(obs, policy, cfg, seed, *, trainer=None, refitter=None):
    """Fit the two directional ratio networks under one frozen training policy."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(obs.yp))
    validation_size = 2 * (len(ids) // 10)
    training = _pairs(obs.xp, obs.yp, ids[validation_size:], rng)
    valid = _pairs(obs.xp, obs.yp, ids[:validation_size], rng)
    full = _pairs(obs.xp, obs.yp, np.arange(len(ids)), rng)
    n = len(obs.yp)
    width = policy['width_by_n'][str(n)]
    cap_multiplier = policy.get('cap_multiplier', 1.0)
    cap = cap_multiplier * (n // 2) ** cfg['cap_power']
    train = train_path if trainer is None else trainer
    refit_network = refit if refitter is None else refitter
    (fits, records) = ([], [])
    for direction in range(2):
        (fit, record) = train(
            *training[direction], valid[direction], width, cap, seed + width + 100 * direction, cfg, policy)
        fits.append(fit)
        records.append(record)
    loss = aggregate_validation(fits, valid)
    constant = loss >= -1.0
    options = dict(refit_fraction=policy.get('refit_fraction', 0.25), refit_min=policy.get('refit_min', 20), refit_max=policy.get(
        'refit_max', 100), refit_learning_rate=policy.get('refit_learning_rate', 0.001), paired_batch=True, refit_full_batch=policy.get('refit_full_batch', False))
    final = []
    for (direction, (a, b)) in enumerate(full):
        if constant:
            model = _constant(obs.xp.shape[1] + 1, a.mean(0),
                              np.maximum(a.std(0), 1e-06), cap, width)
            record = dict(trained_updates=0, objective_evaluations=0)
        else:
            (model, record) = refit_network(copy.deepcopy(
                fits[direction]), a, b, records[direction], seed + 1000 + direction, cfg, options)
        final.append(model)
    metadata = dict(selected=dict(width=width), constant_selected=constant)
    return (final, metadata)


def covariance_initialization(net, a, b):
    """Use training-only cross covariance as a starting point; every weight stays free."""
    center = a.mean(0)
    scale = np.maximum(a.std(0), 1e-6)
    z = (b - center) / scale
    x = z[:, :-1] - z[:, :-1].mean(0)
    y = z[:, -1] - z[:, -1].mean()
    covariance = np.mean(x * y[:, None], axis=0)
    norm = float(np.linalg.norm(covariance))
    if norm < 1e-8:
        return dict(kind="covariance", fallback_random=True, covariance_norm=norm)
    direction = covariance / norm
    layer = net.layers[0]
    width = layer.weight.shape[0]
    levels = np.linspace(-1, 1, int(np.ceil(width / 4)))
    with torch.no_grad():
        original = layer.weight.detach().clone()
        for j in range(width):
            sx = 1.0 if j % 4 < 2 else -1.0
            sy = 1.0 if j % 4 in [0, 2] else -1.0
            layer.weight[j, :-1] = (
                0.5 * sx * torch.tensor(direction, dtype=layer.weight.dtype) +
                0.2 * original[j, :-1]
            )
            layer.weight[j, -1] = 0.5 * sy + 0.2 * original[j, -1]
            layer.bias[j] = float(levels[j // 4])
    assert all(p.requires_grad for p in net.parameters())
    return dict(
        kind="covariance",
        fallback_random=False,
        covariance_norm=norm,
        training_direction=direction.tolist(),
        all_weights_unrestricted=True,
    )


def aggregate_validation(fits, valid):
    """Original raw-rho LS on independent cross-half validation products; no normalization."""
    losses = []
    for (a, b) in valid:
        (raw, _) = dependence_weights(fits, a[:, :-1], a[:, -1])
        joint = sum((f.predict(b) for f in fits)) / 2
        losses.append(float(np.mean(raw ** 2) - 2 * np.mean(joint)))
    return float(np.mean(losses))


@njit
def row_predictions(y, a, b, w2, b2, w3, b3, cap):
    n = len(y)
    h1 = len(a)
    h2 = len(b2)
    lo = y[0]
    hi = y[-1]
    first = np.empty(h1)
    first_id = np.empty(h1, np.int64)
    count = 0
    for j in range(h1):
        if a[j] != 0.0:
            z = -b[j] / a[j]
            if lo < z < hi:
                first[count] = z
                first_id[count] = j
                count += 1
    index = np.argsort(first[:count])
    first = first[:count][index]
    first_id = first_id[:count][index]
    right = first[0] if count else hi
    middle = lo + (right - lo) / 2
    sa = np.zeros(h2)
    sb = b2.copy()
    for j in range(h1):
        if a[j] * middle + b[j] > 0:
            for l in range(h2):
                sa[l] += w2[l, j] * a[j]
                sb[l] += w2[l, j] * b[j]
    output = np.empty(n)
    k = 0
    left = lo
    pieces = 0
    for segment in range(count + 1):
        right = first[segment] if segment < count else hi
        if k < n and y[k] <= right:
            second = np.empty(h2)
            second_id = np.empty(h2, np.int64)
            nc = 0
            for l in range(h2):
                if sa[l] != 0.0:
                    z = -sb[l] / sa[l]
                    if left < z < right:
                        second[nc] = z
                        second_id[nc] = l
                        nc += 1
            order = np.argsort(second[:nc])
            second = second[:nc][order]
            second_id = second_id[:nc][order]
            end = second[0] if nc else right
            mid = left + (end - left) / 2
            slope = 0.0
            intercept = b3
            for l in range(h2):
                if sa[l] * mid + sb[l] > 0:
                    slope += w3[l] * sa[l]
                    intercept += w3[l] * sb[l]
            for sub in range(nc + 1):
                end = second[sub] if sub < nc else right
                pieces += 1
                while k < n and y[k] <= end:
                    output[k] = min(cap, max(0.0, slope * y[k] + intercept))
                    k += 1
                if sub < nc:
                    l = second_id[sub]
                    change = w3[l] * abs(sa[l])
                    slope += change
                    intercept -= change * second[sub]
        if segment < count:
            j = first_id[segment]
            change = abs(a[j])
            for l in range(h2):
                sa[l] += w2[l, j] * change
                sb[l] -= w2[l, j] * change * right
        left = right
    return output, pieces


@njit
def _weights(y, order, a, b, w2, b2, w3, b3, cap):
    nf, nx, h1 = b.shape
    out = np.empty((nx, len(y)))
    segments = np.zeros(nf, np.int64)
    for i in range(nx):
        total = np.zeros(len(y))
        for j in range(nf):
            pred, count = row_predictions(y, a[j], b[j, i], w2[j], b2[j], w3[j], b3[j], cap[j])
            total += pred
            segments[j] += count
        for k in range(len(y)):
            out[i, order[k]] = total[k] / nf
    return out, segments


def dependence_weights(fits, x, y):
    aa = []
    bb = []
    ww2 = []
    bias2 = []
    ww3 = []
    bias3 = []
    caps = []
    for f in fits:
        layers = f.model.layers
        if len(layers) != 5 or f.model.kind != "ratio" or f.loc != 0.0 or f.spread != 1.0:
            raise ValueError("piecewise engine requires a two-hidden-layer ratio ReLU network")
        w1 = layers[0].weight.detach().cpu().numpy().astype(np.float64)
        b1 = layers[0].bias.detach().cpu().numpy().astype(np.float64)
        aa.append(w1[:, -1] / f.scale[-1])
        bb.append(
            ((x - f.center[:-1]) / f.scale[:-1]) @ w1[:, :-1].T +
            b1 -
            w1[:, -1] * f.center[-1] / f.scale[-1]
        )
        ww2.append(layers[2].weight.detach().cpu().numpy().astype(np.float64))
        bias2.append(layers[2].bias.detach().cpu().numpy().astype(np.float64))
        ww3.append(layers[4].weight.detach().cpu().numpy().astype(np.float64).ravel())
        bias3.append(float(layers[4].bias.detach().cpu().numpy()[0]))
        caps.append(f.model.cap)
    order = np.argsort(y)
    ys = y[order]
    weights, segments = _weights(
        ys, order, *[np.ascontiguousarray(v) for v in [aa, bb, ww2, bias2, ww3, bias3, caps]]
    )
    # Cancellation at an identically zero clipped row must not create a tiny
    # artificial normalization mass. Re-evaluate only these rare rows directly.
    safeguard = np.flatnonzero(weights.sum(1) <= 1e-8)
    for i in safeguard:
        pairs = np.column_stack((np.repeat(x[i: i + 1], len(y), axis=0), y))
        weights[i] = sum(f.predict(pairs) for f in fits) / len(fits)
    return weights, dict(
        mean_affine_pieces=(segments / len(x)).tolist(),
        zero_mass_safeguard_rows=len(safeguard),
    )


class FixedBasis:
    def __init__(self, x, degree=3):
        self.center = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-12)
        self.transformer = PolynomialFeatures(degree, include_bias=True)
        self.transformer.fit(x)

    def __call__(self, x):
        return self.transformer.transform((x - self.center) / self.scale)


def fit_sieve(x, y, cfg=None, seed=None):
    degree = 3 if cfg is None else cfg.get('degree', 3)
    basis = FixedBasis(x, degree)
    p = basis(x)
    (u, s, vt) = svd(p, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    tolerance = max(p.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tolerance))
    if rank != p.shape[1]:
        raise ArithmeticError('prespecified unpenalized Sieve is rank deficient')
    inverse = vt.T / s @ u.T
    record = dict(dimension=p.shape[1], degree=3)
    return (basis, p, inverse, record)


def max10_point(y, order, design, inverse, weights):
    """Find the first donor threshold crossing one half for the weighted CDF power."""
    coefficients = np.cumsum(inverse[:, order], axis=1)
    curve = np.empty(len(y))
    for first in range(0, len(y), 128):
        cdf = np.clip(design @ coefficients[:, first: first + 128], 0, 1)
        curve[first: first + 128] = weights @ (cdf**10) / len(weights)

    above = curve >= 0.5
    crossings = np.flatnonzero(above)
    if not len(crossings):
        raise ArithmeticError("Neural DRW Max10 has no generalized root")
    upcrossings = int(above[0]) + int(np.sum(above[1:] & ~above[:-1]))
    return float(y[order[crossings[0]]]), upcrossings


def cdf_at(y, estimate, source, target, inverse):
    event = (y <= estimate).astype(float)
    coefficients = inverse @ event
    source_cdf = source @ coefficients
    return event, source_cdf, np.clip(target @ coefficients, 0.0, 1.0)


class FixedPredictor:
    def __init__(self, net, center, scale):
        self.net, self.center, self.scale = net, center, scale

    def predict(self, x):
        with torch.no_grad():
            xx = torch.as_tensor((x - self.center) / self.scale, dtype=torch.float32)
            value = torch.sigmoid(self.net(xx)).numpy().astype(np.float64)
        return value


def fixed_network(x, y, seed, kind):
    if kind != "binary":
        raise ValueError("this experiment uses binary conditional CDF regression")
    rng = np.random.default_rng(seed)
    center = x.mean(0)
    scale = np.maximum(x.std(0), 1e-8)
    xx = torch.as_tensor((x - center) / scale, dtype=torch.float32)
    yy = torch.as_tensor(y, dtype=torch.float32)
    torch.manual_seed(seed + WIDTH)
    net = Network(x.shape[1], dict(hidden=[WIDTH, WIDTH], cap=10), "binary")
    optimizer = torch.optim.Adam(net.parameters(), lr=RATE)
    for step in range(STEPS):
        ids = rng.choice(len(y), size=BATCH, replace=True)
        output = net(xx[ids])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, yy[ids])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    return FixedPredictor(net, center, scale), dict(selected=dict(family="fixed_neural",
                                                                  width=WIDTH, updates=STEPS, learning_rate=RATE, batch_size=BATCH),
                                                    model_selection=False)


def weighted_quantile(y, r, tau):
    order = np.argsort(y)
    weights = r[order]
    if weights.sum() <= 0:
        raise ArithmeticError("preliminary ratio mass is zero")
    return float(
        y[order[min(np.searchsorted(np.cumsum(weights) / weights.sum(), tau), len(y) - 1)]]
    )


def fit_localized(obs, cfg, seed, taus, *, fit_ratio):
    """One three-fold partition; each fold has separate pilot and regression data."""
    (xp, y, xq) = (obs.xp, obs.yp, obs.xq)
    (n, m) = (len(y), len(xq))
    taus = np.asarray(taus)
    rng = np.random.default_rng(seed)
    fp = np.array_split(rng.permutation(n), 3)
    fq = np.array_split(rng.permutation(m), 3)
    r = np.empty(n)
    cp = np.empty((n, len(taus)))
    cq = np.empty((m, len(taus)))
    (records, pilots) = ([], [])
    for fold in range(3):
        (pilot, reg) = ((fold + 1) % 3, (fold + 2) % 3)
        (ii, jj, h1, h2) = (fp[fold], fq[fold], fp[pilot], fp[reg])
        (tp, tq) = (np.r_[h1, h2], np.r_[fq[pilot], fq[reg]])
        sk = seed + 10000 * fold
        rf = fit_ratio(xp[tp], xq[tq], cfg['neural'], sk + 1)
        r[ii] = rf.predict(xp[ii])
        hp = np.array_split(rng.permutation(len(h1)), 2)
        hq = np.array_split(rng.permutation(len(fq[pilot])), 2)
        rw = np.empty(len(h1))
        for h in range(2):
            fit = fit_ratio(xp[h1[hp[1 - h]]], xq[fq[pilot][hq[1 - h]]],
                            cfg['neural'], sk + 10 + h)
            rw[hp[h]] = fit.predict(xp[h1[hp[h]]])
        initial = [weighted_quantile(y[h1], rw, tau) for tau in taus]
        for (j, tau) in enumerate(taus):
            (fit, record) = fixed_network(xp[h2], (y[h2] <= initial[j]).astype(
                float), sk + 100 + int(round(tau * 100)), 'binary')
            (cp[ii, j], cq[jj, j]) = (fit.predict(xp[ii]), fit.predict(xq[jj]))
            records.append(record)
        pilots.append(initial)
    return dict(r=r, cp=cp, cq=cq, records=records, pilots=pilots)
