import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_uniform_grid(values, k=200, pad=0.5):
    lo, hi = np.quantile(values, [0.005, 0.995])
    span = hi - lo
    lo -= pad * span
    hi += pad * span
    grid = np.linspace(lo, hi, k)
    dx = grid[1] - grid[0]
    return grid, dx


def hist_density(values, grid):
    edges = np.concatenate([
        [grid[0] - (grid[1] - grid[0]) / 2],
        (grid[:-1] + grid[1:]) / 2,
        [grid[-1] + (grid[1] - grid[0]) / 2],
    ])
    counts, _ = np.histogram(values, bins=edges, density=True)
    return counts


def shift_matrix(grid, delta):
    k = grid.size
    dx = grid[1] - grid[0]
    r0 = grid[0]
    mat = np.zeros((k, k))
    for i in range(k):
        target = grid[i] + delta
        idx = (target - r0) / dx
        if idx < 0 or idx > k - 1:
            continue
        j0 = int(np.floor(idx))
        w1 = idx - j0
        if j0 == k - 1:
            mat[i, j0] = 1.0
        else:
            mat[i, j0] = 1.0 - w1
            mat[i, j0 + 1] = w1
    return mat


def clip_and_normalize(f, dx):
    f = np.clip(f, 0.0, None)
    mass = f.sum() * dx
    if mass > 0:
        f = f / mass
    return f


def run_once(args, rng=None):
    rng = np.random.default_rng(args.seed) if rng is None else rng

    # Label generation
    z = rng.binomial(1, args.pi1, size=args.n)

    # Proxy mechanism
    tpr = args.tpr
    tnr = args.tnr
    hat_z = np.where(
        z == 1,
        rng.binomial(1, tpr, size=args.n),
        rng.binomial(1, 1 - tnr, size=args.n),
    )

    # Regression function h(z)
    h0 = args.h0
    delta = args.delta
    h = h0 + delta * z

    # Residuals R | Z
    r = np.where(
        z == 1,
        rng.normal(args.mu1, args.sigma1, size=args.n),
        rng.normal(args.mu0, args.sigma0, size=args.n),
    )

    y = h + r
    h_hat = h0 + delta * hat_z
    r_tilde = y - h_hat

    # Known joint probabilities
    pi1 = args.pi1
    pi0 = 1 - pi1
    p11 = pi1 * tpr
    p01 = pi1 * (1 - tpr)
    p00 = pi0 * tnr
    p10 = pi0 * (1 - tnr)

    # Grid
    grid, dx = build_uniform_grid(np.concatenate([r, r_tilde]), k=args.k, pad=args.pad)

    # Oracle f
    f1 = hist_density(r[z == 1], grid)
    f0 = hist_density(r[z == 0], grid)
    f1 = clip_and_normalize(f1, dx)
    f0 = clip_and_normalize(f0, dx)

    # Empirical g from proxy residuals
    g1_cond = hist_density(r_tilde[hat_z == 1], grid)
    g0_cond = hist_density(r_tilde[hat_z == 0], grid)
    phat1 = (hat_z == 1).mean()
    phat0 = 1 - phat1
    g1 = g1_cond * phat1
    g0 = g0_cond * phat0

    # Oracle inversion
    s_plus = shift_matrix(grid, delta)
    s_minus = shift_matrix(grid, -delta)
    f0_plus = s_plus @ f0
    f1_minus = s_minus @ f1

    v1 = p10 * (f0_plus - f0)
    v0 = p01 * (f1_minus - f1)

    m = np.array([[p11, p10], [p01, p00]])
    m_inv = np.linalg.inv(m)

    f1_oracle = np.zeros_like(f1)
    f0_oracle = np.zeros_like(f0)
    for i in range(grid.size):
        vec = np.array([g1[i], g0[i]]) - np.array([v1[i], v0[i]])
        est = m_inv @ vec
        f1_oracle[i], f0_oracle[i] = est[0], est[1]

    f1_oracle = clip_and_normalize(f1_oracle, dx)
    f0_oracle = clip_and_normalize(f0_oracle, dx)

    # Proxy recovery (solve stacked system)
    i_k = np.eye(grid.size)
    a = np.block([
        [p11 * i_k, p10 * s_plus],
        [p01 * s_minus, p00 * i_k],
    ])
    b = np.concatenate([g1, g0])
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    f1_proxy = x[: grid.size]
    f0_proxy = x[grid.size :]
    f1_proxy = clip_and_normalize(f1_proxy, dx)
    f0_proxy = clip_and_normalize(f0_proxy, dx)

    # Metrics
    l1_oracle_f1 = np.sum(np.abs(f1_oracle - f1)) * dx
    l1_oracle_f0 = np.sum(np.abs(f0_oracle - f0)) * dx
    l1_proxy_f1 = np.sum(np.abs(f1_proxy - f1)) * dx
    l1_proxy_f0 = np.sum(np.abs(f0_proxy - f0)) * dx

    return {
        "l1_oracle_f1": l1_oracle_f1,
        "l1_oracle_f0": l1_oracle_f0,
        "l1_proxy_f1": l1_proxy_f1,
        "l1_proxy_f0": l1_proxy_f0,
    }


def pretty_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def summarize(metrics_list):
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in metrics_list]))
    out["l1_oracle_mean"] = 0.5 * (out["l1_oracle_f1"] + out["l1_oracle_f0"])
    out["l1_proxy_mean"] = 0.5 * (out["l1_proxy_f1"] + out["l1_proxy_f0"])
    return out


def plot_series(x, y_oracle, y_proxy, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.plot(x, y_oracle, marker="o", label="Oracle inversion")
    ax.plot(x, y_proxy, marker="o", label="Proxy recovery")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean L1 error")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_sweeps(args):
    pretty_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Sweep 1: sample size
    n_values = [int(v) for v in args.n_sweep.split(",")]
    oracle_vals = []
    proxy_vals = []
    for n in n_values:
        args.n = n
        metrics = [run_once(args, rng=rng) for _ in range(args.reps)]
        summary = summarize(metrics)
        oracle_vals.append(summary["l1_oracle_mean"])
        proxy_vals.append(summary["l1_proxy_mean"])
    plot_series(
        n_values,
        oracle_vals,
        proxy_vals,
        xlabel="Sample size",
        title="Recovery error vs sample size",
        out_path=out_dir / "perf_vs_sample_size.png",
    )

    # Sweep 2: class shift delta
    delta_values = [float(v) for v in args.delta_sweep.split(",")]
    oracle_vals = []
    proxy_vals = []
    for delta in delta_values:
        args.delta = delta
        metrics = [run_once(args, rng=rng) for _ in range(args.reps)]
        summary = summarize(metrics)
        oracle_vals.append(summary["l1_oracle_mean"])
        proxy_vals.append(summary["l1_proxy_mean"])
    plot_series(
        delta_values,
        oracle_vals,
        proxy_vals,
        xlabel="Class shift $h(1)-h(0)$",
        title="Recovery error vs class shift",
        out_path=out_dir / "perf_vs_class_shift.png",
    )

    # Sweep 3: proxy accuracy (tpr=tnr)
    acc_values = [float(v) for v in args.acc_sweep.split(",")]
    oracle_vals = []
    proxy_vals = []
    for acc in acc_values:
        args.tpr = acc
        args.tnr = acc
        metrics = [run_once(args, rng=rng) for _ in range(args.reps)]
        summary = summarize(metrics)
        oracle_vals.append(summary["l1_oracle_mean"])
        proxy_vals.append(summary["l1_proxy_mean"])
    plot_series(
        acc_values,
        oracle_vals,
        proxy_vals,
        xlabel="Proxy accuracy (TPR = TNR)",
        title="Recovery error vs proxy accuracy",
        out_path=out_dir / "perf_vs_proxy_accuracy.png",
    )


def simulate(args):
    metrics = run_once(args)
    print("=== Simulation Summary ===")
    print(f"n={args.n}, k={args.k}, delta={args.delta}")
    print(f"pi1={args.pi1:.3f}, tpr={args.tpr:.3f}, tnr={args.tnr:.3f}")
    print("--- L1 distance (oracle inversion) ---")
    print(f"f1: {metrics['l1_oracle_f1']:.4f}")
    print(f"f0: {metrics['l1_oracle_f0']:.4f}")
    print("--- L1 distance (proxy recovery) ---")
    print(f"f1: {metrics['l1_proxy_f1']:.4f}")
    print(f"f0: {metrics['l1_proxy_f0']:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Validate proxy residual inversion via simulation.")
    parser.add_argument("--n", type=int, default=200000)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pad", type=float, default=0.5)

    parser.add_argument("--pi1", type=float, default=0.5)
    parser.add_argument("--tpr", type=float, default=0.85)
    parser.add_argument("--tnr", type=float, default=0.85)

    parser.add_argument("--h0", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=2.0)

    parser.add_argument("--mu0", type=float, default=0.0)
    parser.add_argument("--mu1", type=float, default=0.5)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma1", type=float, default=1.2)

    parser.add_argument("--plot", action="store_true", help="Generate sweep plots")
    parser.add_argument("--out-dir", type=str, default="plots")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-sweep", type=str, default="5000,10000,30000,100000,200000")
    parser.add_argument("--delta-sweep", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--acc-sweep", type=str, default="0.55,0.65,0.75,0.85,0.95")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.plot:
        run_sweeps(args)
    else:
        simulate(args)
