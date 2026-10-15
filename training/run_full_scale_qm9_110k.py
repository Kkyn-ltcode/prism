"""Experiment C: Full-QM9 multi-seed evaluation (N = 110,000, 5 seeds)."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import ProcessedQM9Subset, QM9_TARGET_INDICES
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2
from models.eq_gnn import QPRISMModelV2

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV",
                "mu": "Debye", "alpha": "Bohr³", "u0": "eV"}
TARGET_DISPLAY = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap",
                  "mu": "Dipole", "alpha": "Polariz.", "u0": "U₀"}
MODEL_LABELS = {"prism": "PRISM (SIREN)", "twin": "Classical Twin (tanh)", "qprism": "Q-PRISM (PQC)"}

N_VAL = 10_000
N_TRAIN_MAX = 110_000          # val = perm[110k:120k], test = perm[120k:] (10,831 molecules)
DATA_DIR = PROJECT_ROOT / "data" / "QM9"
OUT_DIR = PROJECT_ROOT / "results" / "experiment_c_full_qm9"
DRY_DIR = PROJECT_ROOT / "results" / "experiment_c_full_qm9_dryrun"
PAPER_FIG = PROJECT_ROOT / "paper" / "figures" / "figure_full_qm9_curves.png"


# Data: fixed val/test per seed, nested train prefixes

_RAW = None


def raw_qm9():
    global _RAW
    if _RAW is None:
        _RAW = QM9(root=str(DATA_DIR))
    return _RAW


def make_split(target, seed, n_train):
    """train = perm[:n_train]; val = perm[110k:120k]; test = perm[120k:]."""
    if n_train > N_TRAIN_MAX:
        raise ValueError(f"n_train must be ≤ {N_TRAIN_MAX}")
    raw = raw_qm9()
    # Same RNG usage as get_qm9_splits → identical train molecules for a given seed.
    torch.manual_seed(seed)
    perm = torch.randperm(len(raw))
    tr_idx = perm[:n_train]
    va_idx = perm[N_TRAIN_MAX:N_TRAIN_MAX + N_VAL]
    te_idx = perm[N_TRAIN_MAX + N_VAL:]
    tr = ProcessedQM9Subset([raw[i] for i in tr_idx], target_name=target)
    va = ProcessedQM9Subset([raw[i] for i in va_idx], target_name=target)
    te = ProcessedQM9Subset([raw[i] for i in te_idx], target_name=target)
    y = torch.cat([d.y for d in tr.data_list], dim=0)
    return tr, va, te, (y.mean().item(), y.std().item())


# Models

def make_model(kind):
    kw = dict(in_features=11, hidden_features=16, num_layers=2, n_q_layers=2, out_dim=1)
    if kind == "prism":
        return PRISMModelV2(**kw, head_type="siren", omega_0=30.0)
    if kind == "twin":
        return ClassicalTwinModelV2(**kw)
    if kind == "qprism":
        return QPRISMModelV2(**kw)
    raise ValueError(kind)


@torch.no_grad()
def evaluate(model, loader, mean, std, device):
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        pred = model(b.x, b.pos, b.edge_index, b.batch) * std + mean
        tot += torch.abs(pred - b.y).sum().item()
        n += b.num_graphs
    return tot / n


def train_run(kind, data, seed, protocol, device, ckpt_path=None, verbose=True):
    tr, va, te, (mean, std) = data
    bs = protocol["batch_size"]
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(tr, batch_size=bs, shuffle=True, generator=g)
    val_loader = DataLoader(va, batch_size=256, shuffle=False)
    test_loader = DataLoader(te, batch_size=256, shuffle=False)

    torch.manual_seed(seed)
    model = make_model(kind).to(device)
    params = count_parameters(model)
    epochs = protocol["epochs"]
    opt = torch.optim.Adam(model.parameters(), lr=protocol["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.MSELoss()

    curve = []
    best_val, best_ep, best_state = float("inf"), -1, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, nb = 0.0, 0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            loss = crit(model(b.x, b.pos, b.edge_index, b.batch), (b.y - mean) / std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            loss_sum += loss.item()
            nb += 1
        sched.step()
        val_mae = evaluate(model, val_loader, mean, std, device)
        curve.append({"epoch": ep, "train_loss": loss_sum / nb, "val_mae": val_mae,
                      "time_s": round(time.time() - t0, 1)})
        if val_mae < best_val:
            best_val, best_ep = val_mae, ep
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            mark = " *" if best_ep == ep else ""
            print(f"        ep {ep:3d}/{epochs}  loss {loss_sum/nb:.4f}  val {val_mae:.4f}{mark}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    test_last = evaluate(model, test_loader, mean, std, device)
    model.load_state_dict(best_state)
    test_best = evaluate(model, test_loader, mean, std, device)
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "mean": mean, "std": std,
                    "best_epoch": best_ep, "kind": kind, "seed": seed}, ckpt_path)

    # Convergence diagnostic: relative val improvement over the last 25% of epochs
    k = max(1, epochs // 4)
    v_then = min(c["val_mae"] for c in curve[:-k]) if len(curve) > k else curve[0]["val_mae"]
    v_now = min(c["val_mae"] for c in curve[-k:])
    late_gain = (v_then - v_now) / v_then if v_then > 0 else 0.0

    return {"test_mae_bestval": test_best, "test_mae_last": test_last,
            "best_val_mae": best_val, "best_epoch": best_ep,
            "late_val_gain_pct": round(100 * late_gain, 2),
            "params": params, "train_time_s": round(time.time() - t0, 1),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "curve": curve, **protocol}


# Storage (one file per target, parallel-safe)

def tpath(out_dir, target):
    return out_dir / f"{target}.json"


def load_target(out_dir, target):
    p = tpath(out_dir, target)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_target(out_dir, target, res):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = tpath(out_dir, target)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    tmp.replace(p)


def valid(rec, protocol):
    return rec is not None and all(rec.get(k) == v for k, v in protocol.items())


# Statistics & reporting

def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for i, idx in enumerate(order[::-1]):
        rank = m - i
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q


def paired(a, b):
    """a vs b; negative mean_diff -> a better."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    d = a - b
    out = {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
           "mean_diff": float(d.mean()), "rel_pct": float(d.mean() / b.mean() * 100)}
    if n >= 2:
        sd = d.std(ddof=1)
        se = sd / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        out.update({"ci95": [float(d.mean() - h), float(d.mean() + h)],
                    "cohens_d": float(d.mean() / sd) if sd > 0 else 0.0,
                    "ttest_p": float(stats.ttest_rel(a, b).pvalue) if sd > 0 else 1.0})
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = 1.0
    return out


def analyze(all_res, targets, n_values, metric):
    rows = []
    for t in targets:
        for n in n_values:
            nk = str(n)
            slot = all_res.get(t, {}).get(nk, {})
            P, T = slot.get("prism", {}), slot.get("twin", {})
            seeds = sorted(set(P) & set(T), key=int)
            if not seeds:
                continue
            a = [P[s][metric] for s in seeds]
            b = [T[s][metric] for s in seeds]
            row = {"target": t, "n_train": n, "seeds": seeds, "prism_vs_twin": paired(a, b),
                   "prism_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                   "twin_std": float(np.std(b, ddof=1)) if len(b) > 1 else 0.0}
            Q = slot.get("qprism", {})
            qs = sorted(set(P) & set(Q), key=int)
            if qs:
                row["prism_vs_qprism"] = paired([P[s][metric] for s in qs], [Q[s][metric] for s in qs])
            row["late_gain"] = {k: float(np.mean([slot[k][s]["late_val_gain_pct"] for s in slot.get(k, {})]))
                                for k in ("prism", "twin", "qprism") if slot.get(k)}
            row["best_epoch"] = {k: [slot[k][s]["best_epoch"] for s in sorted(slot[k], key=int)]
                                 for k in ("prism", "twin", "qprism") if slot.get(k)}
            rows.append(row)
    idx = [i for i, r in enumerate(rows) if "ttest_p" in r["prism_vs_twin"]]
    if idx:
        for i, q in zip(idx, bh_fdr([rows[i]["prism_vs_twin"]["ttest_p"] for i in idx])):
            rows[i]["prism_vs_twin"]["q_fdr"] = float(q)
    return rows


def star(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def print_report(rows, metric, epochs):
    print("\n" + "=" * 118)
    print(f"  EXPERIMENT C — FULL-QM9 MULTI-SEED  (metric: {metric}; paired t-test, BH-FDR over all rows)")
    print("=" * 118)
    print(f"  {'Target':<7}{'N':>8} │ {'PRISM':>16} │ {'Twin':>16} │ {'Δ%':>7} │ {'95% CI of Δ':>20} │ "
          f"{'d':>6} │ {'q':>9} │ {'n':>2}")
    print("  " + "─" * 116)
    for r in rows:
        s = r["prism_vs_twin"]
        ci = s.get("ci95")
        ci_s = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "—"
        q = s.get("q_fdr")
        q_s = f"{q:.4f}{star(q):<3}" if q is not None else "—"
        print(f"  {TARGET_DISPLAY[r['target']]:<7}{r['n_train']:>8,} │ {s['mean_a']:.4f} ± {r['prism_std']:.4f} │ "
              f"{s['mean_b']:.4f} ± {r['twin_std']:.4f} │ {-s['rel_pct']:>+6.1f}% │ {ci_s:>20} │ "
              f"{s.get('cohens_d', 0):>+6.2f} │ {q_s:>9} │ {s['n']:>2}")
        if "prism_vs_qprism" in r:
            sq = r["prism_vs_qprism"]
            print(f"  {'':<15} └ vs Q-PRISM: {sq['mean_b']:.4f}  Δ={-sq['rel_pct']:+.1f}%  "
                  f"p={sq.get('ttest_p', float('nan')):.4f} (n={sq['n']})")
    print("\n  Δ% > 0 ⇒ PRISM better than Twin.")
    print("  Convergence check (mean val-MAE gain over the last 25% of epochs; >2% -> likely undertrained):")
    for r in rows:
        flags = ", ".join(f"{k}={v:.1f}%{' (!)' if v > 2 else ''}" for k, v in r["late_gain"].items())
        be = ", ".join(f"{k}:{v}" for k, v in r["best_epoch"].items())
        print(f"     {TARGET_DISPLAY[r['target']]:<6} N={r['n_train']:<7,} {flags}   best epochs → {be}")
    print(f"  (If many runs pick best_epoch={epochs} and gain >2%, rerun with a larger --epochs.)")


def write_latex(rows, metric, out_dir):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{\textbf{Full-scale QM9, 5 seeds} (standard 110k/10k/10{,}831 split; test set fixed "
             r"across $N$ per seed). Test MAE (eV) at the best-validation epoch, mean $\pm$ std. "
             r"$\Delta$: relative improvement of PRISM over the Classical Twin with 95\% CI of the paired "
             r"difference. Paired $t$-test, Benjamini--Hochberg corrected ($^{*}q<0.05$, $^{**}q<0.01$, "
             r"$^{***}q<0.001$). Both models: 2{,}527 parameters.}",
             r"\label{tab:full_qm9_5seed}", r"\resizebox{\columnwidth}{!}{%",
             r"\begin{tabular}{lrccc}", r"\toprule",
             r"\textbf{Target} & $N$ & \textbf{PRISM} & \textbf{Classical Twin} & $\boldsymbol{\Delta}$ (95\% CI, eV) \\",
             r"\midrule"]
    last_t = None
    for r in rows:
        s = r["prism_vs_twin"]
        name = TARGET_DISPLAY[r["target"]] if r["target"] != last_t else ""
        last_t = r["target"]
        q = s.get("q_fdr")
        st = star(q) if q is not None else ""
        pr = f"{s['mean_a']:.3f} \\pm {r['prism_std']:.3f}"
        if s["mean_diff"] < 0:
            pr = f"\\mathbf{{{pr}}}"
        ci = s.get("ci95")
        ci_s = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—"
        n_s = f"{r['n_train']:,}".replace(",", "{,}")
        lines.append(f"{name} & ${n_s}$ & ${pr}$ & ${s['mean_b']:.3f} \\pm {r['twin_std']:.3f}$ & "
                     f"${-s['rel_pct']:+.1f}\\%^{{{st}}}$ \\; {ci_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    p = out_dir / "table_full_qm9.tex"
    p.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table -> {p}")


def make_figure(all_res, targets, n_plot, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"prism": "#2ca02c", "twin": "#d62728", "qprism": "#1f77b4"}
    fig, axes = plt.subplots(1, len(targets), figsize=(3.8 * len(targets), 3.2), squeeze=False)
    for ax, t in zip(axes[0], targets):
        slot = all_res.get(t, {}).get(str(n_plot), {})
        for k in ("prism", "twin", "qprism"):
            runs = slot.get(k, {})
            if not runs:
                continue
            L = min(len(r["curve"]) for r in runs.values())
            M = np.array([[c["val_mae"] for c in r["curve"][:L]] for r in runs.values()])
            x = np.arange(1, L + 1)
            ax.plot(x, M.mean(0), color=colors[k], label=f"{MODEL_LABELS[k]} (n={len(runs)})")
            ax.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=colors[k], alpha=0.2)
        ax.set_title(f"{TARGET_DISPLAY[t]} — N={n_plot:,}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"validation MAE")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "figure_full_qm9_curves.png"
    fig.savefig(p, dpi=200)
    try:
        PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, PAPER_FIG)
    except OSError:
        pass
    print(f"Figure -> {p}")


# Main

def main():
    ap = argparse.ArgumentParser(description="Experiment C: full-QM9 multi-seed PRISM vs Classical Twin")
    ap.add_argument("--targets", nargs="+", default=["homo", "lumo", "gap"],
                     choices=list(QM9_TARGET_INDICES.keys()))
    ap.add_argument("--n-values", nargs="+", type=int, default=[110_000])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=20, help="20 matches the existing SM table")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--include-qprism", action="store_true",
                    help="Also train Q-PRISM (~7 h per run at N=110k on CPU)")
    ap.add_argument("--qprism-seeds", nargs="+", type=int, default=None,
                    help="Seeds for Q-PRISM (default: same as --seeds)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps (classical models only)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads for this process")
    ap.add_argument("--metric", default="test_mae_bestval", choices=["test_mae_bestval", "test_mae_last"])
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="No per-epoch lines")
    ap.add_argument("--dry-run", action="store_true",
                    help="N=2000, 2 epochs, 3 seeds, HOMO → separate dry-run folder")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = OUT_DIR
    if args.dry_run:
        out_dir = DRY_DIR
        shutil.rmtree(DRY_DIR, ignore_errors=True)
        args.targets, args.n_values, args.seeds, args.epochs = ["homo"], [2000], [42, 43, 44], 2
        print("DRY RUN: HOMO, N=2000, 2 epochs, seeds 42-44 -> " + str(DRY_DIR))

    models = ["prism", "twin"] + (["qprism"] if args.include_qprism else [])
    q_seeds = args.qprism_seeds or args.seeds
    protocol = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}

    if not args.analyze_only:
        n_runs = sum(len(args.seeds) * 2 + (len(q_seeds) if args.include_qprism else 0)
                     for _ in args.targets for _ in args.n_values)
        print("=" * 90)
        print("Experiment C -- full-QM9 multi-seed")
        print(f"   targets {args.targets} | N {args.n_values} | seeds {args.seeds}")
        print(f"   models {models} | protocol {protocol} | device {args.device} | "
              f"threads {torch.get_num_threads()}")
        print(f"   split: train=perm[:N], val=perm[110000:120000], test=perm[120000:] (10,831)")
        print(f"   max runs: {n_runs}   output: {out_dir}")
        if args.include_qprism and max(args.n_values) >= 50_000:
            print("   Q-PRISM at N>=50k is many hours per run on CPU.")
        print("=" * 90)
        t_all = time.time()
        for target in args.targets:
            res = load_target(out_dir, target)
            for n in args.n_values:
                nk = str(n)
                todo = [(k, s) for k in models for s in (q_seeds if k == "qprism" else args.seeds)
                        if not valid(res.get(nk, {}).get(k, {}).get(str(s)), protocol)]
                print(f"\n{'='*70}\n{TARGET_DISPLAY[target]} | N={n:,} | {len(todo)} runs to do\n{'='*70}")
                if not todo:
                    continue
                for seed in sorted({s for _, s in todo}):
                    data = None
                    for kind in [k for k, s in todo if s == seed]:
                        if data is None:
                            t0 = time.time()
                            data = make_split(target, seed, n)
                            print(f"    data seed={seed}: train {len(data[0]):,} / val {len(data[1]):,} / "
                                  f"test {len(data[2]):,}  ({time.time()-t0:.1f}s)")
                        dev = torch.device("cpu") if kind == "qprism" else torch.device(args.device)
                        print(f"    {MODEL_LABELS[kind]:<22} seed={seed}", flush=True)
                        ck = None if args.no_checkpoints else \
                            out_dir / "checkpoints" / f"{target}_N{n}_{kind}_s{seed}.pt"
                        rec = train_run(kind, data, seed, protocol, dev, ck, verbose=not args.quiet)
                        res.setdefault(nk, {}).setdefault(kind, {})[str(seed)] = rec
                        save_target(out_dir, target, res)
                        print(f"    {MODEL_LABELS[kind]:<22} seed={seed}  test(best-val ep {rec['best_epoch']}) "
                              f"= {rec['test_mae_bestval']:.4f}  | last = {rec['test_mae_last']:.4f} "
                              f"{TARGET_UNITS[target]}  ({rec['train_time_s']/60:.1f} min)", flush=True)
                    del data
        print(f"\nWall time {(time.time()-t_all)/60:.1f} min")

    # Analysis over everything on disk for the requested targets / N
    all_res = {t: load_target(out_dir, t) for t in args.targets}
    for metric in ([args.metric, "test_mae_last"] if args.metric != "test_mae_last" else [args.metric]):
        rows = analyze(all_res, args.targets, args.n_values, metric)
        if rows:
            print_report(rows, metric, args.epochs)
    rows = analyze(all_res, args.targets, args.n_values, args.metric)
    if rows:
        (out_dir / "analysis.json").write_text(json.dumps(rows, indent=1))
        if not args.dry_run:
            write_latex(rows, args.metric, out_dir)
        make_figure(all_res, args.targets, max(args.n_values), out_dir)
    else:
        print("No paired results yet.")


if __name__ == "__main__":
    main()
