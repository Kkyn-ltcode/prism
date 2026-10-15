"""Experiment D: Extended omega_0 sensitivity sweep with paired Q-PRISM reference."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch_geometric.loader import DataLoader

from datasets.dataset_qm9 import get_qm9_splits, QM9_TARGET_INDICES
from models.egnn_baseline import count_parameters
from models.classical_twin import ClassicalTwinModelV2
from models.classical_trig_twin import PRISMModelV2
from models.eq_gnn import QPRISMModelV2

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV",
                "mu": "Debye", "alpha": "Bohr³", "u0": "eV"}
TARGET_DISPLAY = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap",
                  "mu": "Dipole", "alpha": "Polariz.", "u0": "U₀"}

DEFAULT_OMEGAS = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
DEFAULT_SEEDS = list(range(42, 52))        # 10 seeds for classical models
DEFAULT_QPRISM_SEEDS = list(range(42, 47))  # 5 seeds for Q-PRISM (expensive)
DEFAULT_TARGETS = ["homo", "lumo", "gap", "mu"]

SAVE_PATH = PROJECT_ROOT / "results" / "experiment_d_omega_sweep.json"
DRYRUN_PATH = PROJECT_ROOT / "results" / "experiment_d_omega_sweep_dryrun.json"
TABLE_PATH = PROJECT_ROOT / "results" / "experiment_d_omega_sweep_table.tex"
FIG_PATH = PROJECT_ROOT / "results" / "figure_omega_sweep.png"
PAPER_FIG_PATH = PROJECT_ROOT / "paper" / "figures" / "figure_omega_sweep.png"


# Models

def make_model(kind, omega=None):
    if kind == "siren":
        return PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                            n_q_layers=2, out_dim=1, head_type="siren", omega_0=omega)
    if kind == "twin":
        return ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                    n_q_layers=2, out_dim=1)
    if kind == "qprism":
        return QPRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                             n_q_layers=2, out_dim=1)
    raise ValueError(kind)


# Training (single protocol for every model)

def train_eval(model, target, seed, protocol):
    device = torch.device("cpu")
    data_dir = str(PROJECT_ROOT / "data" / "QM9")
    train_set, _, test_set, (mean, std) = get_qm9_splits(
        root=data_dir, target=target,
        num_train=protocol["num_train"], num_val=250, num_test=500, seed=seed,
    )
    train_loader = DataLoader(train_set, batch_size=protocol["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=protocol["batch_size"], shuffle=False)

    torch.manual_seed(seed)
    model = model.to(device)
    epochs = protocol["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            loss = criterion(pred, (batch.y - mean) / std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    total_abs, n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch) * std + mean
            total_abs += torch.abs(pred - batch.y).sum().item()
            n += batch.num_graphs
    return total_abs / n, round(time.time() - t0, 1)


# Cache helpers

def load_results(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"metadata": {}}


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)  # atomic: Ctrl+C can never leave a half-written JSON


def get_slot(results, target, kind, omega):
    """Return the dict {seed_str: record} for (target, kind, omega)."""
    t = results.setdefault(target, {})
    if kind == "siren":
        return t.setdefault("siren", {}).setdefault(str(float(omega)), {})
    return t.setdefault(kind, {})


def is_valid(rec, protocol):
    return (rec is not None
            and rec.get("epochs") == protocol["epochs"]
            and rec.get("lr") == protocol["lr"]
            and rec.get("batch_size") == protocol["batch_size"]
            and rec.get("num_train") == protocol["num_train"])


def run_one(results, path, target, kind, omega, seed, protocol):
    slot = get_slot(results, target, kind, omega)
    key = str(seed)
    label = f"{kind}" + (f" ω₀={omega:g}" if kind == "siren" else "")
    if is_valid(slot.get(key), protocol):
        print(f"    {label:<16} seed={seed}  cached {slot[key]['test_mae']:.4f}")
        return
    if key in slot:
        print(f"    {label:<16} seed={seed}  cached record has a DIFFERENT protocol -> recomputing")
    model = make_model(kind, omega)
    params = count_parameters(model)
    mae, elapsed = train_eval(model, target, seed, protocol)
    slot[key] = {"test_mae": mae, "params": params, "time_s": elapsed, **protocol}
    print(f"    {label:<16} seed={seed}  MAE={mae:.4f} {TARGET_UNITS[target]}  ({elapsed}s)")
    save_results(results, path)


# Statistics

def paired_arrays(slot_a, slot_b):
    seeds = sorted(set(slot_a) & set(slot_b), key=int)
    a = np.array([slot_a[s]["test_mae"] for s in seeds])
    b = np.array([slot_b[s]["test_mae"] for s in seeds])
    return seeds, a, b


def paired_test(a, b):
    """Test of a vs b (negative mean_diff = a better). Returns dict."""
    n = len(a)
    if n < 3:
        return None
    diff = a - b
    sd = np.std(diff, ddof=1)
    d = float(np.mean(diff) / sd) if sd > 0 else 0.0
    _, t_p = stats.ttest_rel(a, b)
    try:
        _, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_p = 1.0
    return {"n": n, "mean_diff": float(np.mean(diff)),
            "rel_change_pct": float(np.mean(diff) / np.mean(b) * 100),
            "cohens_d": d, "ttest_p": float(t_p), "wilcoxon_p": float(w_p)}


def bh_fdr(pvals):
    """Benjamini–Hochberg adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = m - rank_from_end
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    return q


def mean_std(slot, seeds=None):
    keys = seeds if seeds is not None else sorted(slot, key=int)
    v = np.array([slot[s]["test_mae"] for s in keys if s in slot])
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def analyze(results, targets, omegas):
    """Compute paired tests (PRISM(ω₀) vs twin / Q-PRISM) with BH-FDR over the whole family."""
    rows = []
    for target in targets:
        t = results.get(target, {})
        twin, qp = t.get("twin", {}), t.get("qprism", {})
        for om in omegas:
            sl = t.get("siren", {}).get(str(float(om)), {})
            if not sl:
                continue
            m, s, n = mean_std(sl)
            row = {"target": target, "omega": om, "mean": m, "std": s, "n": n,
                   "vs_twin": None, "vs_qprism": None}
            if twin:
                _, a, b = paired_arrays(sl, twin)
                row["vs_twin"] = paired_test(a, b)
            if qp:
                _, a, b = paired_arrays(sl, qp)
                row["vs_qprism"] = paired_test(a, b)
            rows.append(row)
    for field in ("vs_twin", "vs_qprism"):
        idx = [i for i, r in enumerate(rows) if r[field]]
        if idx:
            q = bh_fdr([rows[i][field]["ttest_p"] for i in idx])
            for i, qi in zip(idx, q):
                rows[i][field]["q_fdr"] = float(qi)
    return rows


# Reporting

def sig_marker(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def print_report(results, rows, targets, omegas):
    print("\n" + "=" * 110)
    print("  EXPERIMENT D -- omega_0 SENSITIVITY (paired, BH-FDR corrected)")
    print("=" * 110)
    for target in targets:
        t = results.get(target, {})
        tm, ts, tn = mean_std(t.get("twin", {}))
        qm, qs, qn = mean_std(t.get("qprism", {}))
        print(f"\n  {TARGET_DISPLAY[target]} ({TARGET_UNITS[target]})")
        if tm is not None:
            print(f"    Classical Twin : {tm:.4f} ± {ts:.4f}  (n={tn})")
        if qm is not None:
            print(f"    Q-PRISM (PQC)  : {qm:.4f} ± {qs:.4f}  (n={qn})")
        print(f"    {'ω₀':>6} │ {'PRISM MAE':>16} │ {'Δ vs Twin':>10} {'q':>8} │ {'Δ vs Q-PRISM':>12} {'q':>8}")
        print(f"    {'─'*6}─┼─{'─'*16}─┼─{'─'*19}─┼─{'─'*21}")
        for r in [r for r in rows if r["target"] == target]:
            vt, vq = r["vs_twin"], r["vs_qprism"]
            ts_ = f"{vt['rel_change_pct']:+8.1f}% {vt['q_fdr']:>7.4f}{sig_marker(vt['q_fdr']):<3}" if vt else "         —"
            qs_ = f"{vq['rel_change_pct']:+10.1f}% {vq['q_fdr']:>7.4f}{sig_marker(vq['q_fdr']):<3}" if vq else "           —"
            mark = "  ← default" if r["omega"] == 30.0 else ""
            print(f"    {r['omega']:>6g} │ {r['mean']:.4f} ± {r['std']:.4f} │ {ts_} │ {qs_}{mark}")

        sub = [r for r in rows if r["target"] == target and r["mean"] is not None]
        if sub:
            best = min(sub, key=lambda r: r["mean"])
            print(f"    Best ω₀ = {best['omega']:g} (MAE {best['mean']:.4f})")
            band_t = [r["omega"] for r in sub if r["vs_twin"] and r["vs_twin"]["mean_diff"] < 0
                      and r["vs_twin"]["q_fdr"] < 0.05]
            band_q = [r["omega"] for r in sub if r["vs_qprism"] and r["vs_qprism"]["mean_diff"] < 0
                      and r["vs_qprism"]["q_fdr"] < 0.05]
            print(f"    ω₀ with PRISM significantly BETTER than Twin    (q<0.05): {band_t}")
            if qm is not None:
                print(f"    ω₀ with PRISM significantly BETTER than Q-PRISM (q<0.05): {band_q}")
            worse_t = [r["omega"] for r in sub if r["vs_twin"] and r["vs_twin"]["mean_diff"] > 0
                       and r["vs_twin"]["q_fdr"] < 0.05]
            if worse_t:
                print(f"    ω₀ with PRISM significantly WORSE than Twin  (q<0.05): {worse_t}")


def write_latex(results, rows, targets, omegas):
    cols = "l" + "c" * (len(omegas) + 2)
    hdr = " & ".join([r"\textbf{Target}", r"\textbf{Twin}", r"\textbf{Q-PRISM}"]
                     + [rf"$\omega_0={o:g}$" for o in omegas])
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{\textbf{Extended SIREN $\omega_0$ sensitivity with paired Q-PRISM reference} "
             r"($N=1{,}000$, 2{,}527 params). Test MAE (eV; Dipole in Debye), mean $\pm$ std. "
             r"Markers: paired $t$-test vs.\ Classical Twin, Benjamini--Hochberg corrected over the "
             r"whole $\omega_0\times$target family ($^{*}q<0.05$, $^{**}q<0.01$, $^{***}q<0.001$). "
             r"$^{\ddagger}$: PRISM significantly better than Q-PRISM ($q<0.05$, paired). "
             r"Best PRISM value per row in \textbf{bold}.}",
             r"\label{tab:omega_sweep_extended}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             hdr + r" \\", r"\midrule"]
    for target in targets:
        t = results.get(target, {})
        tm, ts, _ = mean_std(t.get("twin", {}))
        qm, qs, _ = mean_std(t.get("qprism", {}))
        trows = {r["omega"]: r for r in rows if r["target"] == target}
        best = min((r["mean"] for r in trows.values()), default=None)
        cells = [TARGET_DISPLAY[target],
                 f"${tm:.3f} \\pm {ts:.3f}$" if tm is not None else "—",
                 f"${qm:.3f} \\pm {qs:.3f}$" if qm is not None else "—"]
        for om in omegas:
            r = trows.get(om)
            if not r:
                cells.append("—")
                continue
            txt = f"{r['mean']:.3f} \\pm {r['std']:.3f}"
            if best is not None and abs(r["mean"] - best) < 1e-12:
                txt = f"\\mathbf{{{txt}}}"
            vt, vq = r["vs_twin"], r["vs_qprism"]
            if vt and vt["mean_diff"] < 0:
                s = sig_marker(vt["q_fdr"])
                if s:
                    txt += f"^{{{s}}}"
            if vq and vq["mean_diff"] < 0 and vq["q_fdr"] < 0.05:
                txt += r"^{\ddagger}"
            cells.append(f"${txt}$")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    TABLE_PATH.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table -> {TABLE_PATH}")


def make_figure(results, rows, targets, omegas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.3), squeeze=False)
    for ax, target in zip(axes[0], targets):
        t = results.get(target, {})
        sub = sorted([r for r in rows if r["target"] == target], key=lambda r: r["omega"])
        if not sub:
            continue
        x = [r["omega"] for r in sub]
        y = np.array([r["mean"] for r in sub])
        e = np.array([r["std"] for r in sub])
        ax.errorbar(x, y, yerr=e, fmt="o-", color="#d62728", capsize=3, lw=1.6,
                    label="PRISM (SIREN)")
        tm, ts, _ = mean_std(t.get("twin", {}))
        if tm is not None:
            ax.axhline(tm, color="#1f77b4", ls="--", lw=1.4, label="Classical Twin")
            ax.axhspan(tm - ts, tm + ts, color="#1f77b4", alpha=0.12)
        qm, qs, _ = mean_std(t.get("qprism", {}))
        if qm is not None:
            ax.axhline(qm, color="#2ca02c", ls="-.", lw=1.4, label="Q-PRISM (PQC)")
            ax.axhspan(qm - qs, qm + qs, color="#2ca02c", alpha=0.12)
        ax.axvline(30, color="gray", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\omega_0$")
        ax.set_title(f"{TARGET_DISPLAY[target]} [{TARGET_UNITS[target]}]", fontsize=10)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("Test MAE")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    try:
        PAPER_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIG_PATH, PAPER_FIG_PATH)
    except OSError:
        pass
    print(f"Figure -> {FIG_PATH}  (copy: {PAPER_FIG_PATH})")


# Main

def main():
    ap = argparse.ArgumentParser(description="Experiment D: extended ω₀ sweep + paired Q-PRISM")
    ap.add_argument("--omega-values", nargs="+", type=float, default=DEFAULT_OMEGAS)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    choices=list(QM9_TARGET_INDICES.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="Seeds for Twin and PRISM")
    ap.add_argument("--qprism-seeds", nargs="+", type=int, default=DEFAULT_QPRISM_SEEDS,
                    help="Seeds for Q-PRISM (subset of --seeds recommended; pairing uses the overlap)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-qprism", action="store_true", help="Run only Twin + PRISM")
    ap.add_argument("--qprism-only", action="store_true", help="Run only the Q-PRISM reference")
    ap.add_argument("--analyze-only", action="store_true", help="No training; rebuild report/table/figure")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="2 epochs, ω₀∈{10,30}, 3 seeds, HOMO; writes to a SEPARATE dry-run file")
    args = ap.parse_args()

    if args.skip_qprism and args.qprism_only:
        ap.error("--skip-qprism and --qprism-only are mutually exclusive")

    path = SAVE_PATH
    if args.dry_run:
        path = DRYRUN_PATH
        if DRYRUN_PATH.exists():
            DRYRUN_PATH.unlink()  # dry-run always starts clean
        args.epochs, args.omega_values = 2, [10.0, 30.0]
        args.seeds, args.qprism_seeds, args.targets = [42, 43, 44], [42, 43, 44], ["homo"]
        print("DRY RUN: 2 epochs, omega_0=[10,30], seeds 42-44, HOMO -> writing to dry-run file only")

    protocol = {"epochs": args.epochs, "lr": args.lr,
                "batch_size": args.batch_size, "num_train": args.num_train}
    results = load_results(path)
    results["metadata"] = {
        "description": "Experiment D: extended omega_0 sweep with paired Q-PRISM reference",
        "omega_values": args.omega_values, "targets": args.targets,
        "seeds": args.seeds, "qprism_seeds": args.qprism_seeds,
        "protocol": protocol, "params": 2527,
        "qpu_backend": "default.qubit + backprop",
    }

    if not args.analyze_only:
        total = (0 if args.qprism_only else len(args.targets) * len(args.seeds) * (1 + len(args.omega_values))) \
            + (0 if args.skip_qprism else len(args.targets) * len(args.qprism_seeds))
        print("=" * 80)
        print("Experiment D: omega_0 sweep")
        print(f"   omega_0: {args.omega_values}\n   targets: {args.targets}")
        print(f"   seeds (Twin/PRISM): {args.seeds}\n   seeds (Q-PRISM): "
              f"{[] if args.skip_qprism else args.qprism_seeds}")
        print(f"   protocol: {protocol}\n   max runs: {total}   output: {path}")
        print("=" * 80)

        t_start = time.time()
        for target in args.targets:
            print(f"\n{'='*70}\n{TARGET_DISPLAY[target]} ({TARGET_UNITS[target]})\n{'='*70}")
            if not args.qprism_only:
                for seed in args.seeds:
                    run_one(results, path, target, "twin", None, seed, protocol)
                for om in args.omega_values:
                    for seed in args.seeds:
                        run_one(results, path, target, "siren", om, seed, protocol)
            if not args.skip_qprism:
                for seed in args.qprism_seeds:
                    run_one(results, path, target, "qprism", None, seed, protocol)
        print(f"\nTraining wall time: {(time.time() - t_start)/60:.1f} min")

    # Analysis uses only records matching the current protocol
    rows = analyze(results, args.targets, args.omega_values)
    print_report(results, rows, args.targets, args.omega_values)
    results["analysis"] = rows
    save_results(results, path)

    if not args.dry_run:
        write_latex(results, rows, args.targets, args.omega_values)
        if not args.no_figure:
            make_figure(results, rows, args.targets, args.omega_values)
    print(f"\nResults -> {path}")


if __name__ == "__main__":
    main()
