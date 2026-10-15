"""Experiment B: Deeper circuit scaling (L=1, 2, 4, 8, 16 PQC layers)."""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import sys
from pathlib import Path
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.loader import DataLoader
from datasets.dataset_qm9 import get_qm9_splits
from models.eq_gnn import QPRISMModelV2
from models.classical_trig_twin import PRISMModelV2
from models.classical_twin import ClassicalTwinModelV2
from models.egnn_baseline import count_parameters

# Configuration

DEPTH_VALUES = [1, 2, 4, 8, 16]

TARGET_UNITS = {"homo": "eV", "lumo": "eV", "gap": "eV", "mu": "Debye"}
TARGET_DISPLAY = {"homo": "HOMO", "lumo": "LUMO", "gap": "Gap", "mu": "Dipole"}

SAVE_PATH = PROJECT_ROOT / "results" / "experiment_b_depth_scaling.json"
DRYRUN_PATH = PROJECT_ROOT / "results" / "experiment_b_depth_scaling_dryrun.json"
TABLE_PATH = PROJECT_ROOT / "results" / "experiment_b_depth_scaling_table.tex"
PAPER_TABLE_PATH = PROJECT_ROOT / "paper" / "experiment_b_depth_scaling_table.tex"
FIG_PATH = PROJECT_ROOT / "results" / "figure_depth_scaling.png"
PAPER_FIG_PATH = PROJECT_ROOT / "paper" / "figures" / "figure_depth_scaling.png"

DEFAULT_PROTOCOL = {
    "num_train": 1000,
    "num_val": 250,
    "num_test": 500,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 15,
}


# Model factories

def make_model(kind, n_q_layers=2):
    """Create a model. For PRISM/Twin, n_q_layers is accepted but ignored by the head."""
    if kind == "qprism":
        return QPRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                             n_q_layers=n_q_layers, out_dim=1)
    if kind == "prism":
        return PRISMModelV2(in_features=11, hidden_features=16, num_layers=2,
                            n_q_layers=n_q_layers, out_dim=1,
                            head_type="siren", omega_0=30.0)
    if kind == "twin":
        return ClassicalTwinModelV2(in_features=11, hidden_features=16, num_layers=2,
                                    n_q_layers=n_q_layers, out_dim=1)
    raise ValueError(f"Unknown model kind: {kind}")


# Training (shared protocol for all models)

def train_eval(model, target, seed, protocol):
    """Train one model and return test MAE in original units."""
    device = torch.device("cpu")
    data_dir = str(PROJECT_ROOT / "data" / "QM9")
    train_set, val_set, test_set, (mean, std) = get_qm9_splits(
        root=data_dir, target=target,
        num_train=protocol["num_train"],
        num_val=protocol["num_val"],
        num_test=protocol["num_test"],
        seed=seed,
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
    for ep in range(epochs):
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
    wall_time = time.time() - t0

    # Evaluate test MAE in original units
    model.eval()
    total_abs, n = 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            pred_orig = pred * std + mean
            total_abs += (pred_orig - batch.y).abs().sum().item()
            n += batch.y.numel()

    test_mae = total_abs / n
    return test_mae, wall_time


# Statistics (paired, with BH-FDR)

def compute_paired_stats(maes_a, maes_b):
    """Paired comparison: a vs b (same seeds).  Returns dict of stats."""
    a, b = np.array(maes_a), np.array(maes_b)
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    try:
        _, wilcoxon_p = stats.wilcoxon(a, b)
    except ValueError:
        wilcoxon_p = 1.0
    _, ttest_p = stats.ttest_rel(a, b)

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": float(cohens_d),
        "wilcoxon_p": float(wilcoxon_p),
        "ttest_p": float(ttest_p),
        "n": n,
    }


def bh_fdr(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction.  Returns list of (original_p, adjusted_p, significant)."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    for rank_minus1 in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_minus1]
        rank = rank_minus1 + 1
        adj = min(prev, p * n / rank)
        adjusted[orig_idx] = adj
        prev = adj
    return [(pvalues[i], adjusted[i], adjusted[i] < alpha) for i in range(n)]


# Reporting: LaTeX table

def generate_latex_table(results, targets):
    """Generate LaTeX table for the depth scaling experiment."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{PQC Depth Scaling: Test MAE for Q-PRISM at varying circuit depth $L$",
        r"  vs.\ fixed-depth PRISM and Classical Twin baselines ($N{=}1{,}000$, 5 seeds).}",
        r"\label{tab:depth_scaling}",
        r"\small",
    ]

    # One sub-table per target
    for target in targets:
        unit = TARGET_UNITS.get(target, "")
        display = TARGET_DISPLAY.get(target, target)

        lines.append(r"\begin{subtable}{\linewidth}")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{display} ({unit})}}")

        col_spec = r"r" + r"c" * 3 + r"cc"
        lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
        lines.append(r"\toprule")
        lines.append(r"$L$ & Q-PRISM & PRISM & Twin & $\Delta_\%$ & $p$ \\")
        lines.append(r"\midrule")

        for L in DEPTH_VALUES:
            Lkey = str(L)
            if Lkey not in results or target not in results[Lkey]:
                continue
            td = results[Lkey][target]
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            tw = td.get("twin", {})
            st = td.get("stats_qprism_vs_prism", {})

            if not qp:
                continue

            qp_str = f"${qp['mean_mae']:.4f}\\pm{qp['std_mae']:.4f}$"
            pr_str = f"${pr['mean_mae']:.4f}\\pm{pr['std_mae']:.4f}$" if pr else "---"
            tw_str = f"${tw['mean_mae']:.4f}\\pm{tw['std_mae']:.4f}$" if tw else "---"

            # Relative diff: (PRISM - Q-PRISM) / PRISM × 100  (positive = Q-PRISM better)
            if pr and pr["mean_mae"] > 0:
                delta_pct = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                delta_str = f"${delta_pct:+.1f}\\%$"
            else:
                delta_str = "---"

            pval = st.get("ttest_p", 1.0)
            if pval < 0.001:
                p_str = "$<0.001$"
            elif pval < 1.0:
                p_str = f"${pval:.3f}$"
            else:
                p_str = "---"

            nparams = qp.get("params", "")
            lines.append(
                rf"{L} & {qp_str} & {pr_str} & {tw_str} & {delta_str} & {p_str} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{subtable}")
        lines.append(r"\vspace{1em}")
        lines.append("")

    lines.append(r"\end{table}")
    return "\n".join(lines)


# Reporting: Figure

def generate_figure(results, targets, fig_path):
    """Generate depth-scaling figure: MAE vs L for each target."""
    import matplotlib.pyplot as plt

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 4.5), squeeze=False)

    for col, target in enumerate(targets):
        ax = axes[0, col]
        display = TARGET_DISPLAY.get(target, target)
        unit = TARGET_UNITS.get(target, "")

        # Collect Q-PRISM across depths
        L_vals, qp_means, qp_stds = [], [], []
        pr_mean, pr_std = None, None
        tw_mean, tw_std = None, None

        for L in DEPTH_VALUES:
            Lkey = str(L)
            if Lkey not in results or target not in results[Lkey]:
                continue
            td = results[Lkey][target]

            qp = td.get("qprism")
            if qp:
                L_vals.append(L)
                qp_means.append(qp["mean_mae"])
                qp_stds.append(qp["std_mae"])

            # Classical baselines are the same at every L; grab from any
            pr = td.get("prism")
            if pr and pr_mean is None:
                pr_mean = pr["mean_mae"]
                pr_std = pr["std_mae"]
            tw = td.get("twin")
            if tw and tw_mean is None:
                tw_mean = tw["mean_mae"]
                tw_std = tw["std_mae"]

        if not L_vals:
            ax.set_visible(False)
            continue

        L_vals = np.array(L_vals)
        qp_means = np.array(qp_means)
        qp_stds = np.array(qp_stds)

        # Q-PRISM curve with error bars
        ax.errorbar(L_vals, qp_means, yerr=qp_stds, marker="o", capsize=3,
                     color="#1f77b4", linewidth=2, label="Q-PRISM (PQC)")

        # PRISM horizontal band
        if pr_mean is not None:
            ax.axhline(pr_mean, color="#2ca02c", linewidth=1.5, linestyle="--", label="PRISM")
            ax.axhspan(pr_mean - pr_std, pr_mean + pr_std, color="#2ca02c", alpha=0.1)

        # Twin horizontal band
        if tw_mean is not None:
            ax.axhline(tw_mean, color="#d62728", linewidth=1.5, linestyle=":", label="Twin")
            ax.axhspan(tw_mean - tw_std, tw_mean + tw_std, color="#d62728", alpha=0.1)

        ax.set_xlabel("PQC Depth $L$ (n_q_layers)", fontsize=11)
        ax.set_ylabel(f"Test MAE ({unit})", fontsize=11)
        ax.set_title(f"{display}", fontsize=12, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEPTH_VALUES)
        ax.set_xticklabels([str(v) for v in DEPTH_VALUES])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Add a parameter annotation along the bottom
    param_info = []
    for L in DEPTH_VALUES:
        Lkey = str(L)
        for target in targets:
            if Lkey in results and target in results[Lkey]:
                qp = results[Lkey][target].get("qprism")
                if qp:
                    param_info.append((L, qp["params"]))
                    break
    if param_info:
        param_text = "  |  ".join(f"L={L}: {p:,}p" for L, p in param_info)
        fig.text(0.5, 0.01, f"Q-PRISM params: {param_text}",
                 ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {fig_path}")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: PQC Depth Scaling (L=1,2,4,8,16)")
    parser.add_argument("--targets", nargs="+", default=list(TARGET_DISPLAY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--depths", nargs="+", type=int, default=DEPTH_VALUES,
                        help="PQC depth values to test")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-qprism", action="store_true",
                        help="Skip Q-PRISM (only train classical baselines)")
    parser.add_argument("--dry-run", action="store_true",
                        help="2-epoch smoke test; results saved to separate file")
    args = parser.parse_args()

    protocol = dict(DEFAULT_PROTOCOL)
    protocol["epochs"] = args.epochs
    protocol["num_train"] = args.num_train
    protocol["batch_size"] = args.batch_size

    is_dry = args.dry_run
    if is_dry:
        protocol["epochs"] = 2
        protocol["num_train"] = 100
        protocol["num_test"] = 100
        save_path = DRYRUN_PATH
    else:
        save_path = SAVE_PATH

    depths = sorted(args.depths)
    targets = args.targets
    seeds = args.seeds

    # ── Dry-run header ──
    if is_dry:
        print("=" * 80)
        print("  DRY RUN — Smoke test (2 epochs, N=100)")
        print("=" * 80)
        print(f"  Depths:  {depths}")
        print(f"  Targets: {targets}")
        print(f"  Seeds:   {seeds}")
        print()

        print("  Parameter counts by depth:")
        for L in depths:
            qp = make_model("qprism", n_q_layers=L)
            pr = make_model("prism", n_q_layers=L)
            tw = make_model("twin", n_q_layers=L)
            q_head = sum(p.numel() for n, p in qp.named_parameters() if "head_" in n)
            print(f"    L={L:>2}: Q-PRISM={count_parameters(qp):>6,}  "
                  f"(head={q_head:>5,})  |  "
                  f"PRISM={count_parameters(pr):>6,}  |  "
                  f"Twin={count_parameters(tw):>6,}")
        print()

    # Load checkpoint
    results = {}
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)
        print(f"Loaded checkpoint: {save_path}")

    total_start = time.time()

    # Phase 1: Classical baselines (depth-independent — train once)
    # PRISM and Twin do NOT change with n_q_layers.  We store their
    # results under every L key for convenience, but only train once.

    classical_cache = {}  # {(kind, target, seed): mae}

    for kind in ["prism", "twin"]:
        for target in targets:
            print(f"\n  -- {kind.upper()} x {TARGET_DISPLAY.get(target, target)} --")
            for seed in seeds:
                cache_key = (kind, target, seed)

                # Check if already cached in any depth's results
                cached_mae = None
                for L in depths:
                    Lkey = str(L)
                    existing = (results.get(Lkey, {}).get(target, {}).get(kind, {})
                                .get("maes_by_seed", {}))
                    if str(seed) in existing:
                        cached_mae = existing[str(seed)]
                        break

                if cached_mae is not None:
                    classical_cache[cache_key] = cached_mae
                    print(f"    {kind} seed={seed} (cached: {cached_mae:.4f})")
                    continue

                torch.manual_seed(seed)
                model = make_model(kind, n_q_layers=2)  # n_q_layers irrelevant
                mae, wall = train_eval(model, target, seed, protocol)
                classical_cache[cache_key] = mae
                print(f"    {kind} seed={seed} -> MAE={mae:.4f} ({wall:.0f}s)")

    # Phase 2: Q-PRISM at each depth

    for L in depths:
        Lkey = str(L)
        if Lkey not in results:
            results[Lkey] = {}

        print(f"\n{'='*80}")
        print(f"  DEPTH L={L}")
        print(f"{'='*80}")

        for target in targets:
            if target not in results[Lkey]:
                results[Lkey][target] = {}

            # Populate classical baselines for this depth
            for kind in ["prism", "twin"]:
                seed_maes = {}
                for seed in seeds:
                    cache_key = (kind, target, seed)
                    if cache_key in classical_cache:
                        seed_maes[str(seed)] = classical_cache[cache_key]

                maes_list = list(seed_maes.values())
                if maes_list:
                    model_for_count = make_model(kind, n_q_layers=L)
                    results[Lkey][target][kind] = {
                        "mean_mae": float(np.mean(maes_list)),
                        "std_mae": float(np.std(maes_list, ddof=1)) if len(maes_list) > 1 else 0.0,
                        "maes": maes_list,
                        "maes_by_seed": seed_maes,
                        "params": count_parameters(model_for_count),
                        "note": "depth-independent; trained once at n_q_layers=2",
                    }

            # Q-PRISM at this depth
            if not args.skip_qprism:
                existing_qp = results[Lkey][target].get("qprism", {})
                existing_seeds = existing_qp.get("maes_by_seed", {})

                # Skip if all seeds already done
                missing_seeds = [s for s in seeds if str(s) not in existing_seeds]
                if not missing_seeds and len(existing_seeds) >= len(seeds):
                    print(f"  Q-PRISM L={L} x {TARGET_DISPLAY.get(target, target)} "
                          f"already done (MAE={existing_qp['mean_mae']:.4f})")
                else:
                    seed_maes = dict(existing_seeds)  # preserve existing
                    param_count = existing_qp.get("params", 0)
                    total_time = existing_qp.get("total_train_time_s", 0.0)

                    for seed in seeds:
                        if str(seed) in seed_maes:
                            print(f"    Q-PRISM L={L} seed={seed} (cached)")
                            continue

                        print(f"    Q-PRISM L={L} seed={seed} "
                              f"({TARGET_DISPLAY.get(target, target)})...", end="", flush=True)
                        torch.manual_seed(seed)
                        model = make_model("qprism", n_q_layers=L)
                        if param_count == 0:
                            param_count = count_parameters(model)
                        mae, wall = train_eval(model, target, seed, protocol)
                        seed_maes[str(seed)] = mae
                        total_time += wall
                        print(f" MAE={mae:.4f} ({wall:.0f}s)")

                    maes_list = [seed_maes[str(s)] for s in seeds if str(s) in seed_maes]
                    results[Lkey][target]["qprism"] = {
                        "mean_mae": float(np.mean(maes_list)),
                        "std_mae": float(np.std(maes_list, ddof=1)) if len(maes_list) > 1 else 0.0,
                        "maes": maes_list,
                        "maes_by_seed": seed_maes,
                        "params": param_count,
                        "total_train_time_s": round(total_time, 1),
                    }

            # Paired stats: Q-PRISM vs PRISM
            qp_data = results[Lkey][target].get("qprism", {})
            pr_data = results[Lkey][target].get("prism", {})
            if qp_data and pr_data:
                paired_seeds = sorted(set(qp_data.get("maes_by_seed", {}).keys())
                                      & set(pr_data.get("maes_by_seed", {}).keys()))
                if len(paired_seeds) >= 3:
                    qp_vals = [qp_data["maes_by_seed"][s] for s in paired_seeds]
                    pr_vals = [pr_data["maes_by_seed"][s] for s in paired_seeds]
                    results[Lkey][target]["stats_qprism_vs_prism"] = compute_paired_stats(
                        qp_vals, pr_vals)

            # Paired stats: Q-PRISM vs Twin
            tw_data = results[Lkey][target].get("twin", {})
            if qp_data and tw_data:
                paired_seeds = sorted(set(qp_data.get("maes_by_seed", {}).keys())
                                      & set(tw_data.get("maes_by_seed", {}).keys()))
                if len(paired_seeds) >= 3:
                    qp_vals = [qp_data["maes_by_seed"][s] for s in paired_seeds]
                    tw_vals = [tw_data["maes_by_seed"][s] for s in paired_seeds]
                    results[Lkey][target]["stats_qprism_vs_twin"] = compute_paired_stats(
                        qp_vals, tw_vals)

            # Checkpoint after each target at each depth
            with open(save_path, "w") as f:
                json.dump(results, f, indent=2)

    total_wall = time.time() - total_start

    # BH-FDR correction across all comparisons

    all_pvals = []  # [(Lkey, target, comparison_key), p]
    for Lkey in results:
        for target in targets:
            td = results.get(Lkey, {}).get(target, {})
            for stat_key in ["stats_qprism_vs_prism", "stats_qprism_vs_twin"]:
                st = td.get(stat_key)
                if st:
                    all_pvals.append(((Lkey, target, stat_key), st["ttest_p"]))

    if all_pvals:
        raw_ps = [p for _, p in all_pvals]
        fdr_results = bh_fdr(raw_ps)
        for i, ((Lkey, target, stat_key), _) in enumerate(all_pvals):
            orig_p, adj_p, sig = fdr_results[i]
            results[Lkey][target][stat_key]["fdr_adjusted_p"] = adj_p
            results[Lkey][target][stat_key]["fdr_significant_005"] = sig

    # Final save
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary Report

    print(f"\n\n{'='*110}")
    print(f"  EXPERIMENT B: PQC DEPTH SCALING RESULTS")
    print(f"  Total wall time: {total_wall/60:.1f} min")
    print(f"{'='*110}")

    for target in targets:
        display = TARGET_DISPLAY.get(target, target)
        unit = TARGET_UNITS.get(target, "")
        print(f"\n  Target: {display} ({unit})")
        print(f"  {'L':>3}  {'Q-PRISM':>14}  {'PRISM':>14}  {'Twin':>14}  "
              f"{'Δ(Q-P)%':>8}  {'p(Q-P)':>10}  {'d(Q-P)':>7}  {'params':>7}")
        print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*10}  {'─'*7}  {'─'*7}")

        for L in depths:
            Lkey = str(L)
            td = results.get(Lkey, {}).get(target, {})
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            tw = td.get("twin", {})
            st = td.get("stats_qprism_vs_prism", {})

            qp_str = f"{qp['mean_mae']:.4f}±{qp['std_mae']:.4f}" if qp else "—"
            pr_str = f"{pr['mean_mae']:.4f}±{pr['std_mae']:.4f}" if pr else "—"
            tw_str = f"{tw['mean_mae']:.4f}±{tw['std_mae']:.4f}" if tw else "—"

            if qp and pr and pr["mean_mae"] > 0:
                delta = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "—"

            pval = st.get("fdr_adjusted_p", st.get("ttest_p", 1.0))
            sig_marker = ""
            if pval < 0.001:
                sig_marker = "***"
            elif pval < 0.01:
                sig_marker = "**"
            elif pval < 0.05:
                sig_marker = "*"
            p_str = f"{pval:.4f}{sig_marker}" if st else "—"
            d_str = f"{st.get('cohens_d', 0):+.2f}" if st else "—"
            params_str = f"{qp.get('params', 0):,}" if qp else "—"

            print(f"  {L:>3}  {qp_str:>14}  {pr_str:>14}  {tw_str:>14}  "
                  f"{delta_str:>8}  {p_str:>10}  {d_str:>7}  {params_str:>7}")

    # Key Finding
    print(f"\n  Key Finding: Does Q-PRISM advantage grow with PQC depth?")
    for target in targets:
        display = TARGET_DISPLAY.get(target, target)
        first_delta, last_delta = None, None
        for L in depths:
            Lkey = str(L)
            td = results.get(Lkey, {}).get(target, {})
            qp = td.get("qprism", {})
            pr = td.get("prism", {})
            if qp and pr and pr["mean_mae"] > 0:
                delta = (pr["mean_mae"] - qp["mean_mae"]) / pr["mean_mae"] * 100
                if first_delta is None:
                    first_delta = (L, delta)
                last_delta = (L, delta)
        if first_delta and last_delta:
            diff = last_delta[1] - first_delta[1]
            trend = ("GROWS" if diff > 1.0 else
                     "SHRINKS" if diff < -1.0 else
                     "FLAT")
            print(f"    {display:6s}: L={first_delta[0]}->{last_delta[0]}  "
                  f"delta={first_delta[1]:+.1f}% -> {last_delta[1]:+.1f}%  ->  {trend}")

    # Generate outputs
    try:
        latex = generate_latex_table(results, targets)
        table_path = DRYRUN_PATH.with_suffix(".tex") if is_dry else TABLE_PATH
        with open(table_path, "w") as f:
            f.write(latex)
        if not is_dry:
            with open(PAPER_TABLE_PATH, "w") as f:
                f.write(latex)
        print(f"\n  LaTeX table: {table_path}")
    except Exception as e:
        print(f"\n  LaTeX table error: {e}")

    try:
        fig_out = DRYRUN_PATH.with_suffix(".png") if is_dry else FIG_PATH
        generate_figure(results, targets, fig_out)
        if not is_dry:
            generate_figure(results, targets, PAPER_FIG_PATH)
    except Exception as e:
        print(f"\n  Figure error: {e}")

    print(f"\n  Results saved: {save_path}")


if __name__ == "__main__":
    main()
