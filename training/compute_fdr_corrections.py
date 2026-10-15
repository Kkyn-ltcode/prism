"""Statistical multiplicity correction suite applying FDR procedures."""

import os
import json
import numpy as np
from scipy import stats

RESULTS_FILE = "./results/extended_table1_v2.json"
OUTPUT_FILE = "./results/statistical_significance_corrected.json"


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg procedure for FDR control."""
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    q_values = np.zeros(n)
    cummin_val = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (sorted_p[i] * n) / rank
        cummin_val = min(cummin_val, val)
        q_values[i] = min(cummin_val, 1.0)

    orig_q = np.zeros(n)
    orig_q[sorted_indices] = q_values
    return orig_q.tolist()


def run_statistical_analysis():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    targets = [t for t in data.keys() if "eqgnn_v2" in data[t]]
    print("=" * 90)
    print("STATISTICAL MULTIPLICITY CORRECTION")
    print("=" * 90)

    records = []

    for target in targets:
        t_data = data[target]
        if "eqgnn_v2" not in t_data:
            continue
        eq_maes = np.array(t_data["eqgnn_v2"]["maes"])

        baselines = [m for m in t_data.keys() if m not in [
            "unit", "display", "type", "statistical_tests", "eqgnn_v2"
        ] and isinstance(t_data[m], dict) and "maes" in t_data[m]]

        for b in baselines:
            b_maes = np.array(t_data[b]["maes"])
            diff = eq_maes - b_maes
            mean_diff = float(np.mean(diff))

            # Paired t-test
            _, p_ttest = stats.ttest_rel(eq_maes, b_maes)
            # Wilcoxon signed-rank
            try:
                _, p_wilcoxon = stats.wilcoxon(eq_maes, b_maes)
            except Exception:
                p_wilcoxon = 1.0

            records.append({
                "target": target,
                "comparison": f"Q-PRISM vs {b}",
                "model_a": "eqgnn_v2",
                "model_b": b,
                "mean_diff": mean_diff,
                "p_ttest": float(p_ttest),
                "p_wilcoxon": float(p_wilcoxon),
            })

        # Also add SIREN vs Classical Twin (tanh) comparison
        if "trig_twin_siren" in t_data and "classical_twin" in t_data:
            s_maes = np.array(t_data["trig_twin_siren"]["maes"])
            c_maes = np.array(t_data["classical_twin"]["maes"])
            diff_sc = s_maes - c_maes
            _, p_sc_t = stats.ttest_rel(s_maes, c_maes)
            try:
                _, p_sc_w = stats.wilcoxon(s_maes, c_maes)
            except Exception:
                p_sc_w = 1.0
            records.append({
                "target": target,
                "comparison": f"PRISM_SIREN vs Classical_Twin_tanh",
                "model_a": "trig_twin_siren",
                "model_b": "classical_twin",
                "mean_diff": float(np.mean(diff_sc)),
                "p_ttest": float(p_sc_t),
                "p_wilcoxon": float(p_sc_w),
            })

    total_tests = len(records)
    print(f"Total hypothesis tests across all targets: m = {total_tests}")

    # Extract p-values for correction
    ttest_p_vals = [r["p_ttest"] for r in records]
    wilcoxon_p_vals = [r["p_wilcoxon"] for r in records]

    # Bonferroni correction: p_bonf = min(1, p * m)
    bonf_ttest = [min(1.0, p * total_tests) for p in ttest_p_vals]
    bonf_wilcoxon = [min(1.0, p * total_tests) for p in wilcoxon_p_vals]

    # Benjamini-Hochberg FDR correction
    fdr_ttest = benjamini_hochberg(ttest_p_vals)
    fdr_wilcoxon = benjamini_hochberg(wilcoxon_p_vals)

    for i, r in enumerate(records):
        r["p_ttest_bonferroni"] = bonf_ttest[i]
        r["p_ttest_fdr_bh"] = fdr_ttest[i]
        r["p_wilcoxon_bonferroni"] = bonf_wilcoxon[i]
        r["p_wilcoxon_fdr_bh"] = fdr_wilcoxon[i]
        r["sig_raw_005"] = r["p_ttest"] < 0.05
        r["sig_fdr_005"] = r["p_ttest_fdr_bh"] < 0.05
        r["sig_bonf_005"] = r["p_ttest_bonferroni"] < 0.05

    # Print summary table
    print("\n" + "=" * 115)
    print(f"{'Target':<8} | {'Comparison':<35} | {'Diff':<8} | {'Raw p (t)':<10} | {'BH FDR q':<10} | {'Bonf p':<10} | {'Sig (FDR < 0.05)'}")
    print("-" * 115)
    for r in records:
        sig_str = "YES (***)" if r["p_ttest_fdr_bh"] < 0.001 else ("YES (**)" if r["p_ttest_fdr_bh"] < 0.01 else ("YES (*)" if r["p_ttest_fdr_bh"] < 0.05 else "ns"))
        print(f"{r['target']:<8} | {r['comparison']:<35} | {r['mean_diff']:<+8.4f} | {r['p_ttest']:<10.4e} | {r['p_ttest_fdr_bh']:<10.4e} | {r['p_ttest_bonferroni']:<10.4e} | {sig_str}")
    print("=" * 115)

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "total_hypotheses": total_tests,
            "correction_methods": ["Bonferroni (FWER)", "Benjamini-Hochberg (FDR)"],
            "comparisons": records
        }, f, indent=2)
    print(f"\nSaved statistical corrections to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_statistical_analysis()
