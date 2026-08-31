#!/usr/bin/env python3
"""
line_split_logreg_top20_biorep_pka.py

Cross-cell-line generalization experiment, following the director's design:

  - dataset unit = (cell line, biological replicate); the 3 technical
    replicates within it were aggregated by build_biorep_datasets.py
    (run that first; bio reps without all 3 tech reps are excluded there)
  - reproducibility features derived from the 3 technical replicates
    (techrep_detect_count / frac / all3, n_biotin_peptides, pep_min/mean)
  - split BY CELL LINE: 10 lines train / 3 lines test; both bio reps of a
    line stay in the same split (guaranteed by construction)
  - ALL C(13,3) = 286 test-line combinations are evaluated -> a
    distribution of generalization performance, not one arbitrary split

Operating-point calibration (per test line, label-free):
  A global threshold calibrated on train OOF probabilities does not
  transfer to unseen lines (see line_split_cv results: recall pinned at
  1.0, FP reduction ~0). Instead, the model is used as a RANKER within
  each held-out line: flag the top X% of that line's own predicted
  probabilities as surface candidates. X never touches test labels.

  Budget levels reported:
    top 10% / 20% / 30% / 50% of each test line's ranked list, plus
    "train_prev" = the SURFY prevalence observed in the 10 train lines
    (label information from TRAIN only).

  Threshold-free metrics (PR-AUC, ROC-AUC per test line) are always
  reported - they measure ranking transfer independent of calibration.

  --with-oof-threshold additionally reproduces the old train-OOF >=95%
  recall operating point for reference (adds 4 inner-CV fits per combo).

Runs:
  R1_top20_biorep       the exact Top-20 whitelist (assay features
                        recomputed per (line, bio rep) dataset)
  R2_top20_plus_repro   R1 + technical-replicate reproducibility features
  R3_top20_plus_repro_pka  R2 + PROPKA lysine-pKa reactivity features
                        (only when model_features/pka_features.csv exists;
                        build it with build_pka_features.py)

Run:  python3 line_split_logreg_top20_biorep_pka.py
      python3 line_split_logreg_top20_biorep_pka.py --with-oof-threshold
      python3 line_split_logreg_top20_biorep_pka.py --max-combos 5   # smoke test
Out:  model_features/line_split_generalization_pka/
"""

import argparse
import itertools
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # repo root; run from here
OUT_DIR = os.path.join(BASE_DIR, "model_features")
DATA_CSV = os.path.join(OUT_DIR, "biorep_datasets.csv")
ANALYSIS_DIR = os.path.join(OUT_DIR, "line_split_generalization_pka")
PKA_CSV = os.path.join(OUT_DIR, "pka_features.csv")

PREDICT_FRACTIONS = [0.10, 0.20, 0.30, 0.50]   # top-X% budgets per test line

FEATURE_WHITELIST = [
    # Assay / labeling evidence (recomputed per (line, bio rep) dataset)
    "n_ms_biotin_observations",                 # +0.992
    "n_actual_biotin_sites",                    # -0.869

    # Membrane-relative geometry
    "min_peptide_membrane_dist",                # +0.857
    "max_peptide_membrane_dist",                # +0.371

    # Peptide / site structure
    "peptide_mean_plddt_v3",                    # -0.839
    "peptide_frac_helix_v3",                    # -0.671
    "site_mean_rsa_v3",                         # -0.602
    "peptide_frac_exposed_v3",                  # +0.325
    "peptide_max_contiguous_exposed_v3",        # -0.322
    "peptide_mean_rsa_v3",                      # -0.284
    "peptide_min_rsa_v3",                       # -0.238

    # Peptide geometry
    "peptide_rg_v3",                            # +0.493
    "peptide_mean_adjacent_ca_distance_v3",     # +0.233
    "peptide_end_to_end_distance_v3",           # -0.209

    # Sequence physicochemistry
    "peptide_cysteine_fraction_v3",             # +0.513
    "peptide_gravy_v3",                         # -0.424
    "peptide_sequence_entropy_v3",              # +0.290

    # 3D local neighborhood / reactive-site packing
    "lys_nz_neighbor_count_6A_v3",              # -0.222
    "site_neighbor_charged_fraction_8A_v3",     # -0.162
    "site_neighbor_hydrophobic_fraction_8A_v3", # -0.136
]

# derived from the 3 technical replicates within each (line, bio rep)
REPRO_FEATURES = [
    "techrep_detect_count",     # detected in k of 3 tech reps
    "techrep_detect_frac",      # k / 3
    "techrep_detect_all3",      # 1 if detected in all 3
    "n_biotin_peptides",        # unique biotin+ peptides in the dataset
    "pep_min",                  # best posterior error probability
    "pep_mean",                 # mean posterior error probability
]

RUNS = {
    "R1_top20_biorep": FEATURE_WHITELIST,
    "R2_top20_plus_repro": FEATURE_WHITELIST + REPRO_FEATURES,
}


# ---------------------------------------------------------------------------
# pipeline helpers (identical recipe to per_cell_line_logreg_top20_v3.py)
# ---------------------------------------------------------------------------
def threshold_at_min_recall(y_true, y_prob, target_recall=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.0
    eligible = np.where(recall[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        return 0.0
    return float(thresholds[eligible[-1]])


def fit_pipeline(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(imputer.fit_transform(X_train))
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_s, y_train)
    return imputer, scaler, model


def transform_pipeline(imputer, scaler, X):
    return scaler.transform(imputer.transform(X))


def inner_oof_threshold(X, y, groups, target_recall=0.95, n_splits=4):
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise RuntimeError("Not enough groups for inner CV.")
    gkf = GroupKFold(n_splits=n_splits)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        imputer, scaler, model = fit_pipeline(X[tr_idx], y[tr_idx])
        oof_prob[va_idx] = model.predict_proba(
            transform_pipeline(imputer, scaler, X[va_idx])
        )[:, 1]
    valid = ~np.isnan(oof_prob)
    if valid.sum() == 0:
        raise RuntimeError("No valid inner-CV predictions.")
    return threshold_at_min_recall(y[valid], oof_prob[valid], target_recall)


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    try:
        ap = average_precision_score(y_true, y_prob)
    except Exception:
        ap = np.nan
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
    baseline_fp = int((y_true == 0).sum())
    fp_reduction = (baseline_fp - fp) / baseline_fp if baseline_fp else np.nan
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": precision, "recall": recall, "specificity": specificity,
        "fp_reduction": fp_reduction,
        "average_precision": ap, "roc_auc": auc,
    }


# ---------------------------------------------------------------------------
# line-split driver
# ---------------------------------------------------------------------------
def evaluate_combo(df, train_lines, test_lines, features, run_name,
                   with_oof_threshold, target_recall=0.95):
    """Fit one global model on the train lines; score every test line."""
    tr = df[df["cell_line"].isin(train_lines)]
    te = df[df["cell_line"].isin(test_lines)]

    X_tr = tr[features].to_numpy(dtype=float)
    y_tr = tr["target_surface"].to_numpy(dtype=int)

    imputer, scaler, model = fit_pipeline(X_tr, y_tr)

    oof_threshold = np.nan
    if with_oof_threshold:
        oof_threshold = inner_oof_threshold(
            X_tr, y_tr, tr["uniprot_accession"].astype(str).to_numpy(),
            target_recall=target_recall, n_splits=4,
        )

    train_prev = float(y_tr.mean())   # label info from TRAIN lines only
    budgets = {f"top{int(f * 100)}pct": f for f in PREDICT_FRACTIONS}
    budgets["train_prev"] = train_prev
    if with_oof_threshold:
        budgets["oof95"] = None       # absolute probability threshold

    rows = []
    for ln in test_lines:
        sub = te[te["cell_line"] == ln]
        X_te = sub[features].to_numpy(dtype=float)
        y_te = sub["target_surface"].to_numpy(dtype=int)
        prob = model.predict_proba(transform_pipeline(imputer, scaler, X_te))[:, 1]

        for budget_name, frac in budgets.items():
            if budget_name == "oof95":
                thr = oof_threshold
            else:
                # label-free: threshold = quantile of THIS line's own scores
                thr = float(np.quantile(prob, 1.0 - frac)) if len(prob) else 0.0
            m = compute_metrics(y_te, prob, thr)
            rows.append({
                "run": run_name,
                "test_lines": "|".join(sorted(test_lines)),
                "cell_line": ln,
                "budget": budget_name,
                "budget_frac": (frac if frac is not None else np.nan),
                "threshold": thr,
                "n": len(sub),
                "n_pos": int(y_te.sum()),
                "baseline_precision": float(y_te.mean()),
                **m,
            })
    coefs = [
        {"run": run_name, "test_lines": "|".join(sorted(test_lines)),
         "feature": f, "coefficient": float(c)}
        for f, c in zip(features, model.coef_[0])
    ]
    return rows, coefs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="recall target for --with-oof-threshold")
    parser.add_argument("--with-oof-threshold", action="store_true",
                        help="also evaluate the train-OOF >=95%% recall "
                             "operating point (slow: adds inner CV)")
    parser.add_argument("--max-combos", type=int, default=None,
                        help="subsample combos for a quick test")
    parser.add_argument("--base-dir", default=BASE_DIR,
                        help="project root containing model_features/")
    args = parser.parse_args()

    global OUT_DIR, DATA_CSV, ANALYSIS_DIR, PKA_CSV
    OUT_DIR = os.path.join(args.base_dir, "model_features")
    DATA_CSV = os.path.join(OUT_DIR, "biorep_datasets.csv")
    ANALYSIS_DIR = os.path.join(OUT_DIR, "line_split_generalization_pka")
    PKA_CSV = os.path.join(OUT_DIR, "pka_features.csv")

    t0 = time.time()
    print("=" * 80)
    print("LINE-SPLIT GENERALIZATION — TOP-20 BIOREP MODEL")
    print("=" * 80)

    if not os.path.exists(DATA_CSV):
        print(f"[FATAL] {DATA_CSV} not found.")
        print("        run build_biorep_datasets.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["target_surface"])
    df["target_surface"] = df["target_surface"].astype(int)
    print(f"[LOAD] biorep_datasets: {df.shape}")

    runs = dict(RUNS)   # local copy; R3 added when pKa features are available
    if os.path.exists(PKA_CSV):
        pka = pd.read_csv(PKA_CSV)
        pka_feats = [c for c in pka.columns if c != "uniprot_accession"]
        df = df.merge(pka, on="uniprot_accession", how="left")
        cov = (df.groupby("uniprot_accession")[pka_feats[0]]
                 .first().notna().mean())
        print(f"[LOAD] pka_features: {pka.shape[0]} proteins, "
              f"{len(pka_feats)} features; accession coverage {cov:.1%}")
        runs["R3_top20_plus_repro_pka"] = (
            FEATURE_WHITELIST + REPRO_FEATURES + pka_feats)
    else:
        print(f"[WARN] {PKA_CSV} not found - running R1/R2 only.")

    all_feats = sorted({f for feats in runs.values() for f in feats})
    missing = [f for f in all_feats if f not in df.columns]
    if missing:
        print("\n[FATAL] Missing features in biorep_datasets.csv:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    lines = sorted(df["cell_line"].dropna().unique())
    n_lines = len(lines)
    if n_lines < 4:
        print(f"[FATAL] only {n_lines} cell lines available.")
        sys.exit(1)

    reps = df.groupby("cell_line")["bio_rep"].nunique()
    print(f"\n[DATASETS] {n_lines} cell lines; bio reps per line: "
          + ", ".join(f"{k}={v}" for k, v in reps.items()))
    print("[FEATURES] " + "; ".join(
        f"{name} ({len(feats)} feats)" for name, feats in runs.items()))

    combos = list(itertools.combinations(lines, 3))
    if args.max_combos:
        combos = combos[:args.max_combos]
    print(f"\n[SPLIT] {len(combos)} test-line combinations "
          f"({n_lines - 3} train / 3 test; bio reps stay with their line)")

    all_rows, all_coefs = [], []
    for ci, test_lines in enumerate(combos, 1):
        train_lines = [ln for ln in lines if ln not in test_lines]
        for run_name, feats in runs.items():
            rows, coefs = evaluate_combo(
                df, train_lines, list(test_lines), feats, run_name,
                with_oof_threshold=args.with_oof_threshold,
                target_recall=args.target_recall,
            )
            for r in rows:
                r["combo"] = ci
            for c in coefs:
                c["combo"] = ci
            all_rows.extend(rows)
            all_coefs.extend(coefs)
        if ci % 25 == 0 or ci == len(combos):
            print(f"  [{ci}/{len(combos)}] {time.time() - t0:.0f}s elapsed")

    res = pd.DataFrame(all_rows)
    coef_df = pd.DataFrame(all_coefs)

    # ------------------------------------------------------------------
    # summaries
    # ------------------------------------------------------------------
    def macro(g):
        return pd.Series({
            "precision": g["precision"].mean(),
            "recall": g["recall"].mean(),
            "fp_reduction": g["fp_reduction"].mean(),
            "pr_auc": g["average_precision"].mean(),
            "roc_auc": g["roc_auc"].mean(),
        })

    combo_macro = (res.groupby(["run", "combo", "budget"])
                   .apply(macro, include_groups=False).reset_index())

    print("\n" + "=" * 80)
    print("GENERALIZATION DISTRIBUTION ACROSS TEST-LINE COMBINATIONS")
    print("(macro = mean over the 3 held-out lines, then over combos)")
    print("=" * 80)
    summary_rows = []
    for (run_name, budget), g in combo_macro.groupby(["run", "budget"]):
        s = {"run": run_name, "budget": budget, "n_combos": len(g)}
        for k in ["precision", "recall", "fp_reduction", "pr_auc", "roc_auc"]:
            v = g[k].dropna()
            s.update({f"{k}_mean": v.mean(), f"{k}_median": v.median(),
                      f"{k}_p10": v.quantile(0.1), f"{k}_p90": v.quantile(0.9)})
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    for run_name in runs:
        sub = summary[summary.run == run_name]
        print(f"\n== {run_name} ==")
        for _, r in sub.iterrows():
            print(f"  {r['budget']:<10} "
                  f"prec {r['precision_mean']:.3f} "
                  f"[{r['precision_p10']:.3f}-{r['precision_p90']:.3f}]  "
                  f"rec {r['recall_mean']:.3f}  "
                  f"FPred {r['fp_reduction_mean']:.3f}  "
                  f"PR-AUC {r['pr_auc_mean']:.3f}  "
                  f"ROC-AUC {r['roc_auc_mean']:.3f}")

    # threshold-free ranking transfer, per test line
    rank = (res[res.budget == "top20pct"]
            .groupby(["run", "cell_line"])
            .agg(baseline=("baseline_precision", "mean"),
                 pr_auc=("average_precision", "mean"),
                 roc_auc=("roc_auc", "mean"),
                 top20_precision=("precision", "mean"),
                 top20_fp_reduction=("fp_reduction", "mean"))
            .reset_index())
    print("\n" + "=" * 80)
    print("PER-TEST-LINE MEANS (threshold-free AUCs + top-20% budget)")
    print("=" * 80)
    for run_name in runs:
        print(f"\n== {run_name} ==")
        print(rank[rank.run == run_name].drop(columns="run")
              .sort_values("pr_auc", ascending=False).to_string(index=False))

    coef_stability = (
        coef_df.groupby(["run", "feature"])["coefficient"]
        .agg(mean_coef="mean", sd_coef="std", n="size",
             positive_fraction=lambda x: float(np.mean(x > 0)))
        .reset_index())
    coef_stability["sign_consistency"] = np.maximum(
        coef_stability["positive_fraction"],
        1 - coef_stability["positive_fraction"])
    coef_stability["abs_mean_coef"] = coef_stability["mean_coef"].abs()
    coef_stability = coef_stability.sort_values(
        ["run", "abs_mean_coef"], ascending=[True, False])

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    res.to_csv(os.path.join(ANALYSIS_DIR, "per_combo_line_results.csv"),
               index=False)
    summary.to_csv(os.path.join(ANALYSIS_DIR, "generalization_summary.csv"),
                   index=False)
    rank.to_csv(os.path.join(ANALYSIS_DIR, "per_test_line_summary.csv"),
                index=False)
    coef_stability.to_csv(os.path.join(ANALYSIS_DIR, "coefficient_stability.csv"),
                          index=False)
    print(f"\n[SAVED] {ANALYSIS_DIR}")
    print(f"[DONE] {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
