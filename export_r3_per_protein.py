#!/usr/bin/env python3
"""
export_r3_per_protein.py — per-protein R3 results export.

R3 = Top-20 whitelist + tech-rep reproducibility + pKa features (35 feats),
the FINAL model per the 2026-08-28 director decision (no SignalP6).

R3 is not one model but 286 fits (all C(13,3) test-line combos). Every cell
line is held out in C(12,2) = 66 of them, so each (cell_line, bio_rep,
protein) row receives 66 OUT-OF-SAMPLE predictions from models trained only
on the other 10 lines. This script reruns those fits and aggregates them
into one flat table:

  identity    cell_line, bio_rep, uniprot_accession, gene_name
  truth       target_surface (SURFY surface_score >= 4)
  OOS scores  r3_prob_mean/sd/min/max over the 66 held-out fits, n_fits
  ranking     r3_rank_pct_in_line = fraction of the cell line's rows
              (both bio reps pooled, same unit the model ranks) scoring
              BELOW this row's mean prob
  operating points (per row):
      pred_top10/20/30/50pct   rank_pct >= 1 - budget        (label-free)
      pred_train_prev          rank_pct >= 1 - mean train prevalence
      oof95_flag_frac          fraction of the 66 fits whose OWN oof95
                               threshold this row's prob cleared
      pred_oof95               oof95_flag_frac >= 0.5        (majority vote)
      oof95_thr_mean           mean oof95 threshold across fits
  verdict     class_<budget> = TP/FP/TN/FN vs target_surface

Also emits a protein-level rollup (mean over bio reps):
  r3_per_protein_predictions_byprotein.csv

Run (from the repo root):
  python export_r3_per_protein.py                    # full 286 combos, ~45 min
  python export_r3_per_protein.py --max-combos 5     # smoke test
  python export_r3_per_protein.py --skip-oof         # ~10 min, no oof95 cols
  python export_r3_per_protein.py --include-features # append all 35 R3 features

Out: model_features/r3_per_protein_predictions.csv
     model_features/r3_per_protein_predictions_byprotein.csv
"""

import argparse
import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "model_features")
DATA_CSV = os.path.join(OUT_DIR, "biorep_datasets.csv")
PKA_CSV = os.path.join(OUT_DIR, "pka_features.csv")
UNIPROT_TSV = os.path.join(BASE_DIR, "data",
                           "uniprot_human_with_surface_score.tsv")

FEATURE_WHITELIST = [
    "n_ms_biotin_observations", "n_actual_biotin_sites",
    "min_peptide_membrane_dist", "max_peptide_membrane_dist",
    "peptide_mean_plddt_v3", "peptide_frac_helix_v3", "site_mean_rsa_v3",
    "peptide_frac_exposed_v3", "peptide_max_contiguous_exposed_v3",
    "peptide_mean_rsa_v3", "peptide_min_rsa_v3",
    "peptide_rg_v3", "peptide_mean_adjacent_ca_distance_v3",
    "peptide_end_to_end_distance_v3",
    "peptide_cysteine_fraction_v3", "peptide_gravy_v3",
    "peptide_sequence_entropy_v3",
    "lys_nz_neighbor_count_6A_v3", "site_neighbor_charged_fraction_8A_v3",
    "site_neighbor_hydrophobic_fraction_8A_v3",
]
REPRO_FEATURES = [
    "techrep_detect_count", "techrep_detect_frac", "techrep_detect_all3",
    "n_biotin_peptides", "pep_min", "pep_mean",
]
ID_COLS = ["cell_line", "bio_rep", "uniprot_accession"]


def fit_pipeline(X_train, y_train):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(imputer.fit_transform(X_train))
    model = LogisticRegression(class_weight="balanced", max_iter=2000,
                               solver="lbfgs", random_state=42, n_jobs=1)
    model.fit(X_s, y_train)
    return imputer, scaler, model


def threshold_at_min_recall(y_true, y_prob, target_recall=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.0
    eligible = np.where(recall[:-1] >= target_recall)[0]
    if len(eligible) == 0:
        return 0.0
    return float(thresholds[eligible[-1]])


def inner_oof_threshold(X, y, groups, target_recall=0.95, n_splits=4):
    n_splits = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    oof_prob = np.full(len(y), np.nan, dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        imputer, scaler, model = fit_pipeline(X[tr_idx], y[tr_idx])
        oof_prob[va_idx] = model.predict_proba(
            scaler.transform(imputer.transform(X[va_idx])))[:, 1]
    valid = ~np.isnan(oof_prob)
    return threshold_at_min_recall(y[valid], oof_prob[valid], target_recall)


def verdict(pred, truth):
    if pred and truth:
        return "TP"
    if pred and not truth:
        return "FP"
    if not pred and truth:
        return "FN"
    return "TN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--include-features", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_csv(DATA_CSV).dropna(subset=["target_surface"])
    df["target_surface"] = df["target_surface"].astype(int)

    pka = pd.read_csv(PKA_CSV)
    pka_feats = [c for c in pka.columns if c != "uniprot_accession"]
    df = df.merge(pka, on="uniprot_accession", how="left")
    feats = FEATURE_WHITELIST + REPRO_FEATURES + pka_feats
    print(f"[LOAD] {df.shape[0]:,} rows; R3 = {len(feats)} features")

    if os.path.exists(UNIPROT_TSV):
        uni = pd.read_csv(UNIPROT_TSV, sep="\t",
                          usecols=["Entry", "Gene Names (primary)"])
        uni = uni.drop_duplicates("Entry").rename(
            columns={"Entry": "uniprot_accession",
                     "Gene Names (primary)": "gene_name"})
        df = df.merge(uni, on="uniprot_accession", how="left")

    df["_row_id"] = np.arange(len(df))
    lines = sorted(df["cell_line"].unique())
    combos = list(itertools.combinations(lines, 3))
    if args.max_combos:
        combos = combos[:args.max_combos]
    print(f"[SPLIT] {len(combos)} combos (oof95: {not args.skip_oof})")

    prob_sum = np.zeros(len(df))
    prob_sumsq = np.zeros(len(df))
    prob_min = np.full(len(df), np.inf)
    prob_max = np.full(len(df), -np.inf)
    n_fits = np.zeros(len(df), dtype=int)
    oof_flag = np.zeros(len(df))
    oof_thr_sum = np.zeros(len(df))
    train_prev_sum = np.zeros(len(df))

    for ci, test_lines in enumerate(combos, 1):
        train_lines = [ln for ln in lines if ln not in test_lines]
        tr = df[df["cell_line"].isin(train_lines)]
        te = df[df["cell_line"].isin(test_lines)]
        X_tr = tr[feats].to_numpy(dtype=float)
        y_tr = tr["target_surface"].to_numpy(dtype=int)
        imputer, scaler, model = fit_pipeline(X_tr, y_tr)

        thr = np.nan
        if not args.skip_oof:
            thr = inner_oof_threshold(
                X_tr, y_tr, tr["uniprot_accession"].astype(str).to_numpy(),
                target_recall=args.target_recall)

        prob = model.predict_proba(
            scaler.transform(imputer.transform(
                te[feats].to_numpy(dtype=float))))[:, 1]
        idx = te["_row_id"].to_numpy()
        prob_sum[idx] += prob
        prob_sumsq[idx] += prob ** 2
        prob_min[idx] = np.minimum(prob_min[idx], prob)
        prob_max[idx] = np.maximum(prob_max[idx], prob)
        n_fits[idx] += 1
        train_prev_sum[idx] += y_tr.mean()
        if not args.skip_oof:
            oof_thr_sum[idx] += thr
            oof_flag[idx] += (prob >= thr)

        if ci % 25 == 0 or ci == len(combos):
            print(f"  [{ci}/{len(combos)}] {time.time() - t0:.0f}s elapsed")

    n = np.maximum(n_fits, 1)
    df["r3_n_fits"] = n_fits
    df["r3_prob_mean"] = prob_sum / n
    var = prob_sumsq / n - df["r3_prob_mean"] ** 2
    df["r3_prob_sd"] = np.sqrt(np.maximum(var, 0))
    df["r3_prob_min"] = np.where(n_fits > 0, prob_min, np.nan)
    df["r3_prob_max"] = np.where(n_fits > 0, prob_max, np.nan)
    df["train_prev_mean"] = train_prev_sum / n
    if not args.skip_oof:
        df["oof95_thr_mean"] = oof_thr_sum / n
        df["oof95_flag_frac"] = oof_flag / n
        df["pred_oof95"] = df["oof95_flag_frac"] >= 0.5

    # within-line rank percentile of the mean OOS prob (bio reps pooled,
    # the same unit the model ranks within a held-out line)
    df["r3_rank_pct_in_line"] = (
        df.groupby("cell_line")["r3_prob_mean"].rank(pct=True))

    for frac in (0.10, 0.20, 0.30, 0.50):
        col = f"pred_top{int(frac * 100)}pct"
        df[col] = df["r3_rank_pct_in_line"] >= (1.0 - frac)
    df["pred_train_prev"] = (df["r3_rank_pct_in_line"]
                             >= (1.0 - df["train_prev_mean"]))

    budget_cols = ([f"pred_top{f}pct" for f in (10, 20, 30, 50)]
                   + ["pred_train_prev"]
                   + ([] if args.skip_oof else ["pred_oof95"]))
    for col in budget_cols:
        df[f"class_{col[5:]}"] = [verdict(p, t) for p, t in
                                  zip(df[col], df["target_surface"])]

    out_cols = (ID_COLS
                + (["gene_name"] if "gene_name" in df.columns else [])
                + ["target_surface", "r3_n_fits", "r3_prob_mean",
                   "r3_prob_sd", "r3_prob_min", "r3_prob_max",
                   "r3_rank_pct_in_line", "train_prev_mean"]
                + ([] if args.skip_oof else
                   ["oof95_thr_mean", "oof95_flag_frac"])
                + budget_cols
                + [f"class_{c[5:]}" for c in budget_cols])
    if args.include_features:
        out_cols += feats
    out = df[out_cols].sort_values(
        ["cell_line", "bio_rep", "r3_prob_mean"],
        ascending=[True, True, False])

    path1 = os.path.join(OUT_DIR, "r3_per_protein_predictions.csv")
    out.to_csv(path1, index=False)
    print(f"[SAVED] {path1}: {out.shape[0]:,} rows x {out.shape[1]} cols")

    # protein-level rollup (mean over bio reps; verdicts from the row-level
    # majority across both bio reps)
    agg = {"bio_rep": "count", "target_surface": "first",
           "r3_prob_mean": "mean", "r3_prob_sd": "mean",
           "r3_rank_pct_in_line": "mean"}
    prot = (out.groupby(["cell_line", "uniprot_accession"], as_index=False)
               .agg(agg)
               .rename(columns={"bio_rep": "n_bio_reps",
                                "r3_prob_mean": "r3_prob_mean_bioreps",
                                "r3_prob_sd": "r3_prob_sd_bioreps",
                                "r3_rank_pct_in_line":
                                    "r3_rank_pct_in_line_mean"}))
    if "gene_name" in df.columns:
        prot = prot.merge(
            df[["uniprot_accession", "gene_name"]].drop_duplicates(),
            on="uniprot_accession", how="left")
    for frac in (0.10, 0.20, 0.30, 0.50):
        col = f"pred_top{int(frac * 100)}pct"
        prot[col] = prot["r3_rank_pct_in_line_mean"] >= (1.0 - frac)
        prot[f"class_{col[5:]}"] = [verdict(p, t) for p, t in
                                    zip(prot[col], prot["target_surface"])]
    path2 = os.path.join(OUT_DIR, "r3_per_protein_predictions_byprotein.csv")
    prot.to_csv(path2, index=False)
    print(f"[SAVED] {path2}: {prot.shape[0]:,} rows x {prot.shape[1]} cols")
    print(f"[DONE] {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
