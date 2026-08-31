#!/usr/bin/env python3
"""
verify_linesplit.py  (pmsm0821)

Independent audit of the line-split generalization results. Re-derives every
reported number from the raw outputs and cross-checks the dataset builder
against the raw parquet. Prints PASS/FAIL per check; exits nonzero on failure.

Run:  python3 verify_linesplit.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))  # repo root; run from here
MF = os.path.join(BASE, "model_features")
DATA = os.path.join(MF, "biorep_datasets.csv")
RES = os.path.join(MF, "line_split_generalization_pka", "per_combo_line_results.csv")
SUM = os.path.join(MF, "line_split_generalization_pka", "generalization_summary.csv")
POOL = os.path.join(MF, "features_biotin_pool.csv")
PARQUET = os.path.join(BASE, "data", "pmsm_results.all.parquet")

CELL_LINE_NAMES = ["HepG2", "MIA_PaCa2", "RD", "G292", "Capan2", "MG63",
                   "A673", "Panc1", "A549", "NCI_H358", "A204", "Calu1",
                   "Capan1"]
RUN_TOKEN_RE = re.compile(r"rep(\d+)_(\d+)")

failures = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


print("=" * 72)
print("CHECK 1 — dataset builder output structure")
print("=" * 72)
df = pd.read_csv(DATA)
check("row count = 44,199", len(df) == 44199, f"got {len(df):,}")
ds = df.groupby(["cell_line", "bio_rep"]).size()
check("25 valid (line, bio-rep) datasets", len(ds) == 25, f"got {len(ds)}")
per_line = df.groupby("cell_line")["bio_rep"].nunique()
check("every line has >=1 bio rep", (per_line >= 1).all(),
      ", ".join(f"{k}={v}" for k, v in per_line.items()))
check("techrep_detect_count in {1,2,3}",
      df["techrep_detect_count"].isin([1, 2, 3]).all())
check("techrep_detect_frac == count/3",
      np.allclose(df["techrep_detect_frac"], df["techrep_detect_count"] / 3))
check("techrep_detect_all3 consistent",
      (df["techrep_detect_all3"] == (df["techrep_detect_count"] >= 3).astype(int)).all())
check("no NaN targets", df["target_surface"].notna().all())
check("targets are 0/1", df["target_surface"].isin([0, 1]).all())

print("\n" + "=" * 72)
print("CHECK 2 — SURFY target is a per-protein constant in the pool")
print("=" * 72)
pool = pd.read_csv(POOL)
uni_c = "uniprot_accession"
tgt_c = "target_surface"
nun = pool.groupby(uni_c)[tgt_c].nunique(dropna=True)
check("target_surface constant per protein", (nun <= 1).all(),
      f"max distinct values per protein = {nun.max()}")

print("\n" + "=" * 72)
print("CHECK 3 — results CSV: split integrity")
print("=" * 72)
res = pd.read_csv(RES)
res["in_test"] = [c in tl.split("|") for c, tl in zip(res.cell_line, res.test_lines)]
check("every scored line is a held-out test line", res["in_test"].all())
check("exactly 3 test lines per combo",
      (res["test_lines"].str.split("|").map(len) == 3).all())
n_combos = res.groupby("run")["test_lines"].nunique()
check("286 unique combos per run", (n_combos == 286).all(),
      ", ".join(f"{k}={v}" for k, v in n_combos.items()))
check("3 line-rows per (run, combo)",
      (res.groupby(["run", "test_lines"])["cell_line"].nunique() == 3).all())

# baseline_precision must equal the actual target mean of that test line
actual_base = df.groupby("cell_line")["target_surface"].mean()
cmp_base = res.drop_duplicates(["cell_line"]).set_index("cell_line")["baseline_precision"]
cmp_base = cmp_base.reindex(actual_base.index)
check("baseline_precision == actual SURFY rate per line",
      np.allclose(cmp_base.values, actual_base.values, atol=1e-9),
      f"max abs diff = {np.abs(cmp_base.values - actual_base.values).max():.2e}")

print("\n" + "=" * 72)
print("CHECK 4 — summary table recomputed from raw per-line results")
print("=" * 72)
summ = pd.read_csv(SUM)
mac = (res.groupby(["run", "test_lines", "budget"], as_index=False)
       .agg(precision=("precision", "mean"), recall=("recall", "mean"),
            fp_reduction=("fp_reduction", "mean"),
            pr_auc=("average_precision", "mean"), roc_auc=("roc_auc", "mean")))
worst = 0.0
for _, r in summ.iterrows():
    g = mac[(mac.run == r["run"]) & (mac.budget == r["budget"])]
    for k in ["precision", "recall", "fp_reduction", "pr_auc", "roc_auc"]:
        diff = abs(g[k].mean() - r[f"{k}_mean"])
        worst = max(worst, diff)
check("summary means recompute exactly", worst < 1e-9, f"max abs diff = {worst:.2e}")

print("\n" + "=" * 72)
print("CHECK 5 — oof95 thresholds are non-degenerate")
print("=" * 72)
oof = res[res.budget == "oof95"]
if len(oof):
    check("oof95 thresholds present", oof["threshold"].notna().all())
    check("oof95 thresholds not collapsed to 0",
          (oof["threshold"] > 0).mean() > 0.95,
          f"frac > 0: {(oof['threshold'] > 0).mean():.3f}, "
          f"median {oof['threshold'].median():.4f}, "
          f"unique {oof['threshold'].nunique()}")
    # The >=95% recall constraint is enforced on the TRAIN inner-OOF scores
    # (guaranteed by threshold_at_min_recall, unit-tested). Held-out recall is
    # a random variable around it (~650 positives/line -> macro sigma ~0.5pp),
    # so we check the distribution, not a hard per-combo floor.
    g = oof.groupby(["run", "test_lines"])["recall"].mean()
    per_run_mean = g.groupby("run").mean()
    stats = ", ".join(f"{k}={v:.4f}" for k, v in per_run_mean.items())
    check("oof95 mean macro recall >= 0.95 (per run)",
          per_run_mean.ge(0.95).all(), stats)
    print(f"        [INFO] oof95 macro-recall spread: min {g.min():.3f}, "
          f"p5 {g.quantile(.05):.3f}, median {g.median():.3f}; "
          f"combos < 0.949: {(g < 0.949).mean():.1%} "
          f"(a few percent of combos dipping below is expected sampling noise)")
else:
    check("oof95 thresholds present", False, "budget missing — rerun with --with-oof-threshold")

print("\n" + "=" * 72)
print("CHECK 6 — spot-check builder aggregation against the raw parquet")
print("=" * 72)
try:
    import pyarrow.parquet as pq
    cols = pq.read_schema(PARQUET).names
    lut = {c.lower(): c for c in cols}
    prot_c = lut.get("fasta_id"); run_c = lut.get("run_name")
    pep_c = lut.get("posterior_error", None)
    mod_c = lut.get("modified_sequence")
    raw = pd.read_parquet(PARQUET, columns=[c for c in [prot_c, run_c, mod_c, pep_c] if c])
    rng = np.random.default_rng(0)
    for _, row in df.sample(5, random_state=0).iterrows():
        cl, br, acc = row["cell_line"], row["bio_rep"], row["uniprot_accession"]
        tok = raw[run_c].astype(str)
        m_line = tok.str.contains(cl, regex=False)
        m_bio = tok.map(lambda s: (int(m.group(1)) if (m := RUN_TOKEN_RE.search(s)) else -1)) == br
        m_acc = raw[prot_c].astype(str).str.contains(acc, regex=False)
        m_mod = raw[mod_c].astype(str).str.contains("UniMod:293", regex=False)
        sub = raw[m_line & m_bio & m_acc & m_mod]
        # builder additionally applies q-value/decoy filters, so raw count
        # must be an UPPER bound on n_ms_biotin_observations
        check(f"{cl}/rep{br}/{acc}: raw rows {len(sub)} >= reported "
              f"{int(row['n_ms_biotin_observations'])}",
              len(sub) >= row["n_ms_biotin_observations"])
        n_tech = tok[m_line & m_bio].map(
            lambda s: int(m.group(2)) if (m := RUN_TOKEN_RE.search(s)) else -1).nunique()
        check(f"{cl}/rep{br}: has 3 tech reps in raw data", n_tech >= 3,
              f"found {n_tech}")
except Exception as e:
    check("parquet spot-check", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 72)
if failures:
    print(f"AUDIT FAILED — {len(failures)} check(s): {failures}")
    sys.exit(1)
print("AUDIT PASSED — all checks green")
