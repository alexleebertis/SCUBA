#!/usr/bin/env python3
"""
build_biorep_datasets.py  (pmsm0821)

Director's redesign (2026-08-24):
  - each biological replicate = independent cell culture plate
  - dataset unit = (cell line, biological replicate)
  - aggregate the 3 technical replicates within it
  - derive reproducibility features from the 3 technical replicates
  - exclude biological replicates without the full set of 3 tech reps
    (and runs that don't carry a repX_Y token at all)

run_name convention (confirmed):  ..._repX_Y  ->  biological replicate X,
technical replicate Y   (e.g. 260619_MIA_PaCa2_DIA_rep1_3 = line MIA_PaCa2,
bio rep 1, tech rep 3)

Per (cell_line, bio_rep, protein) row:
  assay (recomputed within the dataset):
    n_ms_biotin_observations   biotin+ peptide-quant rows after FDR filter
    n_actual_biotin_sites      unique labeled lysine positions (via FASTA map)
    n_biotin_peptides          unique biotin+ modified peptides
    pep_min / pep_mean         posterior error probability aggregates
  technical-replicate reproducibility (the 3 tech reps):
    techrep_detect_count       in how many of the 3 tech reps the protein
                               is detected (0-3)
    techrep_detect_frac        count / 3
    techrep_detect_all3        1 if detected in all 3
  structural / sequence / membrane features: merged per-protein from
    local_site_features_v3.csv + membrane_distance_features.csv
    (unchanged - they are protein-level constants)
  target_surface               SURFY label from features_biotin_pool.csv

Run:  python3 build_biorep_datasets.py
Out:  model_features/biorep_datasets.csv
"""

import os
import re
import time
import argparse

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# DEFAULTS (overridable via CLI; paths relative to this script)
# ----------------------------------------------------------------------------
PMSM_DIR = os.path.dirname(os.path.abspath(__file__))
MF       = os.path.join(PMSM_DIR, "model_features")
PARQUET  = os.path.join(PMSM_DIR, "data", "pmsm_results.all.parquet")
FASTA    = os.path.join(PMSM_DIR, "data", "human_proteome.fasta")
POOL_CSV = os.path.join(MF, "features_biotin_pool.csv")
V3_CSV   = os.path.join(MF, "local_site_features_v3.csv")
MEMB_CSV = os.path.join(MF, "membrane_distance_features.csv")
OUT_CSV  = os.path.join(MF, "biorep_datasets.csv")

Q_THRESHOLD  = 0.01
MOD_PATTERNS = ["UniMod:293"]
REQUIRED_TECH_REPS = 3     # bio reps with fewer tech reps are excluded

RUN_TOKEN_RE = re.compile(r"rep(\d+)_(\d+)")   # repX_Y = bio rep X, tech rep Y

CELL_LINE_NAMES = ["HepG2", "MIA_PaCa2", "RD", "G292", "Capan2", "MG63",
                   "A673", "Panc1", "A549", "NCI_H358", "A204", "Calu1",
                   "Capan1"]  # this study's 13 lines; override with --cell-lines

# per-protein structural features merged from v3 / membrane tables
# (everything in the Top-20 except the two recomputed assay features)
V3_FEATURES = [
    "peptide_mean_plddt_v3", "peptide_frac_helix_v3", "site_mean_rsa_v3",
    "peptide_frac_exposed_v3", "peptide_max_contiguous_exposed_v3",
    "peptide_mean_rsa_v3", "peptide_min_rsa_v3", "peptide_rg_v3",
    "peptide_mean_adjacent_ca_distance_v3", "peptide_end_to_end_distance_v3",
    "peptide_cysteine_fraction_v3", "peptide_gravy_v3",
    "peptide_sequence_entropy_v3", "lys_nz_neighbor_count_6A_v3",
    "site_neighbor_charged_fraction_8A_v3",
    "site_neighbor_hydrophobic_fraction_8A_v3",
]
MEMB_FEATURES = ["min_peptide_membrane_dist", "max_peptide_membrane_dist"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_col(cols, candidates):
    lut = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lut:
            return lut[cand.lower()]
    return None


def norm_acc(s):
    s = str(s).strip()
    if s.startswith(("sp|", "tr|")):
        parts = s.split("|")
        if len(parts) >= 2:
            return parts[1]
    return s.split(";")[0].strip()


def extract_cell_line(run_name):
    s = str(run_name)
    for name in CELL_LINE_NAMES:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", s):
            return name
    return None


def parse_run(run_name):
    """-> (bio_rep:int, tech_rep:int) or None if no repX_Y token."""
    m = RUN_TOKEN_RE.search(str(run_name))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_modified_peptide(modseq):
    """-> (clean_sequence, [0-based offsets of biotin-tagged residues])."""
    s = str(modseq)
    offsets, clean, pos = [], [], 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            j = s.find(")", i)
            tag = s[i + 1:j]
            if any(p in tag for p in MOD_PATTERNS) and pos > 0:
                offsets.append(pos - 1)
            i = j + 1
        elif ch.isalpha():
            clean.append(ch)
            pos += 1
            i += 1
        else:
            i += 1
    return "".join(clean), offsets


def load_fasta(path):
    seqs, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name, buf = norm_acc(line[1:].split()[0]), []
            else:
                buf.append(line)
        if name:
            seqs[name] = "".join(buf)
    return seqs


def main():
    global PMSM_DIR, MF, PARQUET, FASTA, POOL_CSV, V3_CSV, MEMB_CSV, OUT_CSV
    global CELL_LINE_NAMES, REQUIRED_TECH_REPS

    ap = argparse.ArgumentParser(
        description="Assemble (cell line, biological replicate, protein) datasets")
    ap.add_argument("--base-dir", default=PMSM_DIR,
                    help="repo root; all other defaults resolve relative to this")
    ap.add_argument("--parquet", default=None,
                    help="raw MaxQuant parquet (default: <base>/data/pmsm_results.all.parquet)")
    ap.add_argument("--fasta", default=None,
                    help="proteome FASTA (default: <base>/data/human_proteome.fasta)")
    ap.add_argument("--cell-lines", default=None,
                    help="comma-separated cell-line names as they appear in run_name "
                         "(default: this study's 13 lines)")
    ap.add_argument("--required-tech-reps", type=int, default=REQUIRED_TECH_REPS,
                    help="bio reps with fewer technical replicates are excluded")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <base>/model_features/biorep_datasets.csv)")
    args = ap.parse_args()

    PMSM_DIR = args.base_dir
    MF       = os.path.join(PMSM_DIR, "model_features")
    PARQUET  = args.parquet or os.path.join(PMSM_DIR, "data", "pmsm_results.all.parquet")
    FASTA    = args.fasta or os.path.join(PMSM_DIR, "data", "human_proteome.fasta")
    POOL_CSV = os.path.join(MF, "features_biotin_pool.csv")
    V3_CSV   = os.path.join(MF, "local_site_features_v3.csv")
    MEMB_CSV = os.path.join(MF, "membrane_distance_features.csv")
    OUT_CSV  = args.out or os.path.join(MF, "biorep_datasets.csv")
    if args.cell_lines:
        CELL_LINE_NAMES = [s.strip() for s in args.cell_lines.split(",") if s.strip()]
    REQUIRED_TECH_REPS = args.required_tech_reps

    t0 = time.time()

    # ---- load + strict filter ----------------------------------------------
    try:
        import pyarrow.parquet as pq
        all_cols = pq.read_schema(PARQUET).names
    except ImportError:
        all_cols = pd.read_parquet(PARQUET).columns
    prot_col  = find_col(all_cols, ["fasta_id", "protein_group", "protein"])
    mod_col   = find_col(all_cols, ["modified_sequence"])
    run_col   = find_col(all_cols, ["run_name"])
    pep_col   = find_col(all_cols, ["posterior_error", "pep"])
    qpep_col  = find_col(all_cols, ["global_peptide_q_value"])
    qprot_col = find_col(all_cols, ["global_protein_group_q_value"])
    decoy_col = find_col(all_cols, ["is_decoy"])
    use = sorted({c for c in [prot_col, mod_col, run_col, pep_col,
                              qpep_col, qprot_col, decoy_col] if c})
    log(f"[COLS] protein={prot_col} run={run_col} pep={pep_col}")
    df = pd.read_parquet(PARQUET, columns=use)

    n0 = len(df)
    if decoy_col is not None:
        df = df[~df[decoy_col].astype(str).str.lower().isin(["true", "1"])]
    if qpep_col is not None:
        df = df[df[qpep_col] <= Q_THRESHOLD]
    if qprot_col is not None:
        df = df[df[qprot_col] <= Q_THRESHOLD]
    tag_re = "|".join(re.escape(p) for p in MOD_PATTERNS)
    df = df[df[mod_col].astype(str).str.contains(tag_re)]
    log(f"[FILTER] {n0:,} -> {len(df):,} rows (no decoys, q<=0.01, biotin+)")

    # ---- run identity -------------------------------------------------------
    df["cell_line"] = df[run_col].map(extract_cell_line)
    parsed = df[run_col].map(parse_run)
    bad_runs = df.loc[parsed.isna() & df["cell_line"].notna(), run_col].unique()
    if len(bad_runs):
        log(f"[EXCLUDE] {len(bad_runs)} runs without a repX_Y token: "
            f"{list(bad_runs)[:5]}{' ...' if len(bad_runs) > 5 else ''}")
    df = df.dropna(subset=["cell_line"])
    df = df[parsed.notna()]
    parsed = parsed.dropna()
    df["bio_rep"]  = [p[0] for p in parsed]
    df["tech_rep"] = [p[1] for p in parsed]

    # explode protein groups, normalize accessions
    df[prot_col] = df[prot_col].astype(str).str.split(";")
    df = df.explode(prot_col)
    df["uniprot"] = df[prot_col].map(norm_acc)
    if pep_col is not None:
        df[pep_col] = pd.to_numeric(df[pep_col], errors="coerce")

    # ---- dataset validity: full set of tech reps per (line, bio rep) --------
    tech = (df.groupby(["cell_line", "bio_rep"])["tech_rep"]
              .nunique().reset_index(name="n_tech"))
    valid = tech[tech["n_tech"] >= REQUIRED_TECH_REPS]
    excluded = tech[tech["n_tech"] < REQUIRED_TECH_REPS]
    log(f"[DATASETS] {len(valid)} valid (line, bio-rep) datasets; "
        f"{len(excluded)} excluded (< {REQUIRED_TECH_REPS} tech reps)")
    if len(excluded):
        log(f"  excluded: {[f'{r.cell_line}/rep{r.bio_rep}({r.n_tech} tech)'
                            for r in excluded.itertuples()]}")
    df = df.merge(valid[["cell_line", "bio_rep"]],
                  on=["cell_line", "bio_rep"], how="inner")

    lines_ok = (valid.groupby("cell_line")["bio_rep"].nunique())
    log("[DATASETS] bio reps per line: "
        + ", ".join(f"{k}={v}" for k, v in lines_ok.items()))

    # ---- labeled lysine positions (for n_actual_biotin_sites) ---------------
    log("[MAP] mapping labeled peptides to sequence positions ...")
    seqs = load_fasta(FASTA)
    pos_cache = {}
    def site_count(acc, modseqs):
        seq = seqs.get(acc)
        if not seq:
            return np.nan
        key = (acc, tuple(sorted(modseqs)))
        if key in pos_cache:
            return pos_cache[key]
        positions = set()
        for ms in modseqs:
            clean, offsets = parse_modified_peptide(ms)
            start = seq.find(clean)
            for off in offsets:
                if start >= 0:
                    positions.add(start + off + 1)
        pos_cache[key] = len(positions) if positions else np.nan
        return pos_cache[key]

    # ---- aggregate per dataset ----------------------------------------------
    log("[AGG] building (cell line, bio rep, protein) rows ...")
    rows = []
    grp = df.groupby(["cell_line", "bio_rep", "uniprot"], sort=True)
    for (cl, br, acc), sub in grp:
        tech_detect = sub.groupby("tech_rep").size()
        row = {
            "cell_line": cl, "bio_rep": br, "uniprot_accession": acc,
            "n_ms_biotin_observations": len(sub),
            "n_biotin_peptides": sub[mod_col].nunique(),
            "n_actual_biotin_sites": site_count(acc, sub[mod_col].unique()),
            "techrep_detect_count": len(tech_detect),
            "techrep_detect_frac": len(tech_detect) / REQUIRED_TECH_REPS,
            "techrep_detect_all3": int(len(tech_detect) >= REQUIRED_TECH_REPS),
        }
        if pep_col is not None:
            row["pep_min"] = sub[pep_col].min()
            row["pep_mean"] = sub[pep_col].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    log(f"[AGG] {len(out):,} dataset rows "
        f"({out['cell_line'].nunique()} lines x "
        f"{out.groupby('cell_line')['bio_rep'].nunique().median():.0f} "
        f"bio reps median)")

    # ---- target + per-protein structural features ---------------------------
    pool = pd.read_csv(POOL_CSV)
    uni_c = find_col(pool.columns, ["uniprot_accession", "uniprot"])
    tgt_c = find_col(pool.columns, ["target_surface", "target"])
    tgt = pool[[uni_c, tgt_c]].drop_duplicates(subset=uni_c)
    tgt.columns = ["uniprot_accession", "target_surface"]
    out = out.merge(tgt, on="uniprot_accession", how="left")
    log(f"[MERGE] target_surface coverage: "
        f"{out['target_surface'].notna().mean():.1%}")

    v3 = pd.read_csv(V3_CSV)
    v3u = find_col(v3.columns, ["uniprot_accession", "uniprot"])
    v3 = v3.rename(columns={v3u: "uniprot_accession"})
    have_v3 = [f for f in V3_FEATURES if f in v3.columns]
    out = out.merge(v3[["uniprot_accession"] + have_v3],
                    on="uniprot_accession", how="left")

    memb = pd.read_csv(MEMB_CSV)
    mu = find_col(memb.columns, ["uniprot_accession", "uniprot"])
    memb = memb.rename(columns={mu: "uniprot_accession"})
    have_memb = [f for f in MEMB_FEATURES if f in memb.columns]
    out = out.merge(memb[["uniprot_accession"] + have_memb],
                    on="uniprot_accession", how="left")

    missing = [f for f in V3_FEATURES + MEMB_FEATURES
               if f not in out.columns]
    if missing:
        log(f"[WARN] features not found in v3/memb tables: {missing}")

    out = out.dropna(subset=["target_surface"])
    out["target_surface"] = out["target_surface"].astype(int)
    out.to_csv(OUT_CSV, index=False)
    log(f"[SAVED] {OUT_CSV} ({len(out):,} rows) in {time.time()-t0:.0f}s")
    log("[NEXT] python3 evaluate_linesplit.py")


if __name__ == "__main__":
    main()
