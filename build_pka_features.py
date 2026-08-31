#!/usr/bin/env python3
"""
build_pka_features.py — lysine-reactivity (pKa) features via PROPKA 3.

Motivation: Muraoka et al. 2018 (Anal. Biochem.) — NHS-biotin labels only the
DEprotonated lysine epsilon-NH2; labeling propensity depends on solvent
accessibility, electrostatics, H-bonds, and pKa. Our Top-20 v3 features have
accessibility/geometry but nothing about intrinsic chemical reactivity.

  f_deprot(pKa, pH) = 1 / (1 + 10^(pKa - pH))     # reactive NH2 fraction

Labeling pH defaults to 7.2 (BERTIS SOP: PBS pH 7.2 labeling, TBS pH 7.2 quench).

PROTEIN-LEVEL features (need only AlphaFold PDBs; per-protein constants,
merged downstream by uniprot_accession — same status as the v3 structural
features; computed from structure only, no labels involved):
  pka_n_lys            lysines in the AlphaFold model
  pka_mean/pka_min/pka_std
  pka_n_reactive       lysines with pKa < 10.0 (>=10x baseline reactivity)
  pka_frac_reactive    pka_n_reactive / pka_n_lys
  pka_fdeprot_sum      total intrinsic labelability at the labeling pH
  pka_fdeprot_mean/pka_fdeprot_max

SITE-LEVEL features (only with --sites CSV of labeled-lysine positions,
columns: uniprot_accession,site_resnum — 1-based UniProt/AlphaFold numbering;
derive from the same UniMod:293 peptide mapping as build_local_site_features_v3):
  site_pka_mean/site_pka_min
  site_fdeprot_sum/site_fdeprot_max

Usage:
  # on the workstation (PDBs already local):
  python3 build_pka_features.py --datasets model_features/biorep_datasets.csv \
      --pdb-dir /path/to/alphafold_pdbs --out model_features/pka_features.csv

  # anywhere (downloads AlphaFold v6 models):
  python3 build_pka_features.py --datasets biorep_datasets.csv --download \
      --pdb-dir af_pdbs --jobs 4 --out pka_features.csv

  # add site-level features:
  python3 build_pka_features.py ... --sites labeled_lysine_sites.csv

Requires: propka3 (pip install propka). Point --propka at the propka3 script
if it is not on PATH. Run with PYTHONPATH set if propka is in a --target dir.
"""

import argparse
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{v}.pdb"
RE_ACTIVE_PKA = 10.0     # pKa below this = >=10x baseline reactivity at pH 7.2


def f_deprot(pka, ph):
    return 1.0 / (1.0 + 10.0 ** (pka - ph))


def find_propka(user_path):
    if user_path:
        return user_path
    from shutil import which
    w = which("propka3")
    if w:
        return w
    sys.exit("[FATAL] propka3 not found on PATH. pip install propka (or pass --propka).")


def download_pdb(acc, pdb_dir):
    for v in (6, 4):
        url = AF_URL.format(acc=acc, v=v)
        dst = os.path.join(pdb_dir, f"AF-{acc}-F1-model_v{v}.pdb")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data.startswith(b"ATOM") or b"\nATOM" in data[:500]:
                with open(dst, "wb") as f:
                    f.write(data)
                return dst
        except Exception:
            continue
    return None


def find_pdb(acc, pdb_dir):
    # project convention (build_local_site_features_v3.py): <ACCESSION>.pdb
    p = os.path.join(pdb_dir, f"{acc}.pdb")
    if os.path.exists(p):
        return p
    for v in (6, 4, 3, 2, 1):
        p = os.path.join(pdb_dir, f"AF-{acc}-F1-model_v{v}.pdb")
        if os.path.exists(p):
            return p
    # tolerate any AF-<acc>*.pdb naming
    try:
        for fn in sorted(os.listdir(pdb_dir)):
            if fn.startswith(f"AF-{acc}") and fn.endswith(".pdb"):
                return os.path.join(pdb_dir, fn)
    except FileNotFoundError:
        pass
    return None


PKA_LINE = re.compile(r"\s*LYS\s+(\d+)\s+(\w)\s+([\d.]+)\s+([\d.]+)")


def parse_propka_lys(pka_file):
    """Return dict resnum -> predicted pKa for all lysines (SUMMARY section)."""
    lys, in_summary = {}, False
    with open(pka_file) as fh:
        for line in fh:
            if "SUMMARY OF THIS PREDICTION" in line:
                in_summary = True
                continue
            if in_summary:
                if not line.strip():
                    break
                m = PKA_LINE.match(line)
                if m:
                    lys[int(m.group(1))] = float(m.group(3))
    return lys


def run_propka(acc, pdb_path, propka_bin, workdir):
    """Run propka3 on pdb_path; return dict resnum -> pKa (cached via .pka).

    Each accession gets its own subdirectory: propka writes <stem>.pka into
    the CWD, and concurrent same-directory writes are flaky on some mounts.
    """
    stem = os.path.splitext(os.path.basename(pdb_path))[0]
    acc_dir = os.path.join(workdir, acc)
    os.makedirs(acc_dir, exist_ok=True)
    pka_file = os.path.join(acc_dir, stem + ".pka")
    if not os.path.exists(pka_file):
        r = subprocess.run([sys.executable, propka_bin, pdb_path],
                           cwd=acc_dir, capture_output=True, text=True,
                           timeout=900)
        if not os.path.exists(pka_file):
            raise RuntimeError(f"propka failed for {acc}: {r.stderr[-300:]}")
    return parse_propka_lys(pka_file)


def protein_features(acc, pdb_dir, propka_bin, workdir, ph, do_download):
    pdb = find_pdb(acc, pdb_dir)
    if pdb is None and do_download:
        pdb = download_pdb(acc, pdb_dir)
    if pdb is None:
        return acc, None, "no PDB"
    try:
        lys = run_propka(acc, pdb, propka_bin, workdir)
    except Exception as e:
        return acc, None, str(e)[:120]
    if not lys:
        return acc, None, "no lysines parsed"
    pkas = np.array(list(lys.values()))
    fd = f_deprot(pkas, ph)
    feats = {
        "pka_n_lys": len(pkas),
        "pka_mean": float(pkas.mean()),
        "pka_min": float(pkas.min()),
        "pka_std": float(pkas.std()),
        "pka_n_reactive": int((pkas < RE_ACTIVE_PKA).sum()),
        "pka_frac_reactive": float((pkas < RE_ACTIVE_PKA).mean()),
        "pka_fdeprot_sum": float(fd.sum()),
        "pka_fdeprot_mean": float(fd.mean()),
        "pka_fdeprot_max": float(fd.max()),
        "_lys": lys,          # internal: per-residue pKa for site-level
    }
    return acc, feats, "ok"


def main():
    _here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", help="biorep_datasets.csv (accession list)")
    ap.add_argument("--accessions", help="text file, one accession per line")
    ap.add_argument("--pdb-dir", default=os.path.join(_here, "af_pdbs"),
                    help="AlphaFold PDB cache (AF-<acc>-F1-model_v*.pdb naming)")
    ap.add_argument("--download", action="store_true",
                    help="download missing AlphaFold v6 models")
    ap.add_argument("--sites", help="CSV: uniprot_accession,site_resnum")
    ap.add_argument("--ph", type=float, default=7.2, help="labeling pH (SOP 7.2)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--propka", help="path to propka3 executable (default: found on PATH)")
    ap.add_argument("--workdir", default=os.path.join(_here, "propka_work"))
    ap.add_argument("--out", default=os.path.join(_here, "model_features", "pka_features.csv"))
    args = ap.parse_args()

    if args.datasets:
        accs = sorted(pd.read_csv(args.datasets, usecols=["uniprot_accession"])
                      ["uniprot_accession"].dropna().astype(str).unique())
    elif args.accessions:
        accs = sorted({ln.strip() for ln in open(args.accessions) if ln.strip()})
    else:
        sys.exit("[FATAL] pass --datasets or --accessions")
    print(f"[LOAD] {len(accs)} unique accessions")

    os.makedirs(args.pdb_dir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    propka_bin = find_propka(args.propka)
    print(f"[PROPKA] {propka_bin} | labeling pH = {args.ph} | jobs = {args.jobs}")

    rows, fails = {}, []
    from collections import Counter
    fail_reasons = Counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(protein_features, a, args.pdb_dir, propka_bin,
                          args.workdir, args.ph, args.download): a
                for a in accs}
        for i, fut in enumerate(as_completed(futs), 1):
            acc, feats, status = fut.result()
            if feats is None:
                fails.append((acc, status))
                fail_reasons[status] += 1
                if len(fails) <= 5:
                    print(f"  [FAIL] {acc}: {status}")
            else:
                rows[acc] = feats
            if i % 250 == 0 or i == len(futs):
                print(f"  [{i}/{len(futs)}] done, {len(fails)} failed")
                if fail_reasons:
                    print("    failure breakdown: "
                          + ", ".join(f"{r} x{n}" for r, n
                                      in fail_reasons.most_common(5)))

    out = pd.DataFrame([{"uniprot_accession": a, **{k: v for k, v in f.items()
                                                     if k != "_lys"}}
                        for a, f in rows.items()])

    if args.sites:
        sites = pd.read_csv(args.sites)
        srows = []
        for acc, g in sites.groupby("uniprot_accession"):
            lys = rows.get(acc, {}).get("_lys")
            if not lys:
                continue
            sp = np.array([lys[int(r)] for r in g["site_resnum"]
                           if int(r) in lys])
            if len(sp) == 0:
                continue
            fd = f_deprot(sp, args.ph)
            srows.append({"uniprot_accession": acc,
                          "site_pka_mean": float(sp.mean()),
                          "site_pka_min": float(sp.min()),
                          "site_fdeprot_sum": float(fd.sum()),
                          "site_fdeprot_max": float(fd.max())})
        sdf = pd.DataFrame(srows, columns=["uniprot_accession", "site_pka_mean",
                                           "site_pka_min", "site_fdeprot_sum",
                                           "site_fdeprot_max"])
        out = out.merge(sdf, on="uniprot_accession", how="left")
        print(f"[SITES] site-level features for {len(sdf)} proteins")

    out.to_csv(args.out, index=False)
    print(f"[SAVED] {args.out}: {out.shape[0]} proteins x "
          f"{out.shape[1] - 1} features")
    if fails:
        print(f"[WARN] {len(fails)} accessions failed (see below, first 15):")
        for a, s in fails[:15]:
            print(f"   {a}: {s}")


if __name__ == "__main__":
    main()
