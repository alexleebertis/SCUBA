#!/usr/bin/env python3
"""
fetch_alphafold_structural_features.py  (v2 — adds RSA, disorder, glycosites)

Re-parses cached AlphaFold PDBs and extracts:
  - avg_plddt, frac_helix, frac_sheet, frac_coil  (existing)
  - mean_rsa      : mean relative solvent accessibility (0-1, from DSSP)
  - frac_exposed  : fraction of residues with RSA > 0.20
  - frac_disordered: fraction of residues with pLDDT < 50
  - n_glycosites  : N-linked glycosylation sequons (N-X-S/T, X!=P)

Requires: biopython, numpy, pandas, requests
          mkdssp (DSSP) — pass --dssp-bin if it is not on PATH
Optional: FASTA file for full-length n_glycosites (more accurate than PDB-derived)

Output: model_features/alphafold_features.csv (see --out)
"""

import os
import sys
import time
import csv
import re
import argparse
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from Bio.PDB import PDBParser, DSSP
except ImportError:
    print("[ERROR] pip install biopython")
    sys.exit(1)

# ------------------------------------------------------------------
# DEFAULTS (all overridable via CLI; paths are relative to this script)
_HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV  = os.path.join(_HERE, "model_features", "features_biotin_pool.csv")
OUT_CSV       = os.path.join(_HERE, "model_features", "alphafold_features.csv")
PDB_CACHE     = os.path.join(_HERE, "alphafold_pdbs")
API_BASE      = "https://alphafold.ebi.ac.uk/api/prediction"
HEADERS       = {"User-Agent": "SCUBA/1.0 (surfaceome research; contact: repo owner)"}
SLEEP_SEC     = 0.15
MAX_RETRIES   = 2
N_WORKERS     = 6
DSSP_BIN      = "mkdssp"

# Optional: path to human FASTA for accurate n_glycosite counting
# If not found, script extracts sequence from PDB (may miss unresolved tails)
FASTA_PATH    = os.path.join(_HERE, "data", "uniprot_human.fasta")

# ------------------------------------------------------------------
# Helpers

def extract_canonical_uniprot(raw_id):
    if pd.isna(raw_id):
        return None
    s = str(raw_id).strip()
    if '|' in s:
        s = s.split('|')[1]
    s = s.split('-')[0]
    if re.fullmatch(r'[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]', s):
        return s
    return None


def get_pdb_url(uid):
    url = f"{API_BASE}/{uid}"
    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list):
                    return data[0].get("pdbUrl")
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    return None


def download_pdb(pdb_url, out_path):
    for _ in range(MAX_RETRIES):
        try:
            r = requests.get(pdb_url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                with open(out_path, 'w') as f:
                    f.write(r.text)
                return True
            time.sleep(SLEEP_SEC)
        except Exception:
            time.sleep(SLEEP_SEC)
    return False


def count_glycosites(seq):
    """Count N-X-S/T motifs where X is not Pro."""
    if not seq or len(seq) < 3:
        return 0
    count = 0
    for i in range(len(seq) - 2):
        if seq[i] == 'N' and seq[i+1] != 'P' and seq[i+2] in ('S', 'T'):
            count += 1
    return count


def parse_structure(pdb_path, uid, fasta_seq=None):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(uid, pdb_path)
    except Exception:
        return None

    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        return None
    chain = chains[0]

    # --- pLDDT from CA B-factors ---
    plddts = []
    for res in chain.get_residues():
        if res.id[0] == ' ' and 'CA' in res:
            plddts.append(res['CA'].get_bfactor())
    if not plddts:
        return None
    avg_plddt = float(np.mean(plddts))
    frac_disordered = float(np.mean(np.array(plddts) < 50))

    # --- Secondary structure + RSA via DSSP ---
    frac_helix = frac_sheet = frac_coil = np.nan
    mean_rsa = frac_exposed = np.nan
    dssp_ok = False
    ss_list = []
    rsa_list = []
    aa_list = []  # for PDB-derived sequence if FASTA missing

    try:
        dssp = DSSP(model, pdb_path, dssp=DSSP_BIN)
        for key in dssp.keys():
            entry = dssp[key]
            ss_list.append(entry[2])          # secondary structure
            rsa_list.append(float(entry[3]))  # relative accessibility
            aa_list.append(entry[1])          # amino acid
            dssp_ok = True
    except Exception:
        pass

    if ss_list:
        ss_arr = np.array(ss_list)
        frac_helix = float(np.mean((ss_arr == 'H') | (ss_arr == 'G') | (ss_arr == 'I')))
        frac_sheet = float(np.mean(ss_arr == 'E'))
        frac_coil  = float(np.mean((ss_arr == 'C') | (ss_arr == 'T') | (ss_arr == 'S') | (ss_arr == 'B')))

    if rsa_list:
        rsa_arr = np.array(rsa_list)
        mean_rsa = float(np.mean(rsa_arr))
        frac_exposed = float(np.mean(rsa_arr > 0.20))

    # --- N-glycosites ---
    if fasta_seq:
        n_glycosites = count_glycosites(fasta_seq)
    else:
        # Fallback: use PDB-derived sequence (may miss unresolved tails)
        seq = ''.join(aa_list)
        n_glycosites = count_glycosites(seq)

    return {
        'uniprot_accession': uid,
        'avg_plddt': round(avg_plddt, 3),
        'frac_helix': round(frac_helix, 4) if not np.isnan(frac_helix) else None,
        'frac_sheet': round(frac_sheet, 4) if not np.isnan(frac_sheet) else None,
        'frac_coil':  round(frac_coil, 4) if not np.isnan(frac_coil) else None,
        'mean_rsa': round(mean_rsa, 4) if not np.isnan(mean_rsa) else None,
        'frac_exposed': round(frac_exposed, 4) if not np.isnan(frac_exposed) else None,
        'frac_disordered': round(frac_disordered, 4),
        'n_glycosites': n_glycosites,
        'n_residues': len(plddts),
        'dssp_used': dssp_ok,
    }


def process_one(uid, fasta_dict):
    pdb_path = os.path.join(PDB_CACHE, f"{uid}.pdb")
    if not os.path.exists(pdb_path):
        pdb_url = get_pdb_url(uid)
        if pdb_url is None:
            return {"_notfound": uid}
        if not download_pdb(pdb_url, pdb_path):
            return None
        time.sleep(SLEEP_SEC)

    fasta_seq = fasta_dict.get(uid) if fasta_dict else None
    return parse_structure(pdb_path, uid, fasta_seq=fasta_seq)


def load_fasta_dict(fasta_path):
    """Return dict {uniprot_id: sequence} from FASTA."""
    seq_dict = {}
    if not os.path.exists(fasta_path):
        return seq_dict

    current_id = None
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id and current_seq:
                    seq_dict[current_id] = ''.join(current_seq)
                # Parse header: >sp|P12345|... or >P12345 ...
                header = line[1:].split()[0]
                if '|' in header:
                    current_id = header.split('|')[1]
                else:
                    current_id = header
                current_seq = []
            else:
                current_seq.append(line)
        if current_id and current_seq:
            seq_dict[current_id] = ''.join(current_seq)
    return seq_dict


# ------------------------------------------------------------------
# Main

def main():
    global FEATURES_CSV, OUT_CSV, PDB_CACHE, FASTA_PATH, SLEEP_SEC, N_WORKERS, DSSP_BIN

    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--features-csv", default=FEATURES_CSV,
                    help="input pool CSV with a uniprot_accession column")
    ap.add_argument("--out", default=OUT_CSV, help="output CSV path")
    ap.add_argument("--pdb-dir", default=PDB_CACHE,
                    help="AlphaFold PDB cache directory (created if missing)")
    ap.add_argument("--fasta", default=FASTA_PATH,
                    help="human proteome FASTA for glycosite counting (optional)")
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help="parallel download/parse threads")
    ap.add_argument("--sleep", type=float, default=SLEEP_SEC,
                    help="politeness delay between API calls (seconds)")
    ap.add_argument("--dssp-bin", default=DSSP_BIN,
                    help="mkdssp executable name or full path")
    args = ap.parse_args()

    FEATURES_CSV, OUT_CSV = args.features_csv, args.out
    PDB_CACHE, FASTA_PATH = args.pdb_dir, args.fasta
    SLEEP_SEC, N_WORKERS, DSSP_BIN = args.sleep, args.workers, args.dssp_bin

    os.makedirs(PDB_CACHE, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)

    print("=" * 60)
    print("AlphaFold Structural Feature Fetcher  (v2 — RSA + disorder + glycosites)")
    print("=" * 60)

    # 0. API test
    print("[TEST] P04637...")
    if not get_pdb_url("P04637"):
        print("[ABORT] API down.")
        sys.exit(1)
    print("       API OK")

    # 1. Load IDs
    if not os.path.exists(FEATURES_CSV):
        print(f"[ERROR] {FEATURES_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    raw_ids = df['uniprot_accession'].dropna().unique()
    uids = sorted(set(filter(None, [extract_canonical_uniprot(x) for x in raw_ids])))
    print(f"[LOAD] {len(uids)} canonical UniProt accessions")

    # 2. Load FASTA (optional)
    fasta_dict = load_fasta_dict(FASTA_PATH)
    if fasta_dict:
        print(f"[FASTA] Loaded {len(fasta_dict)} sequences from {FASTA_PATH}")
    else:
        print(f"[FASTA] Not found at {FASTA_PATH}")
        print("        Will use PDB-derived sequence (may miss unresolved tails).")
        print("        For accurate glycosite counts, provide a human FASTA.")

    # 3. Determine todo list
    # We ALWAYS re-parse existing PDBs to extract the new features
    todo = uids
    print(f"[TODO] {len(todo)} structures to parse")

    # 4. Threaded bulk process (overwrite mode)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'uniprot_accession', 'avg_plddt', 'frac_helix', 'frac_sheet',
            'frac_coil', 'mean_rsa', 'frac_exposed', 'frac_disordered',
            'n_glycosites', 'n_residues', 'dssp_used'
        ])
        writer.writeheader()

        nf = pd_ok = fail = 0

        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            fut2uid = {ex.submit(process_one, uid, fasta_dict): uid for uid in todo}
            for i, fut in enumerate(as_completed(fut2uid), 1):
                uid = fut2uid[fut]
                try:
                    r = fut.result(timeout=120)
                except Exception as e:
                    print(f"  [{i}/{len(todo)}] {uid}: EXC {e}")
                    fail += 1
                    continue

                if r is None:
                    print(f"  [{i}/{len(todo)}] {uid}: FAIL")
                    fail += 1
                elif "_notfound" in r:
                    nf += 1
                    if nf <= 5 or nf % 100 == 0:
                        print(f"  [{i}/{len(todo)}] {uid}: NOT FOUND (missing {nf})")
                else:
                    writer.writerow(r)
                    pd_ok += 1
                    if pd_ok % 100 == 0 or i == len(todo):
                        print(f"  [{i}/{len(todo)}] {uid}: pLDDT={r['avg_plddt']:.1f} "
                              f"RSA={r['mean_rsa']:.2f} exp={r['frac_exposed']:.2f} "
                              f"dis={r['frac_disordered']:.2f} gly={r['n_glycosites']} "
                              f"(ok={pd_ok}, miss={nf}, fail={fail})")

    print(f"\n[SAVED] {OUT_CSV} | parsed={pd_ok} missing={nf} failed={fail}")
    print("[DONE]")


if __name__ == '__main__':
    main()