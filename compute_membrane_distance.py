#!/usr/bin/env python3
"""
compute_membrane_distance.py

Computes membrane plane distance for biotinylated peptides using AlphaFold PDBs.
Also saves membrane_planes.pkl for topology fraction features in
build_local_site_features_v2.py.

Approach (TmAlphaFold-inspired):
  1. Parse PDB + DSSP
  2. Identify TM helices: continuous H/G/I stretches >= 15 residues with 
     mean Kyte-Doolittle hydrophobicity > 0.4
  3. Fit membrane plane: centroid of TM CA atoms, normal = first PC of TM CA coords
  4. For each peptide (mapped to FASTA), get CA coordinates
  5. Compute absolute perpendicular distance from membrane plane

Output features (per protein):
  mean_peptide_membrane_dist  : mean |Z| distance of peptide CA from membrane center
  min_peptide_membrane_dist   : closest approach (TM helices = small)
  max_peptide_membrane_dist   : farthest from membrane (loops = large)
  n_tm_helices                : number of detected TM helices
  tm_helix_axis_x/y/z         : membrane normal vector components
  is_tm_protein               : 1 if n_tm_helices > 0

Output pickle (for topology fractions):
  model_features/membrane_planes.pkl
    {uniprot_id: {"ca_coords": (N,3) array, "normal": (3,), "point": (3,), "tm_helices": [idx, ...]}}

Input:
  features_biotin_pool.csv
  pmsm_results.all.parquet
  human_proteome.fasta
  alphafold_pdbs/{uid}.pdb

Output:
  model_features/membrane_distance_features.csv
  model_features/membrane_planes.pkl
"""
import os
import sys
import re
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser, DSSP
except ImportError:
    print("[ERROR] pip install biopython")
    sys.exit(1)

# DEFAULTS (all overridable via CLI; paths relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_CSV  = os.path.join(BASE_DIR, "model_features", "features_biotin_pool.csv")
PARQUET_PATH  = os.path.join(BASE_DIR, "data", "pmsm_results.all.parquet")
FASTA_PATH    = os.path.join(BASE_DIR, "data", "human_proteome.fasta")
PDB_CACHE     = os.path.join(BASE_DIR, "alphafold_pdbs")
OUT_CSV       = os.path.join(BASE_DIR, "model_features", "membrane_distance_features.csv")
PLANE_PKL     = os.path.join(BASE_DIR, "model_features", "membrane_planes.pkl")
DSSP_BIN      = "mkdssp"

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_clean_sequence(seq):
    if pd.isna(seq):
        return None
    s = str(seq).strip().strip('_')
    s = re.sub(r'\([^)]*UniMod:\d+[^)]*\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\[UNIMOD:\d+\]', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\(UniMod:\d+\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\(cam\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\(ox\)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\[[^\]]*\]', '', s)
    s = re.sub(r'[^A-Z]', '', s)
    return s

def has_biotin_mod(seq):
    if pd.isna(seq):
        return False
    s = str(seq)
    return 'UniMod:293' in s or 'UNIMOD:293' in s or 'biotin' in s.lower()

def extract_uniprot_acc(protein_id):
    if pd.isna(protein_id):
        return None
    s = str(protein_id).strip()
    m = re.search(r'[sptr]\|([A-Z0-9]{6,10})\|', s, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z0-9]{6,10})\b', s)
    if m:
        return m.group(1)
    return s

def parse_fasta(path):
    seqs = {}
    current_acc = None
    current_seq = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_acc:
                    seqs[current_acc] = ''.join(current_seq)
                header = line[1:].split()[0]
                current_acc = extract_uniprot_acc(header)
                current_seq = []
            else:
                current_seq.append(line)
        if current_acc:
            seqs[current_acc] = ''.join(current_seq)
    return seqs

def find_peptide_in_protein(prot_seq, pep_seq):
    if not prot_seq or not pep_seq:
        return -1
    idx = prot_seq.find(pep_seq)
    if idx != -1:
        return idx
    prot_li = prot_seq.replace('I', 'L')
    pep_li = pep_seq.replace('I', 'L')
    return prot_li.find(pep_li)

def mean_hydrophobicity(seq):
    if not seq:
        return 0.0
    vals = [KD_SCALE.get(aa, 0.0) for aa in seq]
    return np.mean(vals) if vals else 0.0

def fit_membrane_plane(ca_coords):
    """
    Fit membrane plane to TM helical CA coordinates.
    Returns: centroid (3,), normal (3,) — normal points along helix axis
    """
    if len(ca_coords) < 3:
        return None, None

    coords = np.array(ca_coords)  # shape (N, 3)
    centroid = np.mean(coords, axis=0)

    # Center coordinates
    centered = coords - centroid

    # SVD: first singular vector = direction of maximum variance
    # For TM helices, this is the helix axis direction
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[0]  # first right singular vector = first PC

    # Ensure consistent orientation (point in general direction of positive Z)
    if normal[2] < 0:
        normal = -normal

    return centroid, normal

def parse_pdb_with_coords(pdb_path, uid):
    """Parse PDB, return {residue_index: (x, y, z)} for CA atoms."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(uid, pdb_path)
    except Exception:
        return None, None, None

    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        return None, None, None
    chain = chains[0]

    # CA coordinates, indexed by residue position (0-based)
    ca_dict = {}
    res_idx = 0
    for res in chain.get_residues():
        if res.id[0] == ' ' and 'CA' in res:
            ca = res['CA']
            ca_dict[res_idx] = np.array(ca.get_coord())
            res_idx += 1

    # DSSP for secondary structure
    ss_dict = {}
    dssp_ok = False
    try:
        dssp = DSSP(model, pdb_path, dssp=DSSP_BIN)
        res_idx = 0
        for key in dssp.keys():
            entry = dssp[key]
            ss_dict[res_idx] = entry[2]  # secondary structure
            res_idx += 1
        dssp_ok = True
    except Exception:
        pass

    return ca_dict, ss_dict, dssp_ok

def identify_tm_helices(ss_dict, ca_dict, fasta_seq, min_len=15, hydro_threshold=0.4):
    """
    Identify TM helices: continuous H/G/I stretches with length >= min_len
    and mean hydrophobicity >= hydro_threshold.
    Returns list of (start, end) tuples (0-based, inclusive start, exclusive end).
    """
    if not ss_dict or not fasta_seq:
        return []

    # Sort by position
    positions = sorted(ss_dict.keys())

    helices = []
    current_start = None
    current_len = 0

    for pos in positions:
        ss = ss_dict[pos]
        if ss in ('H', 'G', 'I'):
            if current_start is None:
                current_start = pos
                current_len = 1
            else:
                current_len += 1
        else:
            if current_start is not None and current_len >= min_len:
                helices.append((current_start, current_start + current_len))
            current_start = None
            current_len = 0

    # Check last stretch
    if current_start is not None and current_len >= min_len:
        helices.append((current_start, current_start + current_len))

    # Filter by hydrophobicity
    tm_helices = []
    for start, end in helices:
        if end <= len(fasta_seq):
            seq = fasta_seq[start:end]
            hydro = mean_hydrophobicity(seq)
            if hydro >= hydro_threshold:
                tm_helices.append((start, end))

    return tm_helices

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global FEATURES_CSV, PARQUET_PATH, FASTA_PATH, PDB_CACHE, OUT_CSV, PLANE_PKL, DSSP_BIN

    ap = argparse.ArgumentParser(description="Compute membrane-plane distances for biotin-labeled sites")
    ap.add_argument("--base-dir", default=BASE_DIR,
                    help="repo root; all other defaults resolve relative to this")
    ap.add_argument("--features-csv", default=None,
                    help="input pool CSV (default: <base>/model_features/features_biotin_pool.csv)")
    ap.add_argument("--parquet", default=None,
                    help="raw MaxQuant parquet (default: <base>/data/pmsm_results.all.parquet)")
    ap.add_argument("--fasta", default=None,
                    help="proteome FASTA (default: <base>/data/human_proteome.fasta)")
    ap.add_argument("--pdb-dir", default=None,
                    help="AlphaFold PDB cache (default: <base>/alphafold_pdbs)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <base>/model_features/membrane_distance_features.csv)")
    ap.add_argument("--dssp-bin", default=DSSP_BIN,
                    help="mkdssp executable name or full path")
    args = ap.parse_args()

    base = args.base_dir
    FEATURES_CSV = args.features_csv or os.path.join(base, "model_features", "features_biotin_pool.csv")
    PARQUET_PATH = args.parquet or os.path.join(base, "data", "pmsm_results.all.parquet")
    FASTA_PATH   = args.fasta or os.path.join(base, "data", "human_proteome.fasta")
    PDB_CACHE    = args.pdb_dir or os.path.join(base, "alphafold_pdbs")
    OUT_CSV      = args.out or os.path.join(base, "model_features", "membrane_distance_features.csv")
    PLANE_PKL    = os.path.join(os.path.dirname(os.path.abspath(OUT_CSV)), "membrane_planes.pkl")
    DSSP_BIN     = args.dssp_bin

    os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)

    print("=" * 70)
    print("Computing membrane plane distances")
    print("=" * 70)

    # 1. Load protein list
    df_pool = pd.read_csv(FEATURES_CSV)
    uids = sorted(df_pool['uniprot_accession'].dropna().unique())
    print(f"[LOAD] features_biotin_pool: {len(uids):,} proteins")

    # 2. Load biotin peptides
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET_PATH)
    all_cols = pf.schema.names

    pep_col = prot_col = None
    for c in all_cols:
        cl = c.lower().replace(' ', '').replace('_', '')
        if cl in ('peptidesequence', 'sequence', 'modifiedpeptide', 'modifiedsequence', 'peptide'):
            pep_col = c
        if cl in ('leadingrazorprotein', 'protein', 'proteins', 'proteinids', 'masterprotein'):
            prot_col = c

    if not pep_col:
        for c in all_cols:
            if 'peptide' in c.lower() or 'seq' in c.lower():
                pep_col = c; break
    if not prot_col:
        for c in all_cols:
            if 'protein' in c.lower():
                prot_col = c; break

    cols_to_read = [c for c in [pep_col, prot_col] if c]
    df_pep = pd.read_parquet(PARQUET_PATH, columns=cols_to_read, engine='pyarrow')
    df_pep['_has_biotin'] = df_pep[pep_col].apply(has_biotin_mod)
    df_pep = df_pep[df_pep['_has_biotin']].copy()
    df_pep.drop(columns=['_has_biotin'], inplace=True)
    df_pep['peptide_clean'] = df_pep[pep_col].apply(extract_clean_sequence)
    df_pep['protein_acc'] = df_pep[prot_col].apply(extract_uniprot_acc)
    df_pep = df_pep[df_pep['peptide_clean'].notna() & df_pep['protein_acc'].notna()].copy()
    df_pep = df_pep[df_pep['peptide_clean'].str.len() >= 2]
    print(f"[INFO] {len(df_pep):,} biotin peptides")

    peptides_by_protein = defaultdict(list)
    for _, row in df_pep.iterrows():
        peptides_by_protein[row['protein_acc']].append(row['peptide_clean'])
    print(f"[INFO] {len(peptides_by_protein):,} proteins with peptides")

    # 3. Load FASTA
    fasta_map = parse_fasta(FASTA_PATH)
    print(f"[LOAD] FASTA: {len(fasta_map):,} sequences")

    # 4. Process each protein
    results = []
    membrane_planes = {}  # <-- NEW: accumulate for pickle
    n_missing_pdb = 0
    n_no_peptides = 0
    n_no_fasta = 0
    n_notm = 0
    n_dssp_fail = 0

    for i, uid in enumerate(uids, 1):
        if i % 100 == 0 or i == len(uids):
            print(f"  [{i}/{len(uids)}] {uid}  (missing={n_missing_pdb}, no_fasta={n_no_fasta}, no_pep={n_no_peptides}, notm={n_notm}, dssp_fail={n_dssp_fail})")

        peptides = peptides_by_protein.get(uid, [])
        if not peptides:
            n_no_peptides += 1
            continue

        prot_seq = fasta_map.get(uid)
        if not prot_seq:
            n_no_fasta += 1
            continue

        # Map peptides to positions
        peptide_regions = []
        for pep in peptides:
            idx = find_peptide_in_protein(prot_seq, pep)
            if idx >= 0:
                peptide_regions.append((idx, idx + len(pep)))

        if not peptide_regions:
            continue

        # Load PDB
        pdb_path = os.path.join(PDB_CACHE, f"{uid}.pdb")
        if not os.path.exists(pdb_path):
            n_missing_pdb += 1
            continue

        ca_dict, ss_dict, dssp_ok = parse_pdb_with_coords(pdb_path, uid)
        if ca_dict is None:
            n_missing_pdb += 1
            continue
        if not dssp_ok:
            n_dssp_fail += 1

        # Identify TM helices
        tm_helices = identify_tm_helices(ss_dict, ca_dict, prot_seq)
        n_tm_helices = len(tm_helices)

        if n_tm_helices == 0:
            n_notm += 1
            # For non-TM proteins, membrane distance is undefined
            results.append({
                'uniprot_accession': uid,
                'is_tm_protein': 0,
                'n_tm_helices': 0,
                'tm_helix_axis_x': np.nan,
                'tm_helix_axis_y': np.nan,
                'tm_helix_axis_z': np.nan,
                'mean_peptide_membrane_dist': np.nan,
                'min_peptide_membrane_dist': np.nan,
                'max_peptide_membrane_dist': np.nan,
                'std_peptide_membrane_dist': np.nan,
                'dssp_ok': dssp_ok,
            })
            continue

        # Collect TM CA coordinates
        tm_ca_coords = []
        for start, end in tm_helices:
            for pos in range(start, end):
                if pos in ca_dict:
                    tm_ca_coords.append(ca_dict[pos])

        if len(tm_ca_coords) < 3:
            n_notm += 1
            continue

        # Fit membrane plane
        centroid, normal = fit_membrane_plane(tm_ca_coords)
        if centroid is None:
            n_notm += 1
            continue

        # Compute peptide distances from membrane plane
        peptide_dists = []
        for start, end in peptide_regions:
            for pos in range(start, end):
                if pos in ca_dict:
                    # Absolute perpendicular distance from plane
                    vec = ca_dict[pos] - centroid
                    dist = abs(np.dot(vec, normal))
                    peptide_dists.append(dist)

        if not peptide_dists:
            continue

        results.append({
            'uniprot_accession': uid,
            'is_tm_protein': 1,
            'n_tm_helices': n_tm_helices,
            'tm_helix_axis_x': round(normal[0], 4),
            'tm_helix_axis_y': round(normal[1], 4),
            'tm_helix_axis_z': round(normal[2], 4),
            'mean_peptide_membrane_dist': round(float(np.mean(peptide_dists)), 2),
            'min_peptide_membrane_dist': round(float(np.min(peptide_dists)), 2),
            'max_peptide_membrane_dist': round(float(np.max(peptide_dists)), 2),
            'std_peptide_membrane_dist': round(float(np.std(peptide_dists)), 2) if len(peptide_dists) > 1 else 0.0,
            'dssp_ok': dssp_ok,
        })

        # -----------------------------------------------------------------
        # NEW: Save membrane plane data for topology fraction features
        # -----------------------------------------------------------------
        # Convert ca_dict to contiguous (N, 3) array
        n_res = len(ca_dict)
        ca_coords_arr = np.zeros((n_res, 3))
        for pos in range(n_res):
            if pos in ca_dict:
                ca_coords_arr[pos] = ca_dict[pos]

        # Convert (start, end) tuples to flat list of residue indices
        tm_helix_indices = []
        for start, end in tm_helices:
            tm_helix_indices.extend(range(start, end))

        membrane_planes[uid] = {
            "ca_coords": ca_coords_arr,
            "normal": normal,
            "point": centroid,
            "tm_helices": tm_helix_indices,
        }

    if not results:
        print("[FATAL] No results generated.")
        sys.exit(1)

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[SAVED] {OUT_CSV}")
    print(f"[INFO] {len(df_out):,} proteins processed")
    print(f"[INFO] TM proteins: {df_out['is_tm_protein'].sum():,}")
    print(f"[INFO] Non-TM proteins: {(df_out['is_tm_protein'] == 0).sum():,}")
    print(f"[INFO] Missing PDB: {n_missing_pdb}, No FASTA: {n_no_fasta}, No peptides: {n_no_peptides}")

    # -----------------------------------------------------------------
    # NEW: Save membrane_planes.pkl
    # -----------------------------------------------------------------
    with open(PLANE_PKL, "wb") as f:
        pickle.dump(membrane_planes, f)
    print(f"[SAVED] {PLANE_PKL}  ({len(membrane_planes):,} proteins with membrane planes)")
    print("[DONE]")

if __name__ == '__main__':
    main()
