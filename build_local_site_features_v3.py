#!/usr/bin/env python3
"""
build_local_site_features_v3.py

Corrected local structural feature extractor for biotinylated peptides.

Major changes vs v2:
  - parses the actual UniMod:293 residue position instead of treating the whole peptide as "labeled"
  - deduplicates structural peptide locations while retaining MS observation counts
  - detects ambiguous peptide mappings and excludes ambiguous locations from structural features
  - uses PDB residue numbering to validate FASTA<->AlphaFold positional mapping
  - computes peptide geometry per peptide, then aggregates (never concatenates separate peptides)
  - replaces global surface_patch_size with peptide-local contiguous exposure
  - drops arbitrary extracellular/cytosolic labels
  - adds 8 Å local 3D neighborhood features around the modified site
  - adds Lys NZ-centered packing counts when NZ is available
"""

import os
import re
import sys
import math
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from Bio.PDB import PDBParser, DSSP
    from Bio.Data.PDBData import protein_letters_3to1
except ImportError:
    print("[ERROR] pip install biopython")
    sys.exit(1)


# mkdssp executable (name on PATH or full path); overridable via --dssp-bin
DSSP_BIN = "mkdssp"

HYDROPATHY = {
    "A": 1.80, "C": 2.50, "D": -3.50, "E": -3.50, "F": 2.80,
    "G": -0.40, "H": -3.20, "I": 4.50, "K": -3.90, "L": 3.80,
    "M": 1.90, "N": -3.50, "P": -1.60, "Q": -3.50, "R": -4.50,
    "S": -0.80, "T": -0.70, "V": 4.20, "W": -0.90, "Y": -1.30,
}
CHARGED = set("DEKRH")
HYDROPHOBIC = set("AVILMFWYC")


def extract_uniprot_acc(protein_id):
    if pd.isna(protein_id):
        return None
    s = str(protein_id).strip()
    first = s.split(";")[0].strip()
    m = re.search(r'[sptr]\|([A-Z0-9]{6,10})\|', first, re.I)
    if m:
        return m.group(1).upper()
    return first.split()[0].split("-")[0].upper()


def parse_fasta(path):
    seqs = {}
    current = None
    chunks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs[current] = "".join(chunks).upper()
                current = extract_uniprot_acc(line[1:].split()[0])
                chunks = []
            else:
                chunks.append(line)
    if current:
        seqs[current] = "".join(chunks).upper()
    return seqs


def parse_modified_peptide(text):
    """
    Return (clean_sequence, modified_local_indices).

    Supports common forms such as:
      K(UniMod:293)
      K[UNIMOD:293]
      <UniMod:293>K

    A prefix modification is attached to the next amino-acid residue.
    """
    if pd.isna(text):
        return None, []

    s = str(text).strip().strip("_")
    residues = []
    modified = []
    pending_mod = False
    i = 0

    while i < len(s):
        ch = s[i]

        if ch.isalpha() and ch.upper() == ch and len(ch) == 1:
            residues.append(ch)
            idx = len(residues) - 1
            if pending_mod:
                modified.append(idx)
                pending_mod = False
            i += 1
            continue

        if ch in "([<":
            close = { "(": ")", "[": "]", "<": ">" }[ch]
            j = s.find(close, i + 1)
            if j == -1:
                i += 1
                continue
            annot = s[i+1:j]
            if re.search(r'(?:UniMod|UNIMOD)\s*:\s*293|biotin', annot, re.I):
                # postfix annotation normally belongs to previous residue;
                # prefix <...> form belongs to next residue.
                if ch == "<" or not residues:
                    pending_mod = True
                else:
                    modified.append(len(residues) - 1)
            i = j + 1
            continue

        i += 1

    clean = "".join(residues).upper()
    return (clean or None), sorted(set(modified))


def has_biotin_mod(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'(?:UniMod|UNIMOD)\s*:\s*293|biotin', str(text), re.I))


def all_occurrences(protein, peptide):
    if not protein or not peptide:
        return []

    def find_all(a, b):
        out = []
        i = a.find(b)
        while i != -1:
            out.append(i)
            i = a.find(b, i + 1)
        return out

    exact = find_all(protein, peptide)
    if exact:
        return exact, "exact"

    # MS cannot distinguish I/L.
    p = protein.replace("I", "L")
    q = peptide.replace("I", "L")
    return find_all(p, q), "IL_equivalent"


def detect_columns(parquet_path):
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(parquet_path).schema.names
    pep = prot = None

    for c in cols:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in ("peptidesequence", "sequence", "modifiedpeptide",
                 "modifiedsequence", "peptide"):
            pep = c
        if k in ("fastaid", "leadingrazorprotein", "protein", "proteins",
                 "proteinids", "masterprotein"):
            prot = c

    if pep is None:
        pep = next((c for c in cols if "peptide" in c.lower() or "seq" in c.lower()), None)
    if prot is None:
        prot = next((c for c in cols if "protein" in c.lower() or "fasta" in c.lower()), None)

    if pep is None or prot is None:
        raise RuntimeError(f"Could not detect peptide/protein columns: {cols}")
    return pep, prot


def three_to_one(resname):
    r = str(resname).upper()
    # BioPython mapping keys may be title-cased depending on version.
    return protein_letters_3to1.get(r, protein_letters_3to1.get(r.title(), "X")).upper()


def parse_structure(pdb_path, uid, fasta_seq):
    """
    Map PDB residues to FASTA 0-based indices using PDB residue numbering.
    AlphaFold canonical models normally use residue numbers 1..N.

    Returns dicts keyed by FASTA index plus residue objects.
    """
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

    residue_map = {}
    ca_coords = {}
    plddt = {}
    checked = matched = 0

    for res in chain.get_residues():
        if res.id[0] != " " or "CA" not in res:
            continue
        pos = int(res.id[1]) - 1
        if pos < 0 or pos >= len(fasta_seq):
            continue
        aa = three_to_one(res.get_resname())
        checked += 1
        if aa == fasta_seq[pos] or {aa, fasta_seq[pos]} <= {"I", "L"}:
            matched += 1
        residue_map[pos] = res
        ca_coords[pos] = np.asarray(res["CA"].get_coord(), dtype=float)
        plddt[pos] = float(res["CA"].get_bfactor())

    identity = matched / checked if checked else 0.0
    if checked < 10 or identity < 0.98:
        return {
            "mapping_valid": False,
            "mapping_identity": identity,
            "residue_map": residue_map,
            "ca_coords": ca_coords,
            "plddt": plddt,
            "rsa": {},
            "ss": {},
            "dssp_ok": False,
        }

    rsa = {}
    ss = {}
    dssp_ok = False
    try:
        dssp = DSSP(model, pdb_path, dssp=DSSP_BIN)
        for key in dssp.keys():
            chain_id, residue_id = key
            pos = int(residue_id[1]) - 1
            if pos < 0 or pos >= len(fasta_seq):
                continue
            entry = dssp[key]
            ss[pos] = entry[2] if entry[2] else "-"
            rsa[pos] = float(entry[3])
        dssp_ok = True
    except Exception:
        pass

    return {
        "mapping_valid": True,
        "mapping_identity": identity,
        "residue_map": residue_map,
        "ca_coords": ca_coords,
        "plddt": plddt,
        "rsa": rsa,
        "ss": ss,
        "dssp_ok": dssp_ok,
    }


def contiguous_exposed_run(rsas, threshold=0.20):
    best = cur = 0
    for r in rsas:
        if pd.notna(r) and r > threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def radius_of_gyration(coords):
    if len(coords) < 2:
        return np.nan
    x = np.asarray(coords, dtype=float)
    c = x.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((x - c) ** 2, axis=1))))


def aggregate(values, fn=np.mean):
    vals = [x for x in values if pd.notna(x)]
    return float(fn(vals)) if vals else np.nan


def shannon_entropy(seq):
    if not seq:
        return np.nan
    counts = pd.Series(list(seq)).value_counts(normalize=True).values
    return float(-(counts * np.log2(counts)).sum())


def main():
    global DSSP_BIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir",
                    default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdb-dir", default=None,
                    help="AlphaFold PDB cache (default: <base-dir>/alphafold_pdbs)")
    ap.add_argument("--dssp-bin", default=DSSP_BIN,
                    help="mkdssp executable name or full path")
    args = ap.parse_args()

    DSSP_BIN = args.dssp_bin
    base = args.base_dir
    out_dir = os.path.join(base, "model_features")
    pool_csv = os.path.join(out_dir, "features_biotin_pool.csv")
    parquet = os.path.join(base, "data", "pmsm_results.all.parquet")
    fasta_path = os.path.join(base, "data", "human_proteome.fasta")
    pdb_dir = args.pdb_dir or os.path.join(base, "alphafold_pdbs")
    out_csv = args.out or os.path.join(out_dir, "local_site_features_v3.csv")
    qc_csv = os.path.join(out_dir, "local_site_features_v3_mapping_qc.csv")
    os.makedirs(out_dir, exist_ok=True)

    # Master cohort = proteins that are actually part of the classification task.
    if not os.path.exists(pool_csv):
        raise FileNotFoundError(f"Modeling cohort not found: {pool_csv}")

    pool = pd.read_csv(pool_csv)
    if "uniprot_accession" not in pool.columns:
        raise RuntimeError("features_biotin_pool.csv lacks 'uniprot_accession'")

    target_uids = (
        pool["uniprot_accession"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.split("-").str[0]
        .str.upper()
        .unique()
    )
    target_uids = set(target_uids)

    print("=" * 72)
    print("LOCAL SITE FEATURES v3 — COHORT-CORRECTED")
    print("=" * 72)
    print(f"[COHORT] rows: {len(pool):,}")
    print(f"[COHORT] unique target proteins: {len(target_uids):,}")
    if "cell_line" in pool.columns:
        print(f"[COHORT] cell lines: {pool['cell_line'].nunique():,}")

    pep_col, prot_col = detect_columns(parquet)
    raw = pd.read_parquet(parquet, columns=[pep_col, prot_col])
    raw = raw[raw[pep_col].apply(has_biotin_mod)].copy()
    raw["accession"] = raw[prot_col].apply(extract_uniprot_acc)

    # Restrict feature generation to the intended modeling cohort.
    n_raw_biotin_rows = len(raw)
    raw = raw[raw["accession"].isin(target_uids)].copy()

    print(f"[MS] biotin-modified rows before cohort filter: {n_raw_biotin_rows:,}")
    print(f"[MS] biotin-modified rows within target cohort: {len(raw):,}")
    print(f"[MS] target proteins represented in parquet: {raw['accession'].nunique():,}")

    parsed = raw[pep_col].apply(parse_modified_peptide)
    raw["peptide_clean"] = [x[0] for x in parsed]
    raw["modified_local_indices"] = [x[1] for x in parsed]
    raw = raw.dropna(subset=["accession", "peptide_clean"])

    fasta = parse_fasta(fasta_path)
    results = []
    qc = []

    grouped = {uid: sub for uid, sub in raw.groupby("accession", sort=False)}

    # Iterate over the target cohort itself so QC denominators stay correct.
    for n, uid in enumerate(sorted(target_uids), 1):
        sub = grouped.get(uid)
        if sub is None or len(sub) == 0:
            qc.append({
                "uniprot_accession": uid,
                "status": "no_biotin_peptide_rows_in_parquet"
            })
            continue

        protein = fasta.get(uid)
        if not protein:
            qc.append({"uniprot_accession": uid, "status": "no_fasta"})
            continue

        # Deduplicate structural evidence by clean peptide + mod positions.
        obs_counter = (
            sub.groupby(["peptide_clean", sub["modified_local_indices"].astype(str)])
            .size()
            .to_dict()
        )

        unique_records = {}
        n_ambiguous = n_unmapped = 0

        for _, row in sub.iterrows():
            pep = row["peptide_clean"]
            mod_local = tuple(row["modified_local_indices"])
            matches, map_mode = all_occurrences(protein, pep)

            if len(matches) == 0:
                n_unmapped += 1
                continue
            if len(matches) != 1:
                n_ambiguous += 1
                continue

            start = matches[0]
            key = (pep, start, mod_local)
            unique_records[key] = {
                "peptide": pep,
                "start": start,
                "end": start + len(pep),
                "mod_local": mod_local,
                "map_mode": map_mode,
            }

        if not unique_records:
            qc.append({
                "uniprot_accession": uid,
                "status": "no_unique_mappings",
                "n_rows": len(sub),
                "n_ambiguous": n_ambiguous,
                "n_unmapped": n_unmapped,
            })
            continue

        pdb_path = os.path.join(pdb_dir, f"{uid}.pdb")
        if not os.path.exists(pdb_path):
            qc.append({"uniprot_accession": uid, "status": "missing_pdb"})
            continue

        st = parse_structure(pdb_path, uid, protein)
        if st is None:
            qc.append({"uniprot_accession": uid, "status": "pdb_parse_failed"})
            continue
        if not st["mapping_valid"]:
            qc.append({
                "uniprot_accession": uid,
                "status": "pdb_fasta_mapping_failed",
                "mapping_identity": st["mapping_identity"],
            })
            continue

        rsa = st["rsa"]
        plddt = st["plddt"]
        ss = st["ss"]
        ca = st["ca_coords"]
        resmap = st["residue_map"]

        site_rsa = []
        site_plddt = []
        site_exposed = []
        site_neighbor_count_8 = []
        site_neighbor_mean_rsa_8 = []
        site_neighbor_mean_plddt_8 = []
        site_neighbor_hydrophobic_frac_8 = []
        site_neighbor_charged_frac_8 = []
        site_neighbor_cys_frac_8 = []
        site_nz_neighbor_count_6 = []
        site_nz_neighbor_count_8 = []
        n_actual_biotin_sites = 0
        n_sites_with_structure = 0

        pep_mean_rsa = []
        pep_min_rsa = []
        pep_max_rsa = []
        pep_std_rsa = []
        pep_frac_exposed = []
        pep_mean_plddt = []
        pep_min_plddt = []
        pep_frac_helix = []
        pep_frac_coil = []
        pep_turn_frac = []
        pep_max_exposed_run = []
        pep_end_to_end = []
        pep_rg = []
        pep_mean_adjacent_ca = []
        pep_nterm_rsa = []
        pep_cterm_rsa = []

        seq_gravy = []
        seq_cys_frac = []
        seq_pg_frac = []
        seq_aromatic_frac = []
        seq_charged_frac = []
        seq_entropy = []
        seq_lengths = []

        for rec in unique_records.values():
            pep = rec["peptide"]
            start, end = rec["start"], rec["end"]
            positions = list(range(start, end))

            # sequence features: unique structural peptides only
            seq_gravy.append(np.mean([HYDROPATHY.get(a, 0.0) for a in pep]))
            seq_cys_frac.append(pep.count("C") / len(pep))
            seq_pg_frac.append((pep.count("P") + pep.count("G")) / len(pep))
            seq_aromatic_frac.append(sum(pep.count(a) for a in "FWY") / len(pep))
            seq_charged_frac.append(sum(a in CHARGED for a in pep) / len(pep))
            seq_entropy.append(shannon_entropy(pep))
            seq_lengths.append(len(pep))

            # Per-peptide structural summaries
            rvals = [rsa.get(p, np.nan) for p in positions]
            pvals = [plddt.get(p, np.nan) for p in positions]
            svals = [ss.get(p, "-") for p in positions]

            valid_r = [x for x in rvals if pd.notna(x)]
            valid_p = [x for x in pvals if pd.notna(x)]
            if valid_r:
                pep_mean_rsa.append(np.mean(valid_r))
                pep_min_rsa.append(np.min(valid_r))
                pep_max_rsa.append(np.max(valid_r))
                pep_std_rsa.append(np.std(valid_r))
                pep_frac_exposed.append(np.mean(np.asarray(valid_r) > 0.20))
                pep_max_exposed_run.append(contiguous_exposed_run(rvals))
                pep_nterm_rsa.append(rvals[0] if pd.notna(rvals[0]) else np.nan)
                pep_cterm_rsa.append(rvals[-1] if pd.notna(rvals[-1]) else np.nan)

            if valid_p:
                pep_mean_plddt.append(np.mean(valid_p))
                pep_min_plddt.append(np.min(valid_p))

            if svals:
                pep_frac_helix.append(np.mean([s in ("H", "G", "I") for s in svals]))
                pep_frac_coil.append(np.mean([s in ("-", " ", "T", "S", "B") for s in svals]))
                pep_turn_frac.append(np.mean([s in ("T", "S") for s in svals]))

            coords = [ca[p] for p in positions if p in ca]
            if len(coords) >= 2:
                coords = np.asarray(coords)
                pep_end_to_end.append(np.linalg.norm(coords[-1] - coords[0]))
                pep_rg.append(radius_of_gyration(coords))
                pep_mean_adjacent_ca.append(
                    np.mean(np.linalg.norm(coords[1:] - coords[:-1], axis=1))
                )

            # Actual modified residue(s), not every peptide residue.
            for local_idx in rec["mod_local"]:
                if local_idx < 0 or local_idx >= len(pep):
                    continue
                pos = start + local_idx
                n_actual_biotin_sites += 1

                if pos in rsa:
                    site_rsa.append(rsa[pos])
                    site_exposed.append(int(rsa[pos] > 0.20))
                if pos in plddt:
                    site_plddt.append(plddt[pos])

                if pos not in resmap or pos not in ca:
                    continue
                n_sites_with_structure += 1

                residue = resmap[pos]
                anchor = (
                    np.asarray(residue["NZ"].get_coord(), dtype=float)
                    if pep[local_idx] == "K" and "NZ" in residue
                    else ca[pos]
                )

                # CA-defined residue neighborhood around actual reactive site.
                neigh = []
                for q, qcoord in ca.items():
                    if q == pos:
                        continue
                    d = np.linalg.norm(qcoord - anchor)
                    if d <= 8.0:
                        neigh.append(q)

                site_neighbor_count_8.append(len(neigh))
                if neigh:
                    nrsa = [rsa[q] for q in neigh if q in rsa]
                    npld = [plddt[q] for q in neigh if q in plddt]
                    site_neighbor_mean_rsa_8.append(np.mean(nrsa) if nrsa else np.nan)
                    site_neighbor_mean_plddt_8.append(np.mean(npld) if npld else np.nan)
                    aas = [protein[q] for q in neigh if 0 <= q < len(protein)]
                    site_neighbor_hydrophobic_frac_8.append(
                        np.mean([a in HYDROPHOBIC for a in aas]) if aas else np.nan
                    )
                    site_neighbor_charged_frac_8.append(
                        np.mean([a in CHARGED for a in aas]) if aas else np.nan
                    )
                    site_neighbor_cys_frac_8.append(
                        np.mean([a == "C" for a in aas]) if aas else np.nan
                    )

                # Lys-NZ packing. For non-Lys / missing NZ, leave NaN.
                if pep[local_idx] == "K" and "NZ" in residue:
                    nz = np.asarray(residue["NZ"].get_coord(), dtype=float)
                    dists = [
                        np.linalg.norm(qcoord - nz)
                        for q, qcoord in ca.items() if q != pos
                    ]
                    site_nz_neighbor_count_6.append(sum(d <= 6.0 for d in dists))
                    site_nz_neighbor_count_8.append(sum(d <= 8.0 for d in dists))

        n_unique_peptides = len(unique_records)
        n_ms_observations = len(sub)

        result = {
            "uniprot_accession": uid,
            "n_ms_biotin_observations": n_ms_observations,
            "n_unique_biotin_peptides": n_unique_peptides,
            "n_unique_mapping_records": n_unique_peptides,
            "n_ambiguous_mapping_rows": n_ambiguous,
            "n_unmapped_rows": n_unmapped,
            "n_actual_biotin_sites": n_actual_biotin_sites,
            "n_biotin_sites_with_structure": n_sites_with_structure,
            "pdb_fasta_mapping_identity": st["mapping_identity"],
            "dssp_ok_v3": int(st["dssp_ok"]),

            # actual modified-site features
            "site_mean_rsa_v3": aggregate(site_rsa),
            "site_max_rsa_v3": aggregate(site_rsa, np.max),
            "site_mean_plddt_v3": aggregate(site_plddt),
            "site_frac_exposed_v3": aggregate(site_exposed),

            # unique peptide-region features
            "peptide_mean_rsa_v3": aggregate(pep_mean_rsa),
            "peptide_min_rsa_v3": aggregate(pep_min_rsa),
            "peptide_max_rsa_v3": aggregate(pep_max_rsa),
            "peptide_std_rsa_v3": aggregate(pep_std_rsa),
            "peptide_mean_plddt_v3": aggregate(pep_mean_plddt),
            "peptide_min_plddt_v3": aggregate(pep_min_plddt),
            "peptide_frac_exposed_v3": aggregate(pep_frac_exposed),
            "peptide_frac_helix_v3": aggregate(pep_frac_helix),
            "peptide_frac_coil_v3": aggregate(pep_frac_coil),
            "peptide_turn_fraction_v3": aggregate(pep_turn_frac),
            "peptide_max_contiguous_exposed_v3": aggregate(pep_max_exposed_run),
            "peptide_end_to_end_distance_v3": aggregate(pep_end_to_end),
            "peptide_rg_v3": aggregate(pep_rg),
            "peptide_mean_adjacent_ca_distance_v3": aggregate(pep_mean_adjacent_ca),
            "peptide_nterm_rsa_v3": aggregate(pep_nterm_rsa),
            "peptide_cterm_rsa_v3": aggregate(pep_cterm_rsa),

            # sequence over unique mapped peptides
            "peptide_gravy_v3": aggregate(seq_gravy),
            "peptide_cysteine_fraction_v3": aggregate(seq_cys_frac),
            "peptide_pg_fraction_v3": aggregate(seq_pg_frac),
            "peptide_aromatic_fraction_v3": aggregate(seq_aromatic_frac),
            "peptide_charged_fraction_v3": aggregate(seq_charged_frac),
            "peptide_sequence_entropy_v3": aggregate(seq_entropy),
            "peptide_length_v3": aggregate(seq_lengths),

            # 3D site neighborhood
            "site_neighbor_count_8A_v3": aggregate(site_neighbor_count_8),
            "site_neighbor_mean_rsa_8A_v3": aggregate(site_neighbor_mean_rsa_8),
            "site_neighbor_mean_plddt_8A_v3": aggregate(site_neighbor_mean_plddt_8),
            "site_neighbor_hydrophobic_fraction_8A_v3": aggregate(site_neighbor_hydrophobic_frac_8),
            "site_neighbor_charged_fraction_8A_v3": aggregate(site_neighbor_charged_frac_8),
            "site_neighbor_cysteine_fraction_8A_v3": aggregate(site_neighbor_cys_frac_8),
            "lys_nz_neighbor_count_6A_v3": aggregate(site_nz_neighbor_count_6),
            "lys_nz_neighbor_count_8A_v3": aggregate(site_nz_neighbor_count_8),
        }
        results.append(result)

        qc.append({
            "uniprot_accession": uid,
            "status": "ok",
            "n_ms_rows": n_ms_observations,
            "n_unique_peptides": n_unique_peptides,
            "n_ambiguous_rows": n_ambiguous,
            "n_unmapped_rows": n_unmapped,
            "n_actual_biotin_sites": n_actual_biotin_sites,
            "mapping_identity": st["mapping_identity"],
            "dssp_ok": st["dssp_ok"],
        })

        if n % 250 == 0:
            print(f"[{n}] processed proteins; output={len(results)}")

    out = pd.DataFrame(results)
    out.to_csv(out_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)

    qc_df = pd.DataFrame(qc)
    print("=" * 72)
    print(f"[SAVED] {out_csv} ({len(out):,} proteins)")
    print(f"[SAVED] {qc_csv}")
    print(f"[COHORT] intended target proteins: {len(target_uids):,}")
    print(f"[COVERAGE] v3 structural-feature proteins: {len(out):,}/{len(target_uids):,} "
          f"({100.0 * len(out) / max(len(target_uids), 1):.2f}%)")
    if len(qc_df):
        print("[QC] status counts:")
        print(qc_df["status"].value_counts(dropna=False).to_string())
    if len(out):
        print(f"[QC] median PDB/FASTA identity: {out['pdb_fasta_mapping_identity'].median():.4f}")
        print(f"[QC] proteins with >=1 actual mapped biotin site: {(out['n_actual_biotin_sites'] > 0).sum():,}")


if __name__ == "__main__":
    main()
