# SCUBA — Surfaceome Classification Using Biotin Accessibility

SCUBA predicts whether a protein detected in a **biotin-positive (NHS-biotin
surface-labeling) mass-spectrometry experiment** is a genuine **cell-surface
protein**, using only label-free inputs: AlphaFold structural features,
sequence/membrane topology features, lysine-reactivity (pKa) features, and
technical-replicate reproducibility counts.

> **Name note:** "SCUBA" is also the name of an unrelated scRNA-seq bifurcation
> analysis tool (Marco et al. 2014, github.com/gcyuan/SCUBA). This project is
> a proteomics / surfaceome classifier and shares nothing with that tool.

## Model (R3 — final)

The shipped model is **R3**, a logistic regression over **35 features**:

- **Top-20 structural/sequence whitelist** (from the v3 local-site feature set)
- **6 technical-replicate reproducibility features** (detection counts/fractions
  across the 3 technical replicates of each biological replicate)
- **9 lysine pKa features** (PROPKA 3 on AlphaFold models; labeling pH 7.2)

Evaluation is **line-split generalization**: train on 10 cell lines, test on 3
held-out lines, over all 286 combinations of 3 test lines from 13. Headline
numbers at the **oof95** operating point (threshold tuned to ≥95% recall on
inner out-of-fold training scores), macro-averaged over 286 combos:

| metric | R3 |
|---|---|
| recall | 0.961 |
| precision | 0.310 |
| false-positive reduction vs baseline | 0.506 |

The classification target is the SURFY-style UniProt surface label
(`surface_score >= 4`). Signal-peptide / SignalP-derived features were
evaluated and **deliberately excluded** (label-coupling conflation: UniProt
surface annotations partially derive from the same signal-peptide evidence).

## Pipeline

Run from the repo root. Each step writes into `model_features/` (git-ignored).

```bash
# 0. Place proprietary inputs in data/ (see "Inputs not in this repo" below)

# 1. Download AlphaFold models + global structural features (needs mkdssp)
python3 fetch_alphafold_structural_features.py

# 2. Local site features around UniMod:293-labeled lysines (v3, cohort-fixed)
python3 build_local_site_features_v3.py

# 3. Membrane-distance / topology features
python3 compute_membrane_distance.py

# 4. Assemble (cell_line, bio_rep, protein) datasets
python3 build_biorep_datasets.py

# 5. Lysine pKa features (PROPKA 3); takes the accession list from step 4
python3 build_pka_features.py \
    --datasets model_features/biorep_datasets.csv \
    --pdb-dir alphafold_pdbs \
    --out model_features/pka_features.csv
#    (or --download --jobs 4 to fetch AlphaFold models on the fly)

# 6. Train + line-split generalization evaluation (286 combos; ~1-2 h)
python3 line_split_logreg_top20_biorep_pka.py --with-oof-threshold

# 7. Independent audit of the split results (PASS/FAIL checks)
python3 verify_linesplit.py

# 8. (optional) Per-protein R3 scores for every line, all 286 combos
python3 export_r3_per_protein.py
```

`docs/build_local_site_features_v3_annotated.py` is a heavily commented
reading copy of step 2 — documentation only, not part of the pipeline.

## Requirements

Python ≥ 3.10; see `requirements.txt`. External (non-pip) dependencies:

- **mkdssp** (DSSP binary) — required by step 1 for RSA/secondary structure
- **PROPKA 3** (`pip install propka`) — required by step 4
- AlphaFold PDBs are downloaded from `alphafold.ebi.ac.uk` (or supply a local
  `--pdb-dir` cache)

## Inputs not in this repo (proprietary / large)

Place these under `data/` before running:

- `data/pmsm_results.all.parquet` — raw MaxQuant evidence tables (all runs)
- `data/uniprot_human.fasta` — reviewed human proteome FASTA
- `data/human_proteome.fasta` — proteome FASTA used by membrane-distance step
- `data/uniprot_human_with_surface_score.tsv` — UniProt annotations incl.
  the SURFY surface score used for the classification target

All generated artifacts (`model_features/`, `alphafold_pdbs/`,
`propka_work/`) are git-ignored.

## Layout

```
fetch_alphafold_structural_features.py step 1
build_local_site_features_v3.py        step 2  (canonical, cohort-fixed)
compute_membrane_distance.py           step 3
build_biorep_datasets.py               step 4
build_pka_features.py                  step 5
line_split_logreg_top20_biorep_pka.py  step 6  (R1/R2/R3 trainer + evaluator)
verify_linesplit.py                    step 7  (independent audit)
export_r3_per_protein.py               step 8  (per-protein R3 export)
docs/build_local_site_features_v3_annotated.py   annotated reading copy
```

## License

TBD — confirm with the team before making the repository public.
