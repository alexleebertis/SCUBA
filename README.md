# SCUBA — Surfaceome Classification Using Biotin Accessibility

**SCUBA** decides whether a protein detected in a cell-surface biotinylation
mass-spectrometry experiment is a **genuine cell-surface protein** or an
intracellular contaminant.

**The problem.** In surfaceome proteomics, intact cells are treated with a
membrane-impermeable NHS-biotin reagent that covalently tags solvent-exposed
lysines on proteins outside the cell. In practice the labeled peptide pool is
heavily contaminated: dead or leaky cells let the reagent in, and abundant
intracellular proteins dominate the signal. Simply taking every
biotin-positive protein as "surface" massively overestimates the surfaceome.

**The approach.** SCUBA scores each detected protein with a logistic
regression over **35 label-free features** in four families, all computed
without any prior surface annotation:

1. **Local structure around each labeled lysine** — solvent accessibility
   (RSA), secondary structure, local packing density, disorder, and peptide
   geometry, extracted from AlphaFold models at the exact modified residue.
2. **Membrane topology** — distance of each labeled site to the predicted
   transmembrane plane, derived from transmembrane-helix detection on the
   structure.
3. **Lysine chemical reactivity** — per-lysine pKa values from PROPKA 3,
   converted to the deprotonated (reactive) amine fraction at the labeling
   pH. NHS-biotin only reacts with deprotonated lysine ε-amines, so intrinsic
   reactivity is a direct physical covariate of labeling.
4. **Technical-replicate reproducibility** — how consistently the protein is
   detected across the technical replicates of each biological replicate.

> **Name note:** "SCUBA" is also the name of an unrelated scRNA-seq
> bifurcation-analysis tool (Marco et al. 2014, github.com/gcyuan/SCUBA).
> This project is a proteomics classifier and shares nothing with that tool.

## Performance

Evaluation uses **leave-cell-lines-out cross-validation**: the model is
trained on 10 cell lines and tested on 3 completely held-out lines, repeated
over all 286 ways to choose 3 test lines from 13. This measures whether the
model transfers to cell types it has never seen — the realistic deployment
scenario.

The reference operating point tunes the decision threshold on training-fold
out-of-fold scores to achieve **≥95% recall** (held-out performance is then
measured, not refit). Macro-averaged over all 286 splits:

| Metric | Value |
|---|---|
| Recall | **0.961** |
| Precision | 0.310 |
| False-positive reduction vs. accepting all biotin+ proteins | **0.506** |
| PR-AUC | 0.656 |
| ROC-AUC | 0.894 |

Trade-off at alternative operating points (same evaluation):

| Operating point | Precision | Recall | FP reduction |
|---|---|---|---|
| Top 10% by score | 0.732 | 0.402 | 0.967 |
| Top 20% by score | 0.612 | 0.661 | 0.905 |
| Top 30% by score | 0.514 | 0.823 | 0.822 |
| Top 50% by score | 0.354 | 0.936 | 0.604 |
| Training prevalence | 0.635 | 0.612 | 0.920 |
| **≥95% recall (reference)** | **0.310** | **0.961** | **0.506** |

The classification target is a UniProt-derived surface-annotation score
(≥4 on the SURFY scale). Signal-peptide predictions (e.g. SignalP) were
evaluated and **deliberately excluded**: UniProt surface annotations are
themselves partially derived from signal-peptide evidence, so including them
would leak the label into the features and inflate apparent performance.

## Pipeline

Run from the repository root. Each step writes into `model_features/`
(git-ignored). `--help` on any script lists all options.

```bash
# 0. Place the input data in data/ (see "Inputs" below)

# 1. Download AlphaFold models and compute global structural features
#    (pLDDT, disorder, secondary structure, solvent accessibility, glycosites)
python3 fetch_alphafold_structural_features.py

# 2. Compute local structural features around each biotin-labeled lysine
python3 build_local_site_features.py

# 3. Compute membrane-topology features (distance of labeled sites to the
#    predicted transmembrane plane)
python3 compute_membrane_distance.py

# 4. Assemble the modeling table: one row per (cell line, biological
#    replicate, protein), aggregating technical replicates
python3 build_biorep_datasets.py

# 5. Compute lysine pKa / reactivity features with PROPKA 3
python3 build_pka_features.py \
    --datasets model_features/biorep_datasets.csv \
    --pdb-dir alphafold_pdbs \
    --out model_features/pka_features.csv
#    (add --download --jobs 4 to fetch AlphaFold models on the fly)

# 6. Train and evaluate with leave-cell-lines-out cross-validation
#    (286 train/test splits; ~1–2 h)
python3 train_and_evaluate.py --with-oof-threshold

# 7. Independently audit the evaluation: re-derives every reported number
#    from raw outputs and cross-checks against the raw MS data (PASS/FAIL)
python3 verify_linesplit.py

# 8. (optional) Export per-protein scores and surface calls for every
#    protein in every cell line
python3 export_per_protein_predictions.py
```

`docs/` contains a heavily commented reading copy of the local-site feature
extractor (step 2) — documentation only, not part of the pipeline.

## Requirements

Python ≥ 3.10; see `requirements.txt`. External (non-pip) dependencies:

- **mkdssp** (DSSP binary) — solvent accessibility and secondary structure in
  steps 1–3. If it is not on `PATH`, pass `--dssp-bin /full/path/to/mkdssp`.
- **PROPKA 3** (`pip install propka`) — pKa features in step 5; the `propka3`
  executable must be on `PATH` or passed via `--propka`.
- AlphaFold models are downloaded from `alphafold.ebi.ac.uk`, or supply a
  local cache via `--pdb-dir`.

## Reproducibility / portability

All scripts are machine-agnostic: every path defaults to being relative to
the repository root, and every environment-specific setting is a CLI flag.
Nothing is hardcoded to a particular workstation.

- Steps 1–3 accept `--base-dir`, `--pdb-dir`, `--fasta`, `--parquet`,
  `--out`, and `--dssp-bin` overrides.
- Step 4 accepts `--cell-lines "LineA,LineB,..."` and
  `--required-tech-reps N` to reuse the pipeline on a different experiment
  (run names must carry a `repX_Y` token = biological replicate X,
  technical replicate Y).
- Step 5 accepts `--pdb-dir`, `--workdir`, `--propka`, `--ph`, and `--jobs`.
- The audit script (step 7) checks the published results of this study — its
  expected row counts are the ground truth being verified, not configuration.

## Inputs (not in this repo — proprietary / large)

Place these under `data/` before running:

- `data/pmsm_results.all.parquet` — raw mass-spectrometry evidence table
  (peptide-spectrum matches with run names, modified sequences, protein IDs,
  and error probabilities), one row per PSM across all runs
- `data/uniprot_human.fasta` — reviewed human proteome FASTA
- `data/human_proteome.fasta` — proteome FASTA used by the membrane-topology
  step
- `data/uniprot_human_with_surface_score.tsv` — UniProt annotations including
  the surface-annotation score used as the classification target

All generated artifacts (`model_features/`, `alphafold_pdbs/`,
`propka_work/`) are git-ignored.

## Repository layout

```
fetch_alphafold_structural_features.py  step 1 — AlphaFold download + global structural features
build_local_site_features.py            step 2 — local structure around labeled lysines
compute_membrane_distance.py            step 3 — membrane-topology features
build_biorep_datasets.py                step 4 — modeling-table assembly
build_pka_features.py                   step 5 — lysine pKa / reactivity features
train_and_evaluate.py                   step 6 — training + leave-cell-lines-out evaluation
verify_linesplit.py                     step 7 — independent audit of the evaluation
export_per_protein_predictions.py       step 8 — per-protein score export
docs/                                   annotated reading copy of the step-2 extractor
```

## License

TBD — confirm with the team before making the repository public.
