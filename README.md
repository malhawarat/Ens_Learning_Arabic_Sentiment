# Ensemble Learning and Annotation-Noise Audit for Arabic Sentiment Analysis

Experiment notebooks, result files, and error annotations accompanying the manuscript:

> **Quantifying the Annotation Ceiling in Arabic Sentiment Benchmarks: A Reconciled Noise Audit of HARD with Cross-Resource Replication on ASTD and ArSAS**
> Mohammad O. Alhawarat, Ayman J. Alnsour, Khalil M. Abdelnaby, Mohammed A. F. Al-Husainy
> Al-Ahliyya Amman University (Jordan) and Al-Azhar University (Egypt)
> *Manuscript in preparation for submission; this README will be updated with the published reference.*

## What this repository contains

The study asks how much of the residual error of strong Arabic sentiment models reflects genuine model failure rather than incorrect ground-truth labels.

1. **Ensemble comparison on HARD.** Four transformer backbones (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT) and four combination rules (hard, soft, and weighted voting; stacking), five random seeds (42, 123, 456, 789, 2024).
2. **Reconciled error audit.** All 401 errors (260 false positives, 141 false negatives) of the best single model (CAMeLBERT, seed 42) were judged by two annotators as label noise or not; disagreements were reconciled by discussion.
3. **Cross-resource replication** on ASTD and ArSAS.
4. **Register-Aware Attention Fusion (RAAF)** — architecture and ablation (a negative accuracy result, reported with mechanism tests).

### Headline results

| HARD test set (10,570 reviews), mean ± SD over five seeds | F1 (%) |
|---|---|
| CAMeLBERT (best single model) | 96.24 ± 0.19 |
| Stacking ensemble (best ensemble) | 96.40 ± 0.14 (p = 0.0232 vs. CAMeLBERT) |

Annotation audit (CAMeLBERT, seed 42): 43.4% of the 401 errors are attributable to label noise (49.2% of false positives, 32.6% of false negatives). Cohen's kappa between the two annotators is 0.406 (false positives) and 0.420 (false negatives): *moderate* agreement. One of the two annotators is an author of the study (see Limitations in the paper).

The ensemble benefit and the backbone ranking do not transfer to the informal-register datasets (ASTD, ArSAS). See the paper for the full statistical treatment.

## Repository structure

```
Ens_Learning_Arabic_Sentiment/
├── code/                              # Google Colab notebooks (no standalone .py scripts)
│   ├── Arabic_Sentiment_Analysis_SingleRuns.ipynb   # HARD: 4 backbones x 4 ensembles, one seed per run
│   ├── Hybrid_Sentiment_MultiLexicon.ipynb          # lexicon-augmentation experiment
│   ├── astd_arsas_replication.ipynb                 # ASTD / ArSAS replication
│   ├── linguistic_marker_analysis.ipynb             # linguistic markers in true model errors
│   ├── raaf_architecture.ipynb                      # RAAF, register-balanced pooling
│   └── raaf_ablation_unbalanced.ipynb               # RAAF ablation, unbalanced pooling
├── data/balanced-reviews.zip          # HARD (balanced file used in the paper)
├── AnnotatedErrors/                   # annotations of the 401 CAMeLBERT (seed 42) errors
├── results/
│   ├── hard/                          # per-seed metrics, individual models and ensembles
│   ├── astd_arsas/                    # per-seed metrics and test-set probabilities
│   ├── raaf/                          # RAAF per-seed test metrics and detailed outputs
│   ├── linguistic_markers/            # per-review marker tags for FP / FN true errors
│   └── cross_model_errors/            # error lists for all four backbones (seed 42), see note
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the experiments

All experiments are Colab notebooks. Open a notebook in Google Colab, select a GPU runtime (a T4 is sufficient), and run the cells in order.

- Notebooks read and write under `/content/drive/MyDrive/EL4ASA` (Google Drive), so mount Drive and place the HARD CSV (unzip `data/balanced-reviews.zip`) there as the notebook instructs.
- `astd_arsas_replication.ipynb` downloads ASTD and ArSAS itself (ASTD from its GitHub source, ArSAS from the Hugging Face `arbml/ArSAS` dataset) and checkpoints per (dataset, model, seed), so free-tier sessions can be resumed.
- Each seed is run separately in `Arabic_Sentiment_Analysis_SingleRuns.ipynb`; run the final analysis cell after all five seeds have finished to produce the aggregate tables.
- `requirements.txt` lists the libraries for running a notebook outside Colab.
- Trained model checkpoints are large binary artifacts rather than data and are not committed; they are available from the corresponding author on reasonable request.

## Annotation files (`AnnotatedErrors/`)

Authoritative files: `camelbert_seed42_FP_Annotated.xlsx` (260 false positives) and `camelbert_seed42_FN_Annotated.xlsx` (141 false negatives).

| Column | Meaning |
|---|---|
| `index` | row index in the HARD test split |
| `review`, `true_label`, `predicted_label` | review text, gold label, CAMeLBERT prediction |
| `confidence`, `prob_negative`, `prob_positive` | model output |
| `Noise Narrator 1`, `Noise Narrator 2` | each annotator's independent judgment: 1 = the gold label is wrong (label noise), 0 = not noise (the two columns are the two annotators; "narrator" is the original column name) |
| `AND` | reconciled label noise: both annotators judged noise |
| `Disagreement` | exactly one annotator judged noise; counted as *ambiguous* |

Each sheet ends with a `COUNT` summary row and blank rows; exclude them before analysis (the row count in the sheet is larger than 260/141 for this reason).

The categories reported in the paper follow from these columns: **label noise** = `AND` = 1; **ambiguous** = `Disagreement` = 1; **model error** = both 0. This gives 128 / 72 / 60 for false positives and 46 / 41 / 54 for false negatives. Cohen's kappa on the two judgment columns reproduces 0.406 (FP) and 0.420 (FN).

> The `.csv` files in the same folder are earlier exports and are **not** consistent with the `.xlsx` files (different columns, trailing blank rows, missing judgments). Use the `.xlsx` files.

## Results folder notes

- `results/hard/ALL_RESULTS_COMBINED.csv` gathers every HARD metric by seed and model.
- `results/astd_arsas/*_summary.csv` give the mean ± SD per model for ASTD and ArSAS.
- `results/raaf/NOTE.md` explains why RAAF checkpoints are not included.
- `results/cross_model_errors/` contains error lists produced by an earlier run for all four backbones. Its CAMeLBERT lists (251 false positives, 140 false negatives, plus "high-confidence" subsets) are **not** the audited set: the audited 401 errors are in `AnnotatedErrors/`, and the paper does not use a confidence filter.

## Datasets

- **HARD** — Elnagar et al. (2018): https://github.com/elnagara/HARD-Arabic-Dataset (the balanced file used here is in `data/`). HARD is balanced by construction (50% positive, 50% negative).
- **ASTD** — Nabil et al. (2015): https://github.com/mahmoudnabil/ASTD
- **ArSAS** — Elmadany et al. (2018): https://huggingface.co/datasets/arbml/ArSAS

Please cite the original dataset papers when using them.

## Scope and limitations

- The error audit covers **one model and one seed** (CAMeLBERT, seed 42); the estimated noise ceiling is a property of HARD as annotated, not of every model.
- The audit is **two-annotator with moderate agreement**, and one annotator is an author; an independent replication is needed.
- Ensemble gains are small, significant only for stacking, and do not generalize to ASTD or ArSAS.

## Citation

Until the paper is published, please cite the repository:

```
Alhawarat, M. O., Alnsour, A. J., Abdelnaby, K. M., & Al-Husainy, M. A. F.
Quantifying the Annotation Ceiling in Arabic Sentiment Benchmarks:
A Reconciled Noise Audit of HARD with Cross-Resource Replication on ASTD and ArSAS.
Code and annotations: https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment
```

## Contact

Mohammad O. Alhawarat — m.hawarat@ammanu.edu.jo

## License

MIT (see `LICENSE`).
