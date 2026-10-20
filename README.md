# Ensemble Learning and Annotation-Noise Audit for Arabic Sentiment Analysis

Code, annotations, and experiment notebooks accompanying the manuscript:

> **Quantifying the Annotation Ceiling in Arabic Sentiment Benchmarks: A Reconciled Noise Audit of HARD with Cross-Resource Replication on ASTD and ArSAS**
> Mohammad O. Alhawarat, Ayman J. Alnsour, Khalil M. Abdelnaby, Mohammed A. F. Al-Husainy
> Al-Ahliyya Amman University (Jordan) and Al-Azhar University (Egypt)
> *Manuscript in preparation for submission; this README will be updated with the published reference.*

## What this repository contains

The study asks how much of the residual error of strong Arabic sentiment models reflects genuine model failure rather than incorrect ground-truth labels. It has four parts:

1. **Ensemble comparison on HARD.** Four transformer backbones (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT) and four combination rules (hard voting, soft voting, weighted voting, stacking), five random seeds.
2. **Reconciled error audit.** All 401 errors (260 false positives, 141 false negatives) of the best single model (CAMeLBERT, seed 42) were independently annotated by two annotators as label noise, genuine model error, or ambiguous, then reconciled. Item-level agreement is reported as Cohen's kappa.
3. **Cross-resource replication** on ASTD and ArSAS.
4. **Register-Aware Attention Fusion (RAAF)** — an architecture and ablation study (reported as a negative accuracy result with mechanism analysis).

### Headline results (HARD test set, 10,570 reviews, mean over five seeds)

| Model | F1 (%) |
|---|---|
| CAMeLBERT (best single model) | 96.24 ± 0.19 |
| Stacking ensemble (best ensemble) | 96.40 ± 0.14 (p = 0.0232 vs. CAMeLBERT) |

Annotation audit (CAMeLBERT, seed 42): 43.4% of the 401 errors are attributable to label noise (49.2% of false positives, 32.6% of false negatives). Cohen's kappa is 0.406 (false positives) and 0.420 (false negatives), i.e. *moderate* agreement; one of the two annotators is an author of the study (see Limitations in the paper).

Neither the ensemble benefit nor the backbone ranking transfers to the informal-register datasets (ASTD, ArSAS). See the paper for the full statistical treatment.

## Repository structure

```
Ens_Learning_Arabic_Sentiment/
├── data/                  # HARD dataset (zipped)
├── code/
│   ├── experiment1_main.py                   # main ensemble experiments (HARD)
│   ├── experiment2_lexicon.py                # lexicon augmentation experiment
│   ├── analyze_errors.py                     # error-annotation analysis
│   ├── Arabic_Sentiment_Analysis_SingleRuns.ipynb
│   ├── Hybrid_Sentiment_MultiLexicon.ipynb
│   ├── astd_arsas_replication.ipynb          # ASTD / ArSAS replication
│   ├── linguistic_marker_analysis.ipynb      # linguistic markers in error set
│   ├── raaf_architecture.ipynb               # RAAF (register-balanced pooling)
│   └── raaf_ablation_unbalanced.ipynb        # RAAF ablation (unbalanced pooling)
├── AnnotatedErrors/       # reconciled annotations of the 401 errors (XLSX)
├── results/               # created on run
├── checkpoints/           # created on run
├── requirements.txt
├── LICENSE
└── README.md
```

> **Maintainer note:** add the four new notebooks and the final annotation files to the folders above before publishing this README, and delete any line in this tree that does not match what is actually committed.

## Annotation files

`AnnotatedErrors/` holds the annotation spreadsheets for the 401 CAMeLBERT (seed 42) errors: one file for false positives and one for false negatives, each with both annotators' independent judgments and the reconciled category (label noise / model error / ambiguous). All files are `.xlsx`.

Agreement can be recomputed from these files: Cohen's kappa on the binary noise / not-noise judgment, per error type, should reproduce the values above.

## Datasets

The datasets are not redistributed except where their licenses allow; obtain them from the original sources:

- **HARD** — Elnagar et al. (2018): https://github.com/elnagara/HARD-Arabic-Dataset
- **ASTD** — Nabil et al. (2015): https://github.com/mahmoudnabil/ASTD
- **ArSAS** — Elmadany et al. (2018): https://huggingface.co/datasets/arbml/ArSAS

HARD is balanced by construction (50% positive / 50% negative) in the file used here (`balanced-reviews.csv`).

## Setup and reproduction

```bash
pip install -r requirements.txt
unzip data/*.zip -d data/
python code/experiment1_main.py        # main ensemble experiments
python code/experiment2_lexicon.py     # lexicon augmentation
python code/analyze_errors.py          # error-annotation analysis
```

The Colab notebooks in `code/` reproduce the replication and RAAF experiments. The main experiment is expensive (about 29 hours on an A100 GPU for all backbones and seeds); trained checkpoints are large binary files and are not committed — they are available from the corresponding author on reasonable request.

## Notes on scope and limitations

- The error audit covers **one model and one seed** (CAMeLBERT, seed 42); the noise ceiling is therefore a property of HARD as annotated, not of every model.
- The audit is **two-annotator with moderate agreement**, and one annotator is an author; a fully independent replication is needed.
- Ensemble gains are small and are tested on HARD; they do not generalize to ASTD or ArSAS.

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
