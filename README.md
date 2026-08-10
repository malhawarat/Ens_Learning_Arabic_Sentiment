# Beyond Accuracy: A Noise-Ceiling Account of Diminishing Ensemble Returns in Arabic Sentiment Analysis

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)

## Authors

Mohammad O. Alhawarat¹\*, Ayman J. Alnsour¹, Khalil M. Abdelnaby¹˒², Mohammed A. F. Al-Husainy¹

¹ Faculty of Information Technology, Al-Ahliyya Amman University, Amman 19328, Jordan
² Systems and Computers Engineering Department, Faculty of Engineering, Al-Azhar University, Nasr City, Cairo 11765, Egypt

\* Corresponding author: m.hawarat@ammanu.edu.jo

**Manuscript status:** submitted to *Language Resources and Evaluation* (Springer). This repository will be updated with the DOI upon acceptance; until then, no other simultaneous submission of this work exists.

## Description

A systematic empirical and theoretical study of ensemble learning for Arabic sentiment analysis, comprising:

1. A five-seed, statistically validated comparison of four ensemble-combination strategies (Hard Voting, Soft Voting, Weighted Voting, Stacking) over four transformer backbones (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT) on the Hotel Arabic Reviews Dataset (HARD).
2. A reconciled, two-annotator quantification of label noise among all 401 high-confidence false positives and false negatives, with item-level inter-annotator agreement reported via Cohen's Kappa.
3. A cross-dataset replication of the full protocol on ASTD and ArSAS (Twitter, informal register), testing whether the HARD-based ensemble benefit and backbone ranking generalize.
4. A theoretical account (Section 5 of the manuscript) deriving a provable noise ceiling on measurable accuracy, a disagreement bound on ensemble headroom, and a non-identifiability result for gated combination over redundant frozen experts -- explaining, from one cause, why ensemble gains are small on this benchmark.
5. Register-Aware Attention Fusion (RAAF, Appendix A of the manuscript): a theoretically-motivated architecture testing whether register-aware combination recovers what register-blind ensembling lacks. It does not improve accuracy; the mechanism analysis is reported as a negative but informative result.

## Dataset Information

### HARD (Hotel Arabic Reviews Dataset)
- **Location:** `data/balanced-reviews.zip`
- **Description:** Balanced subset of hotel reviews from Booking.com. The source file is exactly class-balanced: 52,849 positive and 52,849 negative reviews (50.0%/50.0%).
- **Total reviews:** 105,698
- **Split:** 84,558 train (80%) / 10,570 validation (10%) / 10,570 test (10%), stratified, reshuffled independently per seed
- **Original source:** [HARD Dataset Repository](https://github.com/elnagara/HARD-Arabic-Dataset)
- **Citation:** Elnagar, Khalifa & Einea (2018), see [Citations](#citations)

### ASTD and ArSAS (cross-dataset replication)
- **ASTD:** Nabil et al. (2015), Arabic Twitter sentiment; binarized to 2,419 positive/negative tweets after removing objective and mixed-sentiment classes. Source: [ASTD repository](https://github.com/mahmoudnabil/ASTD)
- **ArSAS:** Elmadany, Mubarak & Magdy (2018), multi-domain Arabic social-media sentiment; binarized to 11,784 positive/negative tweets (37.3% positive) after removing neutral and mixed classes. Source: [ArSAS on Hugging Face](https://huggingface.co/datasets/arbml/ArSAS)
- Neither dataset is redistributed in this repository; both are obtainable from their original sources above. The preparation/binarization code is in `code/astd_arsas_replication.ipynb`.

### Error Annotations
- **Location:** `AnnotatedErrors/`
- **Scope:** All 401 high-confidence errors (260 false positives, 141 false negatives) from CAMeLBERT (seed 42) on the HARD test set -- the full error set, not a confidence-filtered subset.
- **Files:** per-item annotator judgements and reconciled categories (noise / model error / ambiguous) for both FP and FN sets, in both `.csv` and `.xlsx`.
- **Protocol:** two annotators (the corresponding author and a colleague at the same institution) independently categorize each item; disagreements resolved by discussion. Item-level inter-annotator agreement: Cohen's kappa = 0.406 (false positives), 0.420 (false negatives), 0.425 (pooled, n=401) -- moderate agreement, reported as a measured quantity rather than assumed.
- **Headline result:** 43.4% of the 401 errors are confirmed label noise (49.2% of false positives, 32.6% of false negatives), not model failure. A further 28.2% are irreducibly ambiguous.

### Linguistic-Marker Tags
- **Location:** `results/linguistic_markers/`
- Per-review structural/linguistic tags (positive-title/negative-detail structure, terseness, negation density, code-switching, dialectal markers) for the confirmed-model-error subset (n=60 FP, n=54 FN), produced by `code/linguistic_marker_analysis.ipynb`.

## Code Information

| File | Purpose |
|---|---|
| `code/Arabic_Sentiment_Analysis_SingleRuns.ipynb` | Main ensemble experiments: 4 backbones x 5 seeds x 4 combination rules on HARD |
| `code/Hybrid_Sentiment_MultiLexicon.ipynb` | Lexicon augmentation experiments (LABR, Custom Hotel lexicons) |
| `code/linguistic_marker_analysis.ipynb` | Automated structural/linguistic tagging of confirmed model errors |
| `code/astd_arsas_replication.ipynb` | Cross-dataset replication pipeline: data preparation, training, and evaluation on ASTD and ArSAS |
| `code/raaf_architecture.ipynb` | RAAF architecture, register-balanced pooling training run, and mechanism tests (P1-P3) |
| `code/raaf_ablation_unbalanced.ipynb` | RAAF pooling ablation (unbalanced pooling), isolating the training-data confound |

All code is provided as the original Jupyter/Colab notebooks used to produce the results in this
repository, rather than as separately maintained standalone scripts, so there is a single
authoritative source for each result with no risk of the two drifting out of sync.

**Models evaluated:**
- **AraBERT** (`aubmindlab/bert-base-arabert`): trained on 77GB Arabic text with Farasa segmentation, ~135M parameters
- **MARBERT** (`UBC-NLP/MARBERT`): trained on 1B Arabic tweets, WordPiece tokenization, ~163M parameters
- **XLM-RoBERTa** (`xlm-roberta-base`): multilingual, 100 languages, ~270M parameters
- **CAMeLBERT** (`CAMeL-Lab/bert-base-arabic-camelbert-msa`): morphologically-aware Arabic BERT, ~110M parameters

**Ensemble methods:** Hard Voting, Soft Voting, Weighted Voting (by validation accuracy), Stacking (logistic-regression meta-classifier on validation-split base-model probabilities, applied unmodified to the test split -- see the manuscript's Meta-feature provenance and leakage control paragraph).

**Training configuration:** learning rate 2e-5 with 10% linear warmup, batch size 16, up to 3 epochs with early stopping (patience 3 on validation loss), AdamW (weight decay 0.01), max sequence length 128. Full five-seed run: ~29.45 hours on an NVIDIA A100-40GB (Table 2 of the manuscript).

## Results

Full results, all statistical tests, and every table/figure in the manuscript are reproducible from the files in `results/`. Headline numbers:

### Individual Model Performance (HARD, mean ± std across 5 seeds)
| Model | F1-Score (%) |
|---|---|
| AraBERT | 95.89 ± 0.20 |
| MARBERT | 96.05 ± 0.16 |
| XLM-RoBERTa | 96.07 ± 0.21 |
| **CAMeLBERT** | **96.24 ± 0.19** |

### Ensemble Performance (HARD)
| Method | F1-Score (%) | vs.\ best individual |
|---|---|---|
| Hard Voting | 96.34 ± 0.16 | -- |
| Soft Voting | 96.39 ± 0.16 | -- |
| Weighted Voting | 96.39 ± 0.16 | -- |
| **Stacking** | **96.40 ± 0.14** | +0.16, p=0.0232 (\*) |

### Cross-Dataset Replication
On ASTD and ArSAS, the best individual backbone becomes domain-dependent (**MARBERT**, not CAMeLBERT, leads on both Twitter-register datasets), and no ensemble method achieves a statistically significant gain over the best individual model on either dataset (all p > 0.05) -- in contrast to HARD's significant, if modest, gain. See the manuscript's Section on Generalizability Beyond HARD for the full cross-dataset table and discussion.

### Theoretical Ceiling (manuscript Section 5)
Instantiating the random classification noise model with the paper's own annotation gives a provable ceiling on measurable HARD accuracy of at most 98.35%, leaving at most 2.09-2.11 percentage points of headroom for all future modeling work on this benchmark, of which the best measured ensemble gain (+0.16) claims roughly 8%.

## Repository Structure

```
Ens_Learning_Arabic_Sentiment/
├── data/
│   └── balanced-reviews.zip                       # HARD dataset (compressed; ASTD/ArSAS not redistributed, see above)
├── code/
│   ├── Arabic_Sentiment_Analysis_SingleRuns.ipynb  # Main ensemble experiments (Experiment 1)
│   ├── Hybrid_Sentiment_MultiLexicon.ipynb         # Lexicon augmentation (Experiment 2)
│   ├── linguistic_marker_analysis.ipynb            # Structural error-tagging
│   ├── astd_arsas_replication.ipynb                # Cross-dataset replication
│   ├── raaf_architecture.ipynb                     # RAAF (register-balanced pooling)
│   └── raaf_ablation_unbalanced.ipynb              # RAAF pooling ablation
├── AnnotatedErrors/
│   ├── camelbert_seed42_FP_Annotated.{csv,xlsx}    # 260 false positives, full annotation
│   └── camelbert_seed42_FN_Annotated.{csv,xlsx}    # 141 false negatives, full annotation
├── results/
│   ├── hard/                                       # Per-seed individual-model and ensemble results (JSON), all 4 backbones x 5 seeds
│   ├── cross_model_errors/                         # Supplementary per-model error counts (all 4 backbones, seed 42)
│   ├── linguistic_markers/                         # Per-review structural tags (fp/fn)
│   ├── astd_arsas/                                 # ASTD/ArSAS: per-model results + probabilities (4 backbones x 5 seeds x 2 datasets) and per-seed ensemble results
│   └── raaf/                                       # RAAF unbalanced-pooling ablation: per-dataset test results, mechanism-test detail (P1/P2/P3), and centroid-comparison outputs.
│                                                    # Trained checkpoints are not redistributed here (see results/raaf/NOTE.md); the register-balanced run's
│                                                    # results are preserved as executed cell outputs in code/raaf_architecture.ipynb.
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage Instructions

```bash
# Clone repository
git clone https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment.git
cd Ens_Learning_Arabic_Sentiment

# Install dependencies
pip install -r requirements.txt

# Extract HARD dataset
cd data && unzip balanced-reviews.zip && cd ..

```

All experiments are provided as the original Google Colab notebooks (with Google Drive mount
cells) used to produce every result in this repository -- there are no separately maintained
`.py` scripts, so there is exactly one version of each experiment's code, with no risk of a
script and a notebook silently drifting apart. Open a notebook in `code/` in Colab (or locally
with a GPU) and run cells sequentially; see in-notebook markdown for setup instructions,
including the resumable/checkpointed design used because these experiments were run on
free-tier Colab T4 GPUs. All results referenced by the manuscript are already provided in
`results/`, so re-running is only necessary to reproduce, not to obtain, the reported numbers.

## Requirements

```
Python >= 3.8
PyTorch >= 1.10.0
transformers >= 4.18.0
scikit-learn >= 1.0.0
pandas, numpy, scipy, matplotlib, seaborn, openpyxl
```
Full pinned dependencies in `requirements.txt`.

## Citations

**HARD dataset:**
```bibtex
@incollection{elnagar2018hotel,
  title={Hotel Arabic-Reviews Dataset Construction for Sentiment Analysis Applications},
  author={Elnagar, Ashraf and Khalifa, Yasmin S and Einea, Anas},
  booktitle={Intelligent Natural Language Processing: Trends and Applications},
  pages={35--52},
  publisher={Springer},
  year={2018}
}
```

**ASTD dataset:**
```bibtex
@inproceedings{nabil2015astd,
  title={ASTD: Arabic Sentiment Tweets Dataset},
  author={Nabil, Mahmoud and Aly, Mohamed and Atiya, Amir F},
  booktitle={Proceedings of EMNLP},
  pages={2515--2519},
  year={2015}
}
```

**ArSAS dataset:**
```bibtex
@inproceedings{elmadany2018arsas,
  title={ArSAS: An Arabic Speech-Act and Sentiment Corpus of Tweets},
  author={Elmadany, AbdelRahim A and Mubarak, Hamdy and Magdy, Walid},
  booktitle={Proc. 3rd Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT3), LREC},
  year={2018}
}
```

**This work:** citation details will be added upon publication. This manuscript is currently under review; please cite the venue-of-record once available rather than this repository.

## License
MIT License -- see [LICENSE](LICENSE).

## Contribution Guidelines
Contributions are welcome via pull request:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes
4. Open a pull request

## Acknowledgments
We thank the creators of the HARD, ASTD, and ArSAS datasets, the developers of the AraBERT, MARBERT, XLM-RoBERTa, and CAMeLBERT models, and the Hugging Face Transformers library. Computational resources were provided by Google Colab.

## Contact
- **GitHub Issues:** [Open an issue](https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment/issues)
- **Email:** m.hawarat@ammanu.edu.jo
