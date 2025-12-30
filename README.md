# Ensemble Learning for Arabic Sentiment Analysis

[![Paper](https://img.shields.io/badge/Paper-PeerJ%20CS-blue)](https://peerj.com/computer-science/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)

## Title
**Ensemble Learning for Arabic Sentiment Analysis: A Systematic Comparison Using Transformer-Based Models**

## Description
This repository contains the complete implementation code, datasets, and error annotations for our systematic evaluation of ensemble learning techniques for Arabic sentiment analysis. We compare four state-of-the-art transformer models (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT) and four ensemble strategies (Hard Voting, Soft Voting, Weighted Voting, Stacking) on the HARD dataset.

**Key Achievement**: Stacking ensemble achieves 96.40% F1-score with statistical significance (p<0.05) over individual models.

---

## Dataset Information

### HARD (Hotel Arabic Reviews Dataset)
- **Location**: `data/balanced-reviews.csv.zip`
- **Description**: Balanced subset of hotel reviews from Booking.com
- **Total Reviews**: 105,698 reviews
- **Classes**: Binary sentiment (Positive/Negative)
- **Language**: Modern Standard Arabic + dialectal variants (Gulf/Levantine)
- **Split Configuration**:
  - Training: 84,558 samples (80%)
  - Validation: 10,570 samples (10%)
  - Test: 10,570 samples (10%)
- **Original Source**: [HARD Dataset Repository](https://github.com/elnagara/HARD-Arabic-Dataset)

**Citation**:
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

### Error Annotations
- **Location**: `AnnotatedErrors/`
- **Files**:
  - `FP_annotations_camelbert_seed42.csv`: 260 False Positive samples with manual annotations
  - `FN_annotations_camelbert_seed42.csv`: 141 False Negative samples with manual annotations
- **Purpose**: Label noise analysis revealing 43.3% of errors stem from dataset quality issues
- **Annotation Protocol**: Two independent annotators with reconciliation

---

## Code Information

### Experiment 1: Main Ensemble Comparison
**File**: `code/experiment1_main.py`

**Purpose**: Systematic comparison of individual transformer models and ensemble methods

**Models Evaluated**:
- **AraBERT** (aubmindlab/bert-base-arabert): BERT trained on 77GB Arabic text with Farasa segmentation
- **MARBERT** (UBC-NLP/MARBERT): BERT trained on 1B Arabic tweets with dialectal focus
- **XLM-RoBERTa** (xlm-roberta-base): Multilingual model covering 100 languages
- **CAMeLBERT** (CAMeL-Lab/bert-base-arabic-camelbert-msa): Morphologically-aware Arabic BERT

**Ensemble Methods**:
1. Hard Voting: Majority vote from all models
2. Soft Voting: Average of probability distributions
3. Weighted Voting: Probability weighted by validation accuracy
4. Stacking: Logistic regression meta-classifier on base model outputs

**Experimental Setup**:
- 5 random seeds: 42, 123, 456, 789, 2024
- Learning rate: 2e-5 with linear warmup
- Batch size: 16 with gradient accumulation
- Max epochs: 3 with early stopping (patience=3)
- Optimizer: AdamW with weight decay 0.01

**Outputs**:
- Performance metrics (Accuracy, Precision, Recall, F1-Score) for all models
- Statistical significance tests (paired t-tests)
- Confusion matrices
- Model checkpoints

### Experiment 2: Lexicon Augmentation
**File**: `code/experiment2_lexicon.py`

**Purpose**: Evaluate whether sentiment lexicons improve high-baseline transformer models

**Lexicons Tested**:
1. LABR (General-purpose): 4,366 entries (Modern Standard Arabic + Egyptian dialect)
2. Custom Hotel (Domain-specific): 2,000 entries extracted from HARD training data

**Method**: Hybrid approach combining transformer predictions with lexicon-based sentiment scores
- Formula: `hybrid_prob = baseline_prob + α × normalized_lexicon_score`
- Adjustment weights tested: α = 0.05, 0.10, 0.15, 0.20, 0.25

**Key Finding**: No statistically significant improvement (p > 0.05), revealing performance ceiling at 96.42% F1

---

## Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on A100-SXM4-40GB)
- **RAM**: 32GB+ recommended
- **Storage**: ~10GB for models and data

### Software Dependencies
```
Python >= 3.8
PyTorch >= 1.10.0
transformers >= 4.18.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

### Installation
```bash
pip install torch>=1.10.0
pip install transformers>=4.18.0
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

---

## Usage Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment.git
cd Ens_Learning_Arabic_Sentiment
```

### Step 2: Extract Dataset
```bash
cd data
unzip balanced-reviews.csv.zip
cd ..
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Main Ensemble Experiments
```bash
cd code
python experiment1_main.py
```

**Expected Output**:
- Training logs for each model and seed
- Performance metrics saved to `results/experiment1_results.csv`
- Statistical significance tests saved to `results/significance_tests.csv`
- Model checkpoints saved to `checkpoints/`

**Expected Runtime**: ~29.45 hours (4 models × 5 seeds × 1.46 hours per training)

### Step 5: Run Lexicon Augmentation Experiments
```bash
python experiment2_lexicon.py
```

**Expected Output**:
- Lexicon augmentation results saved to `results/experiment2_lexicon_results.csv`
- Comparison plots saved to `results/lexicon_comparison.png`

**Expected Runtime**: ~2-3 hours

### Step 6: Analyze Error Annotations
```bash
python analyze_errors.py
```

**Input**: Loads annotations from `AnnotatedErrors/`  
**Output**: 
- Error analysis report: `results/error_analysis_report.txt`
- Label noise statistics
- Error type distributions

---

## Methodology

### Data Processing
1. **Text Normalization**:
   - Normalize Arabic characters (different forms of Alef, Ya)
   - Remove diacritical marks
   - Normalize whitespace
   - Lowercase Latin characters
2. **Tokenization**: Model-specific tokenizers (Farasa for AraBERT, SentencePiece for MARBERT, CAMeL Tools for CAMeLBERT)
3. **Splitting**: Stratified 80/10/10 split maintaining class balance

### Model Training
1. Load pre-trained transformer models from Hugging Face
2. Add classification head (linear layer)
3. Fine-tune on HARD training set
4. Monitor validation loss for early stopping
5. Save best checkpoint based on validation accuracy

### Ensemble Construction
1. **Training Phase**: Train 4 base models independently across 5 seeds
2. **Validation Phase**: Compute weights for Weighted Voting and train Stacking meta-classifier
3. **Testing Phase**: Generate predictions from all base models, apply ensemble strategy
4. **Evaluation**: Compute metrics and perform statistical significance testing

### Statistical Testing
- **Method**: Paired t-test comparing ensemble vs best individual model
- **Null Hypothesis**: No performance difference between methods
- **Significance Levels**: p<0.05 (*), p<0.01 (**), p<0.001 (***)
- **Samples**: 5 independent runs (random seeds)

---

## Results

### Individual Model Performance
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|-------------|---------------|------------|--------------|
| AraBERT | 95.89 ± 0.20 | 95.90 ± 0.20 | 95.89 ± 0.20 | 95.89 ± 0.20 |
| MARBERT | 96.05 ± 0.16 | 96.06 ± 0.16 | 96.05 ± 0.16 | 96.05 ± 0.16 |
| XLM-RoBERTa | 96.07 ± 0.21 | 96.09 ± 0.21 | 96.07 ± 0.21 | 96.07 ± 0.21 |
| **CAMeLBERT** | **96.24 ± 0.19** | **96.26 ± 0.18** | **96.24 ± 0.19** | **96.24 ± 0.19** |

### Ensemble Performance
| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|--------|-------------|---------------|------------|--------------|
| Hard Voting | 96.34 ± 0.16 | 96.34 ± 0.16 | 96.34 ± 0.16 | 96.34 ± 0.16 |
| Soft Voting | 96.39 ± 0.16 | 96.40 ± 0.16 | 96.39 ± 0.16 | 96.39 ± 0.16 |
| Weighted Voting | 96.39 ± 0.16 | 96.40 ± 0.16 | 96.39 ± 0.16 | 96.39 ± 0.16 |
| **Stacking** | **96.40 ± 0.14** | **96.40 ± 0.13** | **96.40 ± 0.14** | **96.40 ± 0.14** |

### Statistical Significance
| Comparison | t-statistic | p-value | Significance |
|------------|-------------|---------|--------------|
| Stacking vs AraBERT | 8.3848 | 0.0011 | *** |
| Stacking vs MARBERT | 9.0008 | 0.0008 | *** |
| Stacking vs XLM-RoBERTa | 4.7764 | 0.0088 | ** |
| Stacking vs CAMeLBERT | 3.5782 | 0.0232 | * |

---

## Repository Structure
```
Ens_Learning_Arabic_Sentiment/
├── data/
│   └── balanced-reviews.csv.zip          # HARD dataset (compressed)
├── code/
│   ├── experiment1_main.py               # Main ensemble experiments
│   ├── experiment2_lexicon.py            # Lexicon augmentation experiments
│   └── analyze_errors.py                 # Error analysis script
├── AnnotatedErrors/
│   ├── FP_annotations_camelbert_seed42.csv    # False Positive annotations
│   └── FN_annotations_camelbert_seed42.csv    # False Negative annotations
├── results/                              # Output directory (created on run)
├── checkpoints/                          # Model checkpoints (created on run)
├── requirements.txt                      # Python dependencies
├── LICENSE                              # MIT License
└── README.md                            # This file
```

---

## Citations

If you use this code or dataset in your research, please cite:

```bibtex
@article{alhawarat2025ensemble,
  title={Ensemble Learning for Arabic Sentiment Analysis: A Systematic Comparison Using Transformer-Based Models},
  author={Alhawarat, Mohammad},
  journal={PeerJ Computer Science},
  year={2025},
  publisher={PeerJ Inc.}
}
```

**Dataset Citation**:
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

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution Guidelines
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## Author
**Mohammad Alhawarat**  
Department of Data Science and AI  
Al-Ahliyya Amman University, Amman, Jordan  
Email: m.hawarat@ammanu.edu.jo

---

## Acknowledgments
- Google Colab for computational resources (NVIDIA A100 GPU)
- HARD dataset creators (Elnagar et al.)
- Developers of AraBERT, MARBERT, XLM-RoBERTa, and CAMeLBERT
- Hugging Face for the Transformers library
- Anonymous reviewers for constructive feedback

---

## Contact
For questions or issues:
- **GitHub Issues**: [Open an issue](https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment/issues)
- **Email**: m.hawarat@ammanu.edu.jo

---

⭐ **If you find this work useful, please star the repository!**
