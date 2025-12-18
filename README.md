# Ensemble Learning for Arabic Sentiment Analysis

[![Paper](https://img.shields.io/badge/Paper-PeerJ%20CS-blue)](https://peerj.com/computer-science/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)

Official implementation of **"Ensemble Learning for Arabic Sentiment Analysis: A Systematic Comparison Using Transformer-Based Models"** published in PeerJ Computer Science.

## 📝 Abstract

This repository contains the code and data for our systematic evaluation of ensemble learning techniques combining four transformer architectures (AraBERT, MARBERT, XLM-RoBERTa, and CAMeLBERT) for Arabic sentiment analysis. Our experiments demonstrate that stacking ensemble achieves **96.40% F1-score** on the HARD dataset, with statistical significance over individual models.

## 🎯 Key Findings

- **Stacking ensemble** outperforms best individual model (CAMeLBERT) by +0.16% with statistical significance (p<0.05)
- **Lexicon augmentation** shows no significant improvement at high baselines (>96% F1), revealing a performance ceiling
- **43.3% of errors** stem from label noise rather than model limitations
- All ensemble methods demonstrate reduced variance and improved stability across random seeds

## 📂 Repository Structure

```
.
├── data/
│   └── balanced-reviews.csv.zip          # HARD dataset (compressed)
├── code/
│   ├── experiment1_main.py               # Main ensemble experiments
│   └── experiment2_lexicon.py            # Lexicon augmentation experiments
├── AnnotatedErrors/
│   ├── FP_annotations_camelbert_seed42.csv    # False Positive annotations
│   └── FN_annotations_camelbert_seed42.csv    # False Negative annotations
└── README.md
```

## 🗂️ Dataset

The `data/` folder contains the balanced HARD (Hotel Arabic Reviews Dataset) used in our experiments:

- **File**: `balanced-reviews.csv.zip`
- **Source**: Hotel Arabic Reviews Dataset (HARD)
- **Total Reviews**: 105,698 hotel reviews from Booking.com
- **Classes**: Binary sentiment (Positive/Negative)
- **Split**: 80% train / 10% validation / 10% test
- **Languages**: Modern Standard Arabic + dialectal variants (Gulf/Levantine)

**Original Dataset Citation**:
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

## 💻 Code

### Main Experiments (`code/`)

#### 1. **experiment1_main.py**
Main ensemble learning experiments comparing:
- **Individual Models**: AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT
- **Ensemble Methods**: Hard Voting, Soft Voting, Weighted Voting, Stacking
- **Evaluation**: 5 random seeds (42, 123, 456, 789, 2024) with statistical significance testing

**Key Results**:
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------|-------------|---------------|------------|--------------|
| AraBERT | 95.89 ± 0.20 | 95.90 ± 0.20 | 95.89 ± 0.20 | 95.89 ± 0.20 |
| MARBERT | 96.05 ± 0.16 | 96.06 ± 0.16 | 96.05 ± 0.16 | 96.05 ± 0.16 |
| XLM-RoBERTa | 96.07 ± 0.21 | 96.09 ± 0.21 | 96.07 ± 0.21 | 96.07 ± 0.21 |
| CAMeLBERT | 96.24 ± 0.19 | 96.26 ± 0.18 | 96.24 ± 0.19 | 96.24 ± 0.19 |
| **Stacking** | **96.40 ± 0.14** | **96.40 ± 0.13** | **96.40 ± 0.14** | **96.40 ± 0.14** |

#### 2. **experiment2_lexicon.py**
Lexicon augmentation experiments testing:
- **LABR Lexicon**: General-purpose (4,366 entries)
- **Custom Hotel Lexicon**: Domain-specific (2,000 entries)
- **Adjustment Weights**: α = 0.05, 0.10, 0.15, 0.20, 0.25

**Key Finding**: No statistically significant improvement (p > 0.05), suggesting transformer models have internalized sentiment patterns at high baselines.

## 📊 Error Analysis

The `AnnotatedErrors/` folder contains manual annotations of high-confidence errors from CAMeLBERT (seed 42):

### Files:
- **`FP_annotations_camelbert_seed42.csv`**: False Positive annotations (260 samples)
  - 49.2% identified as label noise
  - 27.7% ambiguous cases
  
- **`FN_annotations_camelbert_seed42.csv`**: False Negative annotations (141 samples)
  - 32.6% identified as label noise
  - 29.1% ambiguous cases

### Annotation Protocol:
- Two independent annotators
- Final decision through reconciliation
- Categories: Noise (mislabeled), Correct (true error), Not Sure (ambiguous)

### Key Insight:
Approximately **43.3% of classification errors** stem from dataset label noise rather than model limitations, suggesting the effective performance ceiling on clean data is ~97.5-98%.

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
PyTorch 1.10+
Transformers 4.18+
scikit-learn
pandas
numpy
```

### Installation
```bash
# Clone the repository
git clone https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment.git
cd Ens_Learning_Arabic_Sentiment

# Install dependencies
pip install torch transformers scikit-learn pandas numpy

# Extract dataset
cd data
unzip balanced-reviews.csv.zip
cd ..
```

### Running Experiments

#### Main Ensemble Experiments
```bash
cd code
python experiment1_main.py
```

#### Lexicon Augmentation Experiments
```bash
python experiment2_lexicon.py
```

### Expected Runtime
- Individual model training: ~1.46 hours per seed (NVIDIA A100 GPU)
- Total training time (4 models × 5 seeds): ~29.45 hours
- Ensemble inference: 4× single model inference time

## 📈 Results

### Statistical Significance (Paired t-tests)
| Comparison | t-statistic | p-value | Significance |
|------------|-------------|---------|--------------|
| Stacking vs. AraBERT | 8.3848 | 0.0011 | *** |
| Stacking vs. MARBERT | 9.0008 | 0.0008 | *** |
| Stacking vs. XLM-RoBERTa | 4.7764 | 0.0088 | ** |
| Stacking vs. CAMeLBERT | 3.5782 | 0.0232 | * |

*p < 0.05, **p < 0.01, ***p < 0.001

### Model Details
- **AraBERT**: aubmindlab/bert-base-arabert
- **MARBERT**: UBC-NLP/MARBERT
- **XLM-RoBERTa**: xlm-roberta-base
- **CAMeLBERT**: CAMeL-Lab/bert-base-arabic-camelbert-msa

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{alhawarat2025ensemble,
  title={Ensemble Learning for Arabic Sentiment Analysis: A Systematic Comparison Using Transformer-Based Models},
  author={Alhawarat, Mohammad},
  journal={PeerJ Computer Science},
  year={2025},
  publisher={PeerJ Inc.}
}
```

## 👤 Author

**Mohammad Alhawarat**  
Department of Data Science and AI  
Al-Ahliyya Amman University, Amman, Jordan  
Email: m.hawarat@ammanu.edu.jo

## 🙏 Acknowledgments

- Google Colab for computational resources
- HARD dataset creators
- Developers of AraBERT, MARBERT, XLM-RoBERTa, and CAMeLBERT
- Anonymous reviewers for constructive feedback

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please:
- Open an issue in this repository
- Email: m.hawarat@ammanu.edu.jo

## 🔗 Links

- **Paper**: [PeerJ Computer Science](https://peerj.com/computer-science/)
- **HARD Dataset**: [Original Repository](https://github.com/elnagara/HARD-Arabic-Dataset)
- **Hugging Face Models**: [Transformers Library](https://huggingface.co/models)

---

⭐ If you find this work useful, please consider starring the repository!
