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

The repository includes code in two formats for maximum flexibility:

### Python Scripts (Standalone Execution)

#### **experiment1_main.py**
**Converted from**: `Arabic_Sentiment_Analysis_SingleRuns.ipynb`

**Purpose**: Main ensemble learning experiments with 5-seed evaluation

**Features**:
- Automated dataset loading and preprocessing
- Training pipeline for 4 transformer models (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT)
- Implementation of 4 ensemble strategies (Hard/Soft/Weighted Voting, Stacking)
- Cross-seed evaluation (seeds: 42, 123, 456, 789, 2024)
- Statistical significance testing (paired t-tests)
- Results saving and visualization

**Models Evaluated**:
- **AraBERT** (aubmindlab/bert-base-arabert): BERT trained on 77GB Arabic text with Farasa segmentation
- **MARBERT** (UBC-NLP/MARBERT): BERT trained on 1B Arabic tweets with dialectal focus
- **XLM-RoBERTa** (xlm-roberta-base): Multilingual model covering 100 languages
- **CAMeLBERT** (CAMeL-Lab/bert-base-arabic-camelbert-msa): Morphologically-aware Arabic BERT

**Ensemble Methods**:
1. **Hard Voting**: Majority vote from all models
2. **Soft Voting**: Average of probability distributions
3. **Weighted Voting**: Probability weighted by validation accuracy
4. **Stacking**: Logistic regression meta-classifier on base model outputs

**Training Configuration**:
- Learning rate: 2e-5 with linear warmup (10% of training steps)
- Batch size: 16 (with gradient accumulation if needed)
- Max epochs: 3-5 with early stopping (patience=3 on validation loss)
- Optimizer: AdamW with weight decay 0.01
- Loss function: Cross-entropy loss

**Outputs**:
```
results/
├── individual_results.csv          # Performance metrics for each model/seed
├── ensemble_results.csv            # Ensemble performance metrics
├── significance_tests.csv          # Statistical test results
├── confusion_matrices/             # Confusion matrix visualizations
└── performance_plots/              # Training curves and comparisons
```

**Expected Runtime**: ~29.45 hours total
- Per model training: ~1.46 hours (on NVIDIA A100-40GB)
- 4 models × 5 seeds = 29.45 hours
- Ensemble inference: ~30 minutes additional

---

#### **experiment2_lexicon.py**
**Converted from**: `Hybrid_Sentiment_MultiLexicon.ipynb`

**Purpose**: Evaluate lexicon augmentation at high baselines (>96% F1)

**Features**:
- Hybrid classification combining transformers with sentiment lexicons
- Multi-lexicon comparison (general vs domain-specific)
- Hyperparameter tuning for adjustment weights (α)
- Statistical testing for improvement significance

**Lexicons Tested**:
1. **LABR (General-purpose)**: 
   - Size: 4,366 entries
   - Coverage: Modern Standard Arabic + Egyptian dialect
   - Source: Large-scale Arabic Book Reviews dataset

2. **Custom Hotel (Domain-specific)**:
   - Size: 2,000 entries
   - Coverage: Hotel/hospitality domain terms
   - Extraction: TF-IDF from HARD training data

**Hybrid Method**:
```python
# Combine transformer predictions with lexicon scores
hybrid_prob = baseline_prob + α × normalized_lexicon_score
```
- Adjustment weights tested: α ∈ {0.05, 0.10, 0.15, 0.20, 0.25}
- Normalization: Min-max scaling to [0, 1]
- Decision threshold: 0.5

**Key Finding**: 
No statistically significant improvement (p > 0.05) across all α values, revealing a performance ceiling at 96.42% F1 where transformer models have internalized sentiment patterns.

**Outputs**:
```
results/
├── lexicon_augmentation_results.csv    # Performance with different α values
├── best_alpha_per_lexicon.csv          # Optimal α for each lexicon
├── statistical_tests.csv               # Significance tests
└── lexicon_comparison_plot.png         # Visualization of results
```

**Expected Runtime**: ~2-3 hours
- Baseline model: Pre-trained (1.46 hours)
- Lexicon scoring: ~15-20 minutes per lexicon
- Grid search over α values: ~30 minutes

---

#### **analyze_errors.py**
**Purpose**: Manual error annotation analysis

**Features**:
- Loads False Positive and False Negative annotations
- Computes label noise statistics
- Distinguishes dataset quality issues from model limitations
- Generates distribution plots and detailed reports

**Annotation Protocol**:
- **Categories**: Noise (mislabeled), Correct (true error), Not Sure (ambiguous)
- **Process**: Two independent annotators → Reconciliation → Final decision
- **Sample**: High-confidence errors (probability > 0.98) from CAMeLBERT seed 42

**Analysis Results**:
- **False Positives**: 260 samples analyzed
  - 49.2% Noise (mislabeled as Negative but actually Positive)
  - 27.7% Not Sure (ambiguous sentiment)
  - 23.1% Correct (true model errors)

- **False Negatives**: 141 samples analyzed
  - 32.6% Noise (mislabeled as Positive but actually Negative)
  - 29.1% Not Sure (ambiguous sentiment)
  - 38.3% Correct (true model errors)

**Key Insight**: 43.3% of classification errors stem from dataset label noise rather than model limitations, suggesting effective performance ceiling on clean data: ~97.5-98.0% F1

**Outputs**:
```
results/
├── error_analysis_report.txt       # Detailed statistics
├── error_distribution.png          # Visualization of FP/FN categories
└── noise_statistics.json           # Structured results
```

**Expected Runtime**: < 1 minute

---

### Jupyter Notebooks (Interactive Exploration)

#### **Arabic_Sentiment_Analysis_SingleRuns.ipynb**
- **Original Google Colab notebook** for Experiment 1
- Includes markdown documentation and visualizations
- Google Drive integration for checkpoints
- Step-by-step execution with explanatory text
- Designed for: A100 GPU on Google Colab

#### **Hybrid_Sentiment_MultiLexicon.ipynb**
- **Original Google Colab notebook** for Experiment 2
- Interactive lexicon loading and preprocessing
- Real-time performance comparison plots
- Hyperparameter exploration interface
- Designed for: A100 GPU on Google Colab

**Usage**:
1. Upload notebooks to Google Colab
2. Mount Google Drive for data access
3. Run cells sequentially
4. Results saved to Drive automatically

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

### Option 1: Python Scripts (Recommended for Reproducibility)

#### Step 1: Clone Repository
```bash
git clone https://github.com/malhawarat/Ens_Learning_Arabic_Sentiment.git
cd Ens_Learning_Arabic_Sentiment
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Extract Dataset
```bash
cd data
unzip balanced-reviews.csv.zip
cd ..
```

#### Step 5: Run Main Ensemble Experiments
```bash
cd code
python experiment1_main.py
```

**What happens**:
- Downloads pre-trained models from Hugging Face
- Fine-tunes 4 models across 5 seeds (⏱️ ~29 hours on A100)
- Trains ensemble methods
- Evaluates on test set
- Performs statistical significance testing
- Saves results to `../results/`

**Expected Output**:
```
results/
├── individual_results.csv          # Per-model, per-seed metrics
├── ensemble_results.csv            # Ensemble performance
├── significance_tests.csv          # Statistical tests (p-values, t-stats)
├── confusion_matrices/             # PNG visualizations
└── performance_plots/              # Training curves
```

**Monitor Progress**:
- Training logs display real-time loss and accuracy
- Progress bars show epoch completion
- Validation metrics printed after each epoch
- Best checkpoints saved automatically

#### Step 6: Run Lexicon Augmentation Experiments
```bash
python experiment2_lexicon.py
```

**What happens**:
- Loads best model from Experiment 1
- Downloads/loads sentiment lexicons
- Tests multiple α values (0.05-0.25)
- Compares general vs domain-specific lexicons
- Statistical testing for improvements
- Saves results to `../results/`

**Expected Output**:
```
results/
├── lexicon_augmentation_results.csv    # All α values tested
├── best_alpha_per_lexicon.csv          # Optimal configurations
├── statistical_tests.csv               # Significance tests
└── lexicon_comparison_plot.png         # Visualization
```

**Expected Runtime**: ~2-3 hours

#### Step 7: Analyze Error Annotations
```bash
python analyze_errors.py
```

**What happens**:
- Loads annotated errors from `../AnnotatedErrors/`
- Computes label noise statistics
- Creates visualizations
- Generates detailed report

**Expected Output**:
```
results/
├── error_analysis_report.txt       # Complete statistics
├── error_distribution.png          # Bar charts (FP/FN)
└── noise_statistics.json           # Structured data
```

**Expected Runtime**: < 1 minute

---

### Option 2: Jupyter Notebooks (Interactive)

#### For Google Colab:

1. **Upload Notebooks**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `Arabic_Sentiment_Analysis_SingleRuns.ipynb`
   - Upload `Hybrid_Sentiment_MultiLexicon.ipynb`

2. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Upload Dataset**:
   - Upload `balanced-reviews.csv` to Google Drive
   - Update path in notebook

4. **Run Cells Sequentially**:
   - Click "Runtime" → "Run all"
   - Or run cells individually (Ctrl/Cmd + Enter)

5. **GPU Settings**:
   - Go to "Runtime" → "Change runtime type"
   - Select "GPU" (preferably A100)
   - Click "Save"

#### For Local Jupyter:

1. **Install Jupyter**:
   ```bash
   pip install jupyter ipywidgets
   ```

2. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

3. **Open Notebooks**:
   - Navigate to `code/` directory
   - Open `.ipynb` files
   - Run cells sequentially

**Note**: Local execution requires sufficient GPU memory (16GB+ recommended)

---

### Expected Computational Resources

| Task | GPU | RAM | Time | Storage |
|------|-----|-----|------|---------|
| Experiment 1 (all seeds) | A100-40GB | 32GB | ~29 hours | ~5GB |
| Experiment 1 (single seed) | V100-16GB | 16GB | ~6 hours | ~1GB |
| Experiment 2 | A100-40GB | 32GB | ~2-3 hours | ~1GB |
| Error Analysis | CPU only | 8GB | < 1 min | < 100MB |

**Minimum Requirements**:
- GPU: 16GB VRAM (can run single seed at a time)
- RAM: 16GB system memory
- Storage: 10GB free space
- Internet: For downloading pre-trained models

**Recommended Setup**:
- GPU: NVIDIA A100 40GB or V100 32GB
- RAM: 32GB+ system memory
- Storage: 20GB free space (for checkpoints)
- Internet: High-speed for model downloads

---

### Troubleshooting

**Out of Memory Error**:
```bash
# Reduce batch size in script
# Edit experiment1_main.py:
BATCH_SIZE = 8  # Instead of 16
```

**CUDA Not Available**:
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Model Download Issues**:
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Or use offline mode with pre-downloaded models
export TRANSFORMERS_OFFLINE=1
```

**Slow Training**:
- Enable mixed precision training (FP16)
- Increase gradient accumulation steps
- Use distributed training on multiple GPUs

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
│   ├── experiment1_main.py               # Main ensemble experiments (converted from notebook)
│   ├── experiment2_lexicon.py            # Lexicon augmentation (converted from notebook)
│   ├── analyze_errors.py                 # Error analysis script
│   ├── Arabic_Sentiment_Analysis_SingleRuns.ipynb    # Original notebook (Experiment 1)
│   └── Hybrid_Sentiment_MultiLexicon.ipynb           # Original notebook (Experiment 2)
├── AnnotatedErrors/
│   ├── camelbert_seed42_FP_Annotated.xlsx    # False Positive annotations 
│   └── camelbert_seed42_FN_Annotated.xlsx    # False Negative annotations
├── results/                              # Output directory (created on run)
├── checkpoints/                          # Model checkpoints (created on run)
├── requirements.txt                      # Python dependencies
├── LICENSE                              # MIT License
└── README.md                            # This file
```

**Note**: The code is provided in two formats:
1. **Python scripts** (.py files) - For direct execution
2. **Jupyter notebooks** (.ipynb files) - Original Google Colab notebooks with markdown documentation

---

## Citations

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
