#!/usr/bin/env python3
"""
Arabic Sentiment Analysis - Lexicon Augmentation Experiments
Experiment 2: Hybrid Approach with Sentiment Lexicons
"""

# Cell: Combine LABR Lexicon (Correct Filenames)

import os

print("="*70)
print("COMBINING LABR LEXICON")
print("="*70)

# Path to your lexicons folder
lexicons_dir = '/content/drive/MyDrive/arabic_sentiment_analysis/hybrid_experiment/lexicons'

# Input files (with correct names)
pos_file = f'{lexicons_dir}/Pos.txt'
neg_file = f'{lexicons_dir}/Neg.txt'

# Output file
output_file = f'{lexicons_dir}/LABR_Lexicon.txt'

# Check files
print(f"\nChecking files...")
print(f"  Pos.txt exists: {os.path.exists(pos_file)} ✓")
print(f"  Neg.txt exists: {os.path.exists(neg_file)} ✓")

# Read positive words
print(f"\n📖 Reading Pos.txt...")
with open(pos_file, 'r', encoding='utf-8') as f:
    positive = [line.strip() for line in f if line.strip() and not line.startswith('#')]
print(f"   ✓ Found {len(positive):,} positive words")

# Read negative words
print(f"\n📖 Reading Neg.txt...")
with open(neg_file, 'r', encoding='utf-8') as f:
    negative = [line.strip() for line in f if line.strip() and not line.startswith('#')]
print(f"   ✓ Found {len(negative):,} negative words")

# Combine and save
print(f"\n💾 Creating combined lexicon...")
with open(output_file, 'w', encoding='utf-8') as f:
    # Header
    f.write("word\tscore\tpolarity\n")

    # Positive words
    for word in positive:
        f.write(f"{word}\t0.8\tpositive\n")

    # Negative words
    for word in negative:
        f.write(f"{word}\t-0.8\tnegative\n")

total = len(positive) + len(negative)

print(f"   ✓ Saved to: LABR_Lexicon.txt")

print("\n" + "="*70)
print("STATISTICS")
print("="*70)
print(f"Positive words: {len(positive):,}")
print(f"Negative words: {len(negative):,}")
print(f"Total entries:  {total:,}")
print("="*70)

print(f"\n✅ SUCCESS!")
print(f"   Created: LABR_Lexicon.txt")
print(f"   Location: {output_file}")
print(f"\n✓ Ready to use in your notebook!")
print("="*70)

class HybridConfig:
    """Configuration for hybrid experiment"""

    # Paths
    PROJECT_PATH = '/content/drive/MyDrive/arabic_sentiment_analysis'
    DATA_DIR = f'{PROJECT_PATH}/data'
    MODEL_DIR = f'{PROJECT_PATH}/saved_models'

    HYBRID_PATH = f'{PROJECT_PATH}/hybrid_experiment'
    HYBRID_LEXICONS = f'{HYBRID_PATH}/lexicons'
    HYBRID_RESULTS = f'{HYBRID_PATH}/results'
    HYBRID_FIGURES = f'{HYBRID_PATH}/figures'
    HYBRID_ANALYSIS = f'{HYBRID_PATH}/analysis'

    # Dataset
    DATASET_FILE = 'balanced-reviews.csv'
    TEST_RATIO = 0.2
    RANDOM_SEED = 42  # ← CRITICAL for reproducibility!

    # Models (from original experiment)
    MODELS = {
        'arabert': 'aubmindlab/bert-base-arabert',
        'marbert': 'UBC-NLP/MARBERT',
        'xlm-roberta': 'xlm-roberta-base',
        'camelbert': 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
    }

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    MAX_LENGTH = 512

config = HybridConfig()

print("="*70)
print("CONFIGURATION")
print("="*70)
print(f"Device: {config.DEVICE}")
print(f"Random seed: {config.RANDOM_SEED}")
print(f"Project: {config.PROJECT_PATH}")
print(f"Lexicons: {config.HYBRID_LEXICONS}")
print("="*70)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("="*70)
print("LOADING DATASET")
print("="*70)

# Set seed
np.random.seed(config.RANDOM_SEED)

print(f"\n📊 Loading dataset (seed={config.RANDOM_SEED})...")
data_path = f"{config.DATA_DIR}/{config.DATASET_FILE}"

if not os.path.exists(data_path):
    print(f"❌ File not found: {data_path}")
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)
print(f"✓ Loaded: {len(df):,} samples")

print(f"\n🔀 Splitting data (seed={config.RANDOM_SEED})...")
print("Using EXACT SAME 80/10/10 split as training...")

# Replicate DataManager.split_data() logic
# This creates the EXACT SAME split as your training notebook

# Extract features and labels
if 'text' in df.columns:
    X_all = df['text'].values
elif 'review' in df.columns:
    X_all = df['review'].values
else:
    X_all = df.iloc[:, 0].values

if 'label' in df.columns:
    y_all = df['label'].values
elif 'rating' in df.columns:
    y_all = (df['rating'] >= 4).astype(int).values
else:
    y_all = df.iloc[:, 1].values

# Step 1: Split off test set (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all,
    test_size=0.10,  # 10% for test (matches training)
    random_state=config.RANDOM_SEED,
    stratify=y_all
)

# Step 2: Split train_val into train (80% of original) and val (10% of original)
# val should be 10/90 = 0.111... of train_val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.111111,  # 10% of original = 0.111 of train_val
    random_state=config.RANDOM_SEED,
    stratify=y_train_val
)

print(f"\n✓ Split complete:")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(X_all)*100:.1f}%)")
print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X_all)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X_all)*100:.1f}%)")
print(f"  Test positive: {sum(y_test):,} ({sum(y_test)/len(y_test)*100:.1f}%)")

# Verify test set size matches training
expected_test_size = 10570
if len(X_test) != expected_test_size:
    print(f"\n⚠️  WARNING: Test set size is {len(X_test)}, expected {expected_test_size}")
    print(f"   Difference: {abs(len(X_test) - expected_test_size)} samples")
else:
    print(f"\n✅ Test set size matches training: {len(X_test):,} samples")

print("\n" + "="*70)
print("✓ Data ready!")
print("="*70)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
import pickle

print("="*70)
print("LOADING STACKING ENSEMBLE")
print("="*70)

device = config.DEVICE
print(f"\nDevice: {device}")

# Model configurations
models_config = config.MODELS

# Function to get predictions
def get_predictions_from_model(model, tokenizer, texts, batch_size=32):
    """Get predictions from a transformer model"""
    all_probs = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
        batch = texts[i:i+batch_size]

        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)

        all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)

# Load models (seed 42)
print("\n📦 Loading fine-tuned models...\n")

loaded_models = {}

for model_name, model_path_name in models_config.items():
    model_file = f"{config.MODEL_DIR}/{model_name}_seed{config.RANDOM_SEED}.pt"

    if os.path.exists(model_file):
        print(f"Loading {model_name}...")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path_name)

            # Create base model architecture
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path_name,
                num_labels=2
            )

            # Load your trained weights
            checkpoint = torch.load(model_file, map_location=device, weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                # checkpoint is the model itself
                model = checkpoint
                model.to(device)
                model.eval()

                loaded_models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer
                }

                size_mb = os.path.getsize(model_file) / (1024*1024)
                print(f"  ✓ {model_name} loaded ({size_mb:.1f} MB)")
                continue

            # Load state dict into model
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            model.eval()

            loaded_models[model_name] = {
                'model': model,
                'tokenizer': tokenizer
            }

            size_mb = os.path.getsize(model_file) / (1024*1024)
            print(f"  ✓ {model_name} loaded ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
    else:
        print(f"  ⚠️  File not found: {model_file}")

print(f"\n✓ Loaded {len(loaded_models)}/{len(models_config)} models")

if len(loaded_models) == 0:
    raise Exception("No models loaded!")

# Get predictions from base models
print("\n🔮 Getting predictions on test set...\n")

base_predictions = {}
base_probabilities = {}

for model_name, model_dict in loaded_models.items():
    print(f"{model_name}:")

    probs = get_predictions_from_model(
        model_dict['model'],
        model_dict['tokenizer'],
        X_test,
        batch_size=config.BATCH_SIZE
    )

    preds = np.argmax(probs, axis=1)
    base_predictions[model_name] = preds
    base_probabilities[model_name] = probs[:, 1]

    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, preds)
    print(f"  F1: {f1:.4f} ({f1*100:.2f}%)")

# Create meta-features
print("\n📊 Creating stacking meta-features...")

X_meta_test = np.column_stack([
    base_probabilities[name] for name in sorted(loaded_models.keys())
])

print(f"  Shape: {X_meta_test.shape}")

# Load meta-learner or use averaging
meta_learner_path = f"{config.MODEL_DIR}/meta_learner.pkl"

if os.path.exists(meta_learner_path) and os.path.getsize(meta_learner_path) > 0:
    print(f"\n📦 Loading meta-learner...")
    try:
        with open(meta_learner_path, 'rb') as f:
            meta_learner = pickle.load(f)
        print("  ✓ Meta-learner loaded")
    except:
        meta_learner = None
else:
    meta_learner = None

# Simple averaging if no meta-learner
if meta_learner is None:
    print("\n⚠️  Using probability averaging")

    class SimpleMetaLearner:
        def predict(self, X_meta):
            avg_probs = X_meta.mean(axis=1)
            return (avg_probs > 0.5).astype(int)

        def predict_proba(self, X_meta):
            avg_probs = X_meta.mean(axis=1)
            return np.column_stack([1 - avg_probs, avg_probs])

    meta_learner = SimpleMetaLearner()

# Test ensemble
print("\n🧪 Testing ensemble...\n")

stacking_preds = meta_learner.predict(X_meta_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

baseline_acc = accuracy_score(y_test, stacking_preds)
baseline_prec = precision_score(y_test, stacking_preds)
baseline_rec = recall_score(y_test, stacking_preds)
baseline_f1 = f1_score(y_test, stacking_preds)

print("="*70)
print("BASELINE STACKING RESULTS")
print("="*70)
print(f"\nAccuracy:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"Precision: {baseline_prec:.4f} ({baseline_prec*100:.2f}%)")
print(f"Recall:    {baseline_rec:.4f} ({baseline_rec*100:.2f}%)")
print(f"F1-Score:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")

print(f"\n✓ Baseline F1: {baseline_f1:.4f}")
print("✓ Using this as baseline for lexicon comparison")

# Create stacking model wrapper
class StackingEnsemble:
    def __init__(self, base_models, meta_learner, device='cuda'):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.device = device

    def predict(self, texts):
        base_probs = []
        for model_name in sorted(self.base_models.keys()):
            model_dict = self.base_models[model_name]
            probs = get_predictions_from_model(
                model_dict['model'],
                model_dict['tokenizer'],
                texts,
                batch_size=config.BATCH_SIZE
            )
            base_probs.append(probs[:, 1])

        X_meta = np.column_stack(base_probs)
        return self.meta_learner.predict(X_meta)

    def predict_proba(self, texts):
        base_probs = []
        for model_name in sorted(self.base_models.keys()):
            model_dict = self.base_models[model_name]
            probs = get_predictions_from_model(
                model_dict['model'],
                model_dict['tokenizer'],
                texts,
                batch_size=config.BATCH_SIZE
            )
            base_probs.append(probs[:, 1])

        X_meta = np.column_stack(base_probs)
        return self.meta_learner.predict_proba(X_meta)

stacking_model = StackingEnsemble(loaded_models, meta_learner, device)

print("\n" + "="*70)
print("✓ STACKING MODEL READY!")
print("="*70)
print(f"Baseline F1: {baseline_f1:.4f}")
print("✓ Ready for hybrid evaluation!")
print("="*70)

from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

print("="*70)
print("GENERIC LEXICON FRAMEWORK")
print("="*70)

@dataclass
class LexiconMetadata:
    """Metadata about a lexicon"""
    name: str
    source: str
    size: int
    citation: str
    dialect_coverage: List[str]
    description: str

class BaseLexiconLoader(ABC):
    """Abstract base class for lexicon loaders"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lexicon = {}

        if os.path.exists(file_path):
            self.load()
        else:
            print(f"⚠️  File not found: {file_path}")

    @abstractmethod
    def load(self):
        """Load lexicon from file"""
        pass

    @abstractmethod
    def get_metadata(self) -> LexiconMetadata:
        """Return lexicon metadata"""
        pass

    def get_lexicon(self) -> Dict[str, float]:
        return self.lexicon

    def compute_stats(self) -> Dict:
        """Compute statistics"""
        if not self.lexicon:
            return {}

        positive = sum(1 for s in self.lexicon.values() if s > 0)
        negative = sum(1 for s in self.lexicon.values() if s < 0)
        neutral = sum(1 for s in self.lexicon.values() if s == 0)

        return {
            'total_entries': len(self.lexicon),
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }

    def analyze_coverage(self, texts: List[str]) -> Dict:
        """Analyze coverage on dataset"""
        if not self.lexicon:
            return {}

        coverage = {
            'reviews_with_terms': 0,
            'total_terms': 0,
            'terms_per_review': []
        }

        for text in texts:
            tokens = text.split()
            found = sum(1 for token in tokens if token in self.lexicon)
            if found > 0:
                coverage['reviews_with_terms'] += 1
            coverage['total_terms'] += found
            coverage['terms_per_review'].append(found)

        coverage['coverage_rate'] = coverage['reviews_with_terms'] / len(texts)
        coverage['avg_terms'] = np.mean(coverage['terms_per_review'])
        coverage['std_terms'] = np.std(coverage['terms_per_review'])

        return coverage

    def print_info(self):
        """Print lexicon info"""
        metadata = self.get_metadata()
        stats = self.compute_stats()

        print("="*70)
        print(f"LEXICON: {metadata.name}")
        print("="*70)
        print(f"Source: {metadata.source}")
        print(f"Dialects: {', '.join(metadata.dialect_coverage)}")
        print(f"\nSize: {stats.get('total_entries', 0):,} entries")
        print(f"  Positive: {stats.get('positive', 0):,}")
        print(f"  Negative: {stats.get('negative', 0):,}")
        print(f"\nCitation: {metadata.citation}")
        print("="*70)


# Lexicon Loaders
class LABRLoader(BaseLexiconLoader):
    """LABR/dr_samha_lex lexicon loader"""

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Skip header if present
                first_line = f.readline()
                if not first_line.startswith('word'):
                    # Not a header, process it
                    parts = first_line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            score = float(parts[1].strip())
                            self.lexicon[word] = score
                        except ValueError:
                            polarity = parts[2].strip().lower() if len(parts) > 2 else parts[1].strip().lower()
                            if 'pos' in polarity:
                                self.lexicon[word] = 0.8
                            elif 'neg' in polarity:
                                self.lexicon[word] = -0.8

                # Process rest of file
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            score = float(parts[1].strip())
                            self.lexicon[word] = score
                        except ValueError:
                            polarity = parts[2].strip().lower() if len(parts) > 2 else parts[1].strip().lower()
                            if 'pos' in polarity:
                                self.lexicon[word] = 0.8
                            elif 'neg' in polarity:
                                self.lexicon[word] = -0.8

            print(f"✓ Loaded LABR: {len(self.lexicon):,} entries")
        except Exception as e:
            print(f"❌ Error: {e}")

    def get_metadata(self):
        return LexiconMetadata(
            name="LABR (dr_samha_lex)",
            source="El-Beltagy & Ali (2013)",
            size=len(self.lexicon),
            citation="El-Beltagy, S. R., & Ali, A. (2013). Open issues in sentiment analysis of Arabic social media. WASSA 2013.",
            dialect_coverage=["MSA", "Egyptian"],
            description="General-purpose Arabic sentiment lexicon"
        )


class MPQALoader(BaseLexiconLoader):
    """MPQA Arabic lexicon loader"""

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        polarity = parts[1].strip().lower()

                        if 'pos' in polarity:
                            self.lexicon[word] = 0.8 if 'strong' in polarity else 0.6
                        elif 'neg' in polarity:
                            self.lexicon[word] = -0.8 if 'strong' in polarity else -0.6
            print(f"✓ Loaded MPQA: {len(self.lexicon):,} entries")
        except Exception as e:
            print(f"❌ Error: {e}")

    def get_metadata(self):
        return LexiconMetadata(
            name="MPQA Arabic",
            source="Wilson et al. (2005)",
            size=len(self.lexicon),
            citation="Wilson, T., et al. (2005). Recognizing contextual polarity. HLT/EMNLP.",
            dialect_coverage=["MSA"],
            description="Subjectivity lexicon (Arabic translation)"
        )


class ArSenTDLoader(BaseLexiconLoader):
    """ArSenTD-LEV lexicon loader"""

    def load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        try:
                            score = float(parts[1].strip())
                            self.lexicon[word] = score
                        except ValueError:
                            continue
            print(f"✓ Loaded ArSenTD-LEV: {len(self.lexicon):,} entries")
        except Exception as e:
            print(f"❌ Error: {e}")

    def get_metadata(self):
        return LexiconMetadata(
            name="ArSenTD-LEV",
            source="Salameh et al. (2015)",
            size=len(self.lexicon),
            citation="Salameh, M., et al. (2015). Sentiment after translation. NAACL-HLT.",
            dialect_coverage=["Levantine"],
            description="Levantine dialect-specific lexicon"
        )


class CustomHotelLoader(BaseLexiconLoader):
    """Custom hotel lexicon builder"""

    def __init__(self, file_path: str, train_texts=None, train_labels=None):
        self.train_texts = train_texts
        self.train_labels = train_labels
        super().__init__(file_path)

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    next(f, None)  # Skip header if exists
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and parts[0].strip():
                            word = parts[0].strip()
                            try:
                                score = float(parts[1].strip())
                                self.lexicon[word] = score
                            except ValueError:
                                continue
                print(f"✓ Loaded custom: {len(self.lexicon):,} entries")
            except Exception as e:
                print(f"❌ Error loading: {e}")
        elif self.train_texts is not None and self.train_labels is not None:
            print("📊 Building custom lexicon from training data...")
            self._build_from_data()
            self._save()
        else:
            print("❌ No training data provided to build lexicon")

    def _build_from_data(self):
        """Build lexicon from training data"""
        import re
        from collections import Counter

        print("  🔍 Extracting words from positive reviews...")
        pos_words = Counter()
        neg_words = Counter()

        # Simple Arabic tokenization
        def tokenize(text):
            # Remove non-Arabic characters except spaces
            text = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
            # Split and filter short words
            words = [w for w in text.split() if len(w) >= 2]
            return words

        # Process training data
        for text, label in zip(self.train_texts, self.train_labels):
            words = tokenize(text)
            if label == 1:
                pos_words.update(words)
            else:
                neg_words.update(words)

        print(f"  📈 Positive words: {len(pos_words):,}")
        print(f"  📈 Negative words: {len(neg_words):,}")

        print("  🧮 Computing sentiment scores...")

        # Get all words
        all_words = set(pos_words.keys()) | set(neg_words.keys())

        # Calculate sentiment scores
        for word in all_words:
            pos_count = pos_words.get(word, 0)
            neg_count = neg_words.get(word, 0)
            total = pos_count + neg_count

            # Minimum frequency threshold
            if total < 5:
                continue

            # Calculate sentiment score
            score = (pos_count - neg_count) / total

            # Only keep strong sentiment words
            if abs(score) > 0.3:
                self.lexicon[word] = score

        # Keep top 2000 by absolute score
        sorted_terms = sorted(self.lexicon.items(), key=lambda x: abs(x[1]), reverse=True)
        self.lexicon = dict(sorted_terms[:2000])

        pos_count = sum(1 for s in self.lexicon.values() if s > 0)
        neg_count = sum(1 for s in self.lexicon.values() if s < 0)

        print(f"  ✓ Built custom lexicon: {len(self.lexicon):,} entries")
        print(f"     Positive: {pos_count:,}")
        print(f"     Negative: {neg_count:,}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write("word\tscore\tpolarity\n")
                for word, score in sorted(self.lexicon.items(), key=lambda x: abs(x[1]), reverse=True):
                    polarity = "positive" if score > 0 else "negative"
                    f.write(f"{word}\t{score:.4f}\t{polarity}\n")
            print(f"  💾 Saved to: {self.file_path}")
        except Exception as e:
            print(f"  ⚠️  Save failed: {e}")

    def get_metadata(self):
        return LexiconMetadata(
            name="Custom Hotel",
            source="Extracted from HARD",
            size=len(self.lexicon),
            citation="Domain-specific lexicon from HARD dataset",
            dialect_coverage=["Mixed (data-driven)"],
            description="Hotel review sentiment terms"
        )
print("\n✓ Generic Framework Ready")
print("="*70)

print("\n" + "="*70)
print("LEXICON ANALYZER")
print("="*70)

class ArabicLexiconAnalyzer:
    """Lexicon-based sentiment analyzer"""

    def __init__(self, custom_lexicon: Dict[str, float]):
        if not custom_lexicon:
            raise ValueError("Must provide lexicon!")

        self.lexicon = custom_lexicon
        print(f"✓ Analyzer with {len(self.lexicon):,} terms")

        self.negations = [
            "لا", "لم", "لن", "ليس", "ليست", "لسنا",
            "غير", "بدون", "ما", "مش", "مو"
        ]

        self.intensifiers = {
            "جدا": 1.5, "جداً": 1.5,
            "كثير": 1.4, "كثيراً": 1.4,
            "للغاية": 1.6,
            "حقا": 1.3, "حقاً": 1.3
        }

        self.diminishers = {
            "قليلا": 0.5, "قليلاً": 0.5,
            "شوي": 0.6, "شوية": 0.6,
            "بعض": 0.7
        }

    def detect_negation(self, tokens: List[str], index: int, window: int = 3) -> bool:
        start = max(0, index - window)
        context = tokens[start:index]
        return any(neg in context for neg in self.negations)

    def get_modifier(self, tokens: List[str], index: int) -> float:
        if index + 1 < len(tokens):
            next_word = tokens[index + 1]
            if next_word in self.intensifiers:
                return self.intensifiers[next_word]
            if next_word in self.diminishers:
                return self.diminishers[next_word]

        if index > 0:
            prev_word = tokens[index - 1]
            if prev_word in self.intensifiers:
                return self.intensifiers[prev_word]
            if prev_word in self.diminishers:
                return self.diminishers[prev_word]

        return 1.0

    def analyze(self, text: str):
        tokens = text.split()
        sentiment_scores = []

        for i, token in enumerate(tokens):
            if token in self.lexicon:
                score = self.lexicon[token]

                if self.detect_negation(tokens, i):
                    score *= -1

                modifier = self.get_modifier(tokens, i)
                score *= modifier

                sentiment_scores.append(score)

        if not sentiment_scores:
            return 0.0, 0.0

        avg_score = np.mean(sentiment_scores)

        num_terms = len(sentiment_scores)
        term_confidence = min(1.0, num_terms / 5)

        if avg_score != 0:
            same_sign = sum(1 for s in sentiment_scores if (s > 0) == (avg_score > 0))
            agreement = same_sign / num_terms
        else:
            agreement = 0.5

        confidence = (term_confidence * 0.6 + agreement * 0.4)

        return avg_score, confidence

    def predict(self, text: str):
        score, confidence = self.analyze(text)
        label = 1 if score > 0 else 0
        return label, confidence

print("✓ Lexicon Analyzer Ready")
print("="*70)

print("\n" + "="*70)
print("HYBRID SYSTEM")
print("="*70)

class HybridSentimentAnalyzer:
    """Hybrid: Stacking + Lexicon"""

    def __init__(self, stacking_model, lexicon_analyzer):
        self.stacking = stacking_model
        self.lexicon = lexicon_analyzer
        print("✓ Hybrid system initialized")

    def predict(self, text: str, strategy: str = "disagreement"):
        # Get predictions
        transformer_pred = self.stacking.predict([text])[0]
        transformer_prob = self.stacking.predict_proba([text])[0]
        transformer_conf = max(transformer_prob)

        lexicon_pred, lexicon_conf = self.lexicon.predict(text)

        # Fusion strategies
        if strategy == "override":
            if transformer_conf < 0.6 and lexicon_conf > 0.75:
                return lexicon_pred, lexicon_conf
            return transformer_pred, transformer_conf

        elif strategy == "weighted":
            w_trans = 0.7
            w_lex = 0.3
            combined_score = (w_trans * transformer_prob[1] +
                            w_lex * (lexicon_conf if lexicon_pred == 1 else 1-lexicon_conf))
            final_pred = 1 if combined_score > 0.5 else 0
            return final_pred, combined_score

        elif strategy == "disagreement":
            if transformer_pred != lexicon_pred:
                if lexicon_conf > 0.7:
                    return lexicon_pred, lexicon_conf
            return transformer_pred, transformer_conf

        return transformer_pred, transformer_conf

    def evaluate(self, X_test, y_test, strategy="disagreement"):
        predictions = []
        for text in X_test:
            pred, _ = self.predict(text, strategy)
            predictions.append(pred)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        results = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'strategy': strategy,
            'stats': {
                'agreements': sum(1 for i, p in enumerate(predictions) if p == self.stacking.predict([X_test[i]])[0]),
                'disagreements': sum(1 for i, p in enumerate(predictions) if p != self.stacking.predict([X_test[i]])[0]),
                'lexicon_overrides': 0  # Simplified
            }
        }

        return results

print("✓ Hybrid System Ready")
print("="*70)

print("\n" + "="*70)
print("LEXICON EVALUATOR")
print("="*70)

@dataclass
class LexiconEvaluation:
    lexicon_name: str
    f1_score: float
    delta_f1: float
    coverage_rate: float
    avg_terms: float
    corrected_errors: int
    new_errors: int
    net_improvement: int

class LexiconEvaluator:
    """Evaluate hybrid with different lexicons"""

    def __init__(self, stacking_model, X_test, y_test, baseline_f1):
        self.stacking = stacking_model
        self.X_test = X_test
        self.y_test = y_test
        self.baseline_f1 = baseline_f1
        self.baseline_preds = stacking_model.predict(X_test)
        self.evaluations = {}

    def evaluate_lexicon(self, lexicon_loader, strategy="disagreement"):
        print(f"\n{'='*70}")
        print(f"EVALUATING: {lexicon_loader.get_metadata().name}")
        print(f"{'='*70}")

        lexicon_dict = lexicon_loader.get_lexicon()

        if not lexicon_dict:
            print("❌ Empty lexicon")
            return None

        coverage = lexicon_loader.analyze_coverage(self.X_test)
        print(f"Coverage: {coverage['coverage_rate']*100:.1f}%")
        print(f"Avg terms: {coverage['avg_terms']:.2f}")

        analyzer = ArabicLexiconAnalyzer(custom_lexicon=lexicon_dict)
        hybrid = HybridSentimentAnalyzer(self.stacking, analyzer)

        results = hybrid.evaluate(self.X_test, self.y_test, strategy)

        hybrid_preds = []
        for text in self.X_test:
            pred, _ = hybrid.predict(text, strategy)
            hybrid_preds.append(pred)
        hybrid_preds = np.array(hybrid_preds)

        baseline_errors = set(np.where(self.baseline_preds != self.y_test)[0])
        hybrid_errors = set(np.where(hybrid_preds != self.y_test)[0])

        corrected = len(baseline_errors - hybrid_errors)
        new_errors = len(hybrid_errors - baseline_errors)

        delta_f1 = results['f1_score'] - self.baseline_f1

        eval_result = LexiconEvaluation(
            lexicon_name=lexicon_loader.get_metadata().name,
            f1_score=results['f1_score'],
            delta_f1=delta_f1,
            coverage_rate=coverage['coverage_rate'],
            avg_terms=coverage['avg_terms'],
            corrected_errors=corrected,
            new_errors=new_errors,
            net_improvement=corrected - new_errors
        )

        print(f"\nF1: {eval_result.f1_score:.4f} ({delta_f1:+.4f})")
        print(f"Net: {eval_result.net_improvement:+d} errors")

        self.evaluations[eval_result.lexicon_name] = eval_result
        return eval_result

    def compare_all(self):
        print("\n" + "="*100)
        print("LEXICON COMPARISON")
        print("="*100)

        print(f"\n{'Lexicon':<20} {'F1':<8} {'Δ F1':<8} {'Coverage':<10} {'Net':<10}")
        print("-"*100)

        for name, e in sorted(self.evaluations.items(), key=lambda x: x[1].f1_score, reverse=True):
            print(f"{name:<20} {e.f1_score:.4f}   {e.delta_f1:+.4f}   {e.coverage_rate*100:>5.1f}%     {e.net_improvement:+d}")

        print("="*100)

        best = max(self.evaluations.values(), key=lambda x: x.f1_score)
        print(f"\n🏆 BEST: {best.lexicon_name} (F1={best.f1_score:.4f})")
        print("="*100)

    def get_best(self):
        if not self.evaluations:
            return None
        return max(self.evaluations.values(), key=lambda x: x.f1_score)

# Initialize evaluator
evaluator = LexiconEvaluator(
    stacking_model=stacking_model,
    X_test=X_test,
    y_test=y_test,
    baseline_f1=baseline_f1
)

print("✓ Evaluator Ready")
print("="*70)

# Build Custom Lexicon Manually

print("="*70)
print("BUILDING CUSTOM HOTEL LEXICON")
print("="*70)

import re
from collections import Counter

# Check data availability
print(f"\n✓ Training data: {len(X_train):,} samples")
print(f"  Positive: {sum(y_train):,}")
print(f"  Negative: {len(y_train) - sum(y_train):,}")

# Tokenization
def tokenize_arabic(text):
    """Extract Arabic words"""
    text = re.sub(r'[^\u0600-\u06FF\s]', '', str(text))
    words = [w for w in text.split() if len(w) >= 2]
    return words

# Extract words by sentiment
print("\n🔍 Extracting words from reviews...")
pos_words = Counter()
neg_words = Counter()

for text, label in zip(X_train[:20000], y_train[:20000]):  # Use first 20k for speed
    words = tokenize_arabic(text)
    if label == 1:
        pos_words.update(words)
    else:
        neg_words.update(words)

print(f"  ✓ Positive vocabulary: {len(pos_words):,} unique words")
print(f"  ✓ Negative vocabulary: {len(neg_words):,} unique words")

# Build lexicon
print("\n🧮 Computing sentiment scores...")
lexicon = {}
all_words = set(pos_words.keys()) | set(neg_words.keys())

for word in all_words:
    pos_count = pos_words.get(word, 0)
    neg_count = neg_words.get(word, 0)
    total = pos_count + neg_count

    if total < 5:  # Minimum frequency
        continue

    score = (pos_count - neg_count) / total

    if abs(score) > 0.3:  # Minimum sentiment strength
        lexicon[word] = score

# Keep top 2000
sorted_terms = sorted(lexicon.items(), key=lambda x: abs(x[1]), reverse=True)
lexicon = dict(sorted_terms[:2000])

pos_count = sum(1 for s in lexicon.values() if s > 0)
neg_count = sum(1 for s in lexicon.values() if s < 0)

print(f"\n✓ Built lexicon:")
print(f"  Total: {len(lexicon):,}")
print(f"  Positive: {pos_count:,}")
print(f"  Negative: {neg_count:,}")

# Save
custom_path = f"{config.HYBRID_LEXICONS}/Custom_Hotel.txt"
os.makedirs(os.path.dirname(custom_path), exist_ok=True)

with open(custom_path, 'w', encoding='utf-8') as f:
    f.write("word\tscore\tpolarity\n")
    for word, score in sorted(lexicon.items(), key=lambda x: abs(x[1]), reverse=True):
        polarity = "positive" if score > 0 else "negative"
        f.write(f"{word}\t{score:.4f}\t{polarity}\n")

print(f"\n💾 Saved to: {custom_path}")
print(f"✓ File size: {os.path.getsize(custom_path)/1024:.1f} KB")

print("\n" + "="*70)
print("✓ CUSTOM LEXICON BUILT!")
print("="*70)

print("="*70)
print("EXPERIMENT: Custom Hotel (OPTIMIZED)")
print("="*70)

custom_path = f"{config.HYBRID_LEXICONS}/Custom_Hotel.txt"
custom_loader = CustomHotelLoader(custom_path)

if len(custom_loader.lexicon) == 0:
    print("❌ Custom lexicon empty!")
else:
    custom_loader.print_info()

    print("\n🚀 Running optimized evaluation...")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Batch size: 128")

    import time
    from tqdm import tqdm

    start_time = time.time()

    # Get baseline predictions (should already be cached)
    baseline_preds = stacking_model.predict(X_test)

    print(f"\n✓ Baseline predictions: {time.time() - start_time:.1f}s")

    # Lexicon analysis
    print("🔍 Analyzing with lexicon...")
    lexicon_scores = []

    for i, text in enumerate(tqdm(X_test, desc="Lexicon analysis")):
        tokens = str(text).split()
        score = sum(custom_loader.lexicon.get(token, 0) for token in tokens)
        lexicon_scores.append(score)

    lexicon_scores = np.array(lexicon_scores)

    print(f"✓ Lexicon analysis: {time.time() - start_time:.1f}s")

    # Hybrid combination (simple: adjust by lexicon signal)
    print("🔗 Combining predictions...")

    # Normalize lexicon scores
    lexicon_normalized = lexicon_scores / (np.abs(lexicon_scores).max() + 1e-10)

    # Get probabilities from baseline
    baseline_probs = stacking_model.predict_proba(X_test)[:, 1]

    # Adjust probabilities by lexicon
    hybrid_probs = baseline_probs + 0.1 * lexicon_normalized  # Small adjustment
    hybrid_probs = np.clip(hybrid_probs, 0, 1)
    hybrid_preds = (hybrid_probs > 0.5).astype(int)

    # Evaluate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    hybrid_acc = accuracy_score(y_test, hybrid_preds)
    hybrid_prec = precision_score(y_test, hybrid_preds)
    hybrid_rec = recall_score(y_test, hybrid_preds)
    hybrid_f1 = f1_score(y_test, hybrid_preds)

    baseline_f1 = f1_score(y_test, baseline_preds)

    print("\n" + "="*70)
    print("RESULTS: Custom Hotel Lexicon")
    print("="*70)

    print(f"\nBaseline F1:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Hybrid F1:    {hybrid_f1:.4f} ({hybrid_f1*100:.2f}%)")
    print(f"Improvement:  {(hybrid_f1-baseline_f1):.4f} (+{(hybrid_f1-baseline_f1)*100:.2f}%)")

    print(f"\nHybrid Accuracy:  {hybrid_acc:.4f} ({hybrid_acc*100:.2f}%)")
    print(f"Hybrid Precision: {hybrid_prec:.4f} ({hybrid_prec*100:.2f}%)")
    print(f"Hybrid Recall:    {hybrid_rec:.4f} ({hybrid_rec*100:.2f}%)")

    # Statistical significance
    from scipy.stats import ttest_rel

    baseline_correct = (baseline_preds == y_test).astype(int)
    hybrid_correct = (hybrid_preds == y_test).astype(int)

    t_stat, p_value = ttest_rel(hybrid_correct, baseline_correct)

    print(f"\n📊 Statistical Test:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"   ✓ Significant improvement (p < 0.05)")
    else:
        print(f"   ⚠️  Not significant (p >= 0.05)")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    print("\n" + "="*70)
    print("✓ Custom lexicon evaluated")
    print("="*70)

print("="*70)
print("EXPERIMENT: LABR (dr_samha_lex) - OPTIMIZED")
print("="*70)

# ADD THIS:
print(f"\n🔍 Test set size: {len(X_test):,} samples")
if len(X_test) != 10570:
    print("⚠️  WARNING: Using wrong test set! Should be 10,570")
    print("   Re-run Cell 7 (data loading) first!")
else:
    print("✅ Test set is correct!")

labr_path = f"{config.HYBRID_LEXICONS}/LABR_Lexicon.txt"

print("="*70)
print("EXPERIMENT: LABR (dr_samha_lex) - OPTIMIZED")
print("="*70)

labr_path = f"{config.HYBRID_LEXICONS}/LABR_Lexicon.txt"
labr_loader = LABRLoader(labr_path)

if not os.path.exists(labr_path):
    print("❌ LABR not found!")
else:
    labr_loader.print_info()

    print("\n🚀 Running optimized evaluation...")

    import time
    start_time = time.time()

    # Get baseline predictions
    baseline_preds = stacking_model.predict(X_test)
    baseline_probs = stacking_model.predict_proba(X_test)[:, 1]

    # Lexicon analysis
    print("🔍 Analyzing with LABR lexicon...")
    lexicon_scores = []

    for text in tqdm(X_test, desc="Lexicon analysis"):
        tokens = str(text).split()
        score = sum(labr_loader.lexicon.get(token, 0) for token in tokens)
        lexicon_scores.append(score)

    lexicon_scores = np.array(lexicon_scores)
    lexicon_normalized = lexicon_scores / (np.abs(lexicon_scores).max() + 1e-10)

    # Hybrid combination
    print("🔗 Combining predictions...")
    hybrid_probs = baseline_probs + 0.1 * lexicon_normalized
    hybrid_probs = np.clip(hybrid_probs, 0, 1)
    hybrid_preds = (hybrid_probs > 0.5).astype(int)

    # Evaluate
    hybrid_f1 = f1_score(y_test, hybrid_preds)
    baseline_f1 = f1_score(y_test, baseline_preds)

    print("\n" + "="*70)
    print("RESULTS: LABR Lexicon")
    print("="*70)
    print(f"\nBaseline F1:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    print(f"Hybrid F1:    {hybrid_f1:.4f} ({hybrid_f1*100:.2f}%)")
    print(f"Improvement:  {(hybrid_f1-baseline_f1):.4f} (+{(hybrid_f1-baseline_f1)*100:.2f}%)")

    # Statistical test
    baseline_correct = (baseline_preds == y_test).astype(int)
    hybrid_correct = (hybrid_preds == y_test).astype(int)
    t_stat, p_value = ttest_rel(hybrid_correct, baseline_correct)

    print(f"\n📊 Statistical Test:")
    print(f"   p-value: {p_value:.4f}")
    print(f"   {'✓ Significant' if p_value < 0.05 else '✗ Not significant'}")

    print("\n" + "="*70)