#!/usr/bin/env python3
"""
Arabic Sentiment Analysis - Main Ensemble Experiments
Experiment 1: Individual Models and Ensemble Methods Comparison
"""

class Config:
    """Central configuration for all experiments"""

    # Paths (Google Drive)
    PROJECT_PATH = '/content/drive/MyDrive/arabic_sentiment_analysis'
    DATA_DIR = f'{PROJECT_PATH}/data'
    MODEL_DIR = f'{PROJECT_PATH}/saved_models'
    RESULTS_DIR = f'{PROJECT_PATH}/results'
    FIGURES_DIR = f'{PROJECT_PATH}/figures'

    # Dataset settings
    DATASET_FILE = 'balanced-reviews.csv'
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Model configurations
    MODELS = {
        'arabert': 'aubmindlab/bert-base-arabert',
        'marbert': 'UBC-NLP/MARBERT',
        'xlm-roberta': 'xlm-roberta-base',
        'camelbert': 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
    }

    # Training hyperparameters
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16  # Reduced for Colab GPU
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01

    # Early stopping
    PATIENCE = 3

    # Reproducibility - Use fewer seeds for faster runtime in Colab
    RANDOM_SEEDS = [42, 123, 456, 789, 2024]  # Reduced from 5 to 2 for faster execution

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Number of labels
    NUM_LABELS = 2

config = Config()
print("✓ Configuration loaded")
print(f"  Device: {config.DEVICE}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Random Seeds: {config.RANDOM_SEEDS}")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
import time
import pickle
import warnings
import gc  # ← ADD THIS LINE

warnings.filterwarnings('ignore')

class ArabicPreprocessor:
    """Arabic text preprocessing utilities"""

    @staticmethod
    def normalize_arabic(text):
        """Normalize Arabic text"""
        if not isinstance(text, str):
            return ""

        # Normalize different forms of Alef
        text = re.sub('[إأآا]', 'ا', text)
        text = re.sub('ى', 'ي', text)
        text = re.sub('ة', 'ه', text)

        # Remove diacritics
        arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', text)

        return text

    @staticmethod
    def clean_text(text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""

        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def preprocess(text):
        """Complete preprocessing pipeline"""
        text = ArabicPreprocessor.normalize_arabic(text)
        text = ArabicPreprocessor.clean_text(text)
        return text


class HARDDataset(Dataset):
    """PyTorch Dataset for HARD"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataManager:
    """Manage dataset loading, preprocessing, and splitting"""

    def __init__(self, config):
        self.config = config
        self.preprocessor = ArabicPreprocessor()

    def load_hard_dataset(self, file_path):
        """Load HARD dataset from CSV/TSV"""
        try:
            df = pd.read_csv(file_path, sep=',', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Please ensure the file is in the correct format.")
                return None

        # Standardize column names
        if 'review' in df.columns:
          df = df.rename(columns={'review': 'text'})
        if 'rating' in df.columns:
          df = df.rename(columns={'rating': 'label'})

        print("Preprocessing texts...")
        df['text'] = df['text'].apply(self.preprocessor.preprocess)

        # Remove empty texts
        df = df[df['text'].str.len() > 0]

        # Ensure labels are 0/1
        label_map = {1: 0, 2: 0, 4: 1, 5: 1}
        df['label'] = df['label'].map(label_map)

        return df

    def split_data(self, df, seed=123):
        """Split data into train/val/test"""
        train_val, test = train_test_split(
            df,
            test_size=self.config.TEST_RATIO,
            random_state=seed,
            stratify=df['label']
        )

        val_ratio_adjusted = self.config.VAL_RATIO / (self.config.TRAIN_RATIO + self.config.VAL_RATIO)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=seed,
            stratify=train_val['label']
        )

        print(f"Data split (seed={seed}):")
        print(f"  Train: {len(train)} samples")
        print(f"  Val:   {len(val)} samples")
        print(f"  Test:  {len(test)} samples")

        return train, val, test

    def create_dataloaders(self, train_df, val_df, test_df, tokenizer, batch_size):
        """Create PyTorch DataLoaders"""
        train_dataset = HARDDataset(
            train_df['text'].values,
            train_df['label'].values,
            tokenizer,
            self.config.MAX_LENGTH
        )

        val_dataset = HARDDataset(
            val_df['text'].values,
            val_df['label'].values,
            tokenizer,
            self.config.MAX_LENGTH
        )

        test_dataset = HARDDataset(
            test_df['text'].values,
            test_df['label'].values,
            tokenizer,
            self.config.MAX_LENGTH
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

print("✓ Data preprocessing classes defined")

import torch.nn as nn
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm.auto import tqdm
import time

class SentimentClassifier:
    """Wrapper for fine-tuning BERT-based models"""

    def __init__(self, model_name, config, device):
        self.config = config
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.NUM_LABELS
        )
        self.model.to(device)

        self.best_val_acc = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        return total_loss / len(train_loader), correct / total

    def evaluate(self, val_loader):
        """Evaluate on validation/test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)

        return avg_loss, accuracy, all_preds, all_labels, all_probs

    def train(self, train_loader, val_loader, save_path):
        """Complete training loop with early stopping"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        num_training_steps = len(train_loader) * self.config.NUM_EPOCHS
        num_warmup_steps = int(num_training_steps * self.config.WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc, _, _, _ = self.evaluate(val_loader)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
              self.best_val_acc = val_acc
              self.patience_counter = 0

              # ✅ FIXED: Save complete model (not just state_dict)
              torch.save(self.model, save_path)
              print(f"✓ Best model saved! Val Acc: {val_acc:.4f}")

              # ✅ Verify classifier was saved
              test_load = torch.load(save_path, map_location='cpu', weights_only=False)
              if hasattr(test_load, 'classifier'):
                  if hasattr(test_load.classifier, 'weight'):
                      classifier_weight_sum = test_load.classifier.weight.sum().item()
                      print(f"   ✓ Classifier saved (weight sum: {classifier_weight_sum:.4f})")
                  else:
                      print(f"   ✓ Classifier saved")
              else:
                  print(f"   ⚠️  WARNING: Classifier missing!")
            else:
                self.patience_counter += 1
                print(f"No improvement ({self.patience_counter}/{self.config.PATIENCE})")

                if self.patience_counter >= self.config.PATIENCE:
                    print("Early stopping triggered!")
                    break

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")

        self.model = torch.load(save_path, weights_only=False)
        self.model.to(self.device)
        return self.best_val_acc, training_time

    def predict(self, test_loader):
        """Get predictions on test set"""
        _, accuracy, preds, labels, probs = self.evaluate(test_loader)
        return accuracy, preds, labels, probs

print("✓ Model training classes defined")

from sklearn.linear_model import LogisticRegression

class EnsembleMethods:
    """Implementation of various ensemble strategies"""

    @staticmethod
    def soft_voting(predictions_probs):
        """Soft voting: Average probabilities from all models"""
        avg_probs = np.mean(predictions_probs, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)
        return final_preds, avg_probs

    @staticmethod
    def hard_voting(predictions):
        """Hard voting: Majority vote on predicted classes"""
        stacked = np.column_stack(predictions)
        final_preds = []
        for row in stacked:
            counts = np.bincount(row)
            final_preds.append(np.argmax(counts))
        return np.array(final_preds)

    @staticmethod
    def weighted_voting(predictions_probs, weights):
        """Weighted voting: Weight models by validation performance"""
        # Convert weights to numpy array and normalize
        weights = np.array(weights, dtype=np.float64)
        weights = weights / np.sum(weights)

        # Initialize weighted probabilities
        weighted_probs = np.zeros_like(predictions_probs[0], dtype=np.float64)

        # Apply weighted sum
        for i, probs in enumerate(predictions_probs):
            probs = np.array(probs, dtype=np.float64)  # Ensure probs is numpy array
            weighted_probs += weights[i] * probs

        final_preds = np.argmax(weighted_probs, axis=1)
        return final_preds, weighted_probs

    @staticmethod
    def stacking(train_probs, train_labels, test_probs):
        """Stacking: Train meta-classifier on base model predictions"""
        X_train = np.hstack([probs for probs in train_probs])
        X_test = np.hstack([probs for probs in test_probs])

        meta_clf = LogisticRegression(max_iter=1000, random_state=42)
        meta_clf.fit(X_train, train_labels)

        final_preds = meta_clf.predict(X_test)
        final_probs = meta_clf.predict_proba(X_test)

        return final_preds, final_probs, meta_clf

print("✓ Ensemble methods defined")

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import json

class Evaluator:
    """Evaluation utilities"""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate all evaluation metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist()
        }

    @staticmethod
    def print_results(name, metrics):
        """Pretty print results"""
        print(f"\n{'='*60}")
        print(f"{name} Results")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix']))

    @staticmethod
    def save_results(results, filepath):
        """Save results to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filepath}")

print("✓ Evaluation utilities defined")

import time
from datetime import datetime

class RuntimeTracker:
    """Track and analyze training runtimes"""

    def __init__(self):
        self.runtimes = {}
        self.start_times = {}
        self.experiment_start = None

    def start_experiment(self):
        self.experiment_start = time.time()
        print(f"🕐 Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def start_seed(self, seed):
        self.start_times[seed] = time.time()
        self.runtimes[seed] = {}

    def log_runtime(self, seed, model_name, runtime_seconds):
        if seed not in self.runtimes:
            self.runtimes[seed] = {}
        self.runtimes[seed][model_name] = runtime_seconds

    def format_hours(self, seconds):
        return seconds / 3600

    def get_seed_total(self, seed):
        if seed in self.runtimes:
            return sum(self.runtimes[seed].values())
        return 0

    def generate_runtime_table(self):
        print("\\n" + "="*100)
        print("RUNTIME SUMMARY TABLE")
        print("="*100)

        models = list(config.MODELS.keys())
        seeds = sorted(self.runtimes.keys())

        header = f"{'Model':<20}"
        for seed in seeds:
            header += f"Seed {seed:<8}"
        header += f"{'Average':<12}{'Total (hrs)':<12}"
        print(header)
        print("-"*100)

        for model in models:
            row = f"{model:<20}"
            model_times = []

            for seed in seeds:
                if seed in self.runtimes and model in self.runtimes[seed]:
                    time_hrs = self.format_hours(self.runtimes[seed][model])
                    row += f"{time_hrs:>6.2f}h   "
                    model_times.append(self.runtimes[seed][model])
                else:
                    row += f"{'N/A':<9}"

            if model_times:
                avg_hrs = self.format_hours(np.mean(model_times))
                total_hrs = self.format_hours(sum(model_times))
                row += f"{avg_hrs:>6.2f}h    {total_hrs:>6.2f}h"

            print(row)

        print("-"*100)
        row = f"{'SEED TOTAL':<20}"
        seed_totals = []
        for seed in seeds:
            total = self.get_seed_total(seed)
            total_hrs = self.format_hours(total)
            row += f"{total_hrs:>6.2f}h   "
            seed_totals.append(total)

        if seed_totals:
            avg_seed_hrs = self.format_hours(np.mean(seed_totals))
            grand_total_hrs = self.format_hours(sum(seed_totals))
            row += f"{avg_seed_hrs:>6.2f}h    {grand_total_hrs:>6.2f}h"
        print(row)
        print("="*100)

        if self.experiment_start:
            total_elapsed = time.time() - self.experiment_start
            print(f"\\n⏱️  Total Experiment Time: {self.format_hours(total_elapsed):.2f} hours")

    def export_runtime_analysis(self, output_file):
        with open(output_file, 'w') as f:
            f.write("="*80 + "\\n")
            f.write("RUNTIME ANALYSIS - FOR PAPER\\n")
            f.write("="*80 + "\\n")

            models = list(config.MODELS.keys())
            seeds = sorted(self.runtimes.keys())

            for model in models:
                model_times = []
                for seed in seeds:
                    if seed in self.runtimes and model in self.runtimes[seed]:
                        model_times.append(self.runtimes[seed][model])
                if model_times:
                    avg_hrs = self.format_hours(np.mean(model_times))
                    f.write(f"{model:15s}: {avg_hrs:.2f} hours (average)\\n")

# Initialize global runtime tracker
runtime_tracker = RuntimeTracker()

print("✓ RuntimeTracker initialized")

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class StatisticalAnalyzer:
    """Statistical analysis across multiple seeds"""

    def __init__(self, results_dir, seeds):
        self.results_dir = results_dir
        self.seeds = seeds
        self.all_results = self.load_all_results()

    def load_all_results(self):
        all_data = {}
        for seed in self.seeds:
            filepath = os.path.join(self.results_dir, f'results_seed_{seed}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    all_data[seed] = json.load(f)
        return all_data

    def compute_statistics(self, model_name, metric='f1'):
        values = []
        for seed, results in self.all_results.items():
            if model_name in results:
                values.append(results[model_name][metric])
        if len(values) == 0:
            return None, None
        return np.mean(values), np.std(values)

    def paired_ttest(self, model1, model2, metric='f1'):
        values1, values2 = [], []
        for seed, results in self.all_results.items():
            if model1 in results and model2 in results:
                values1.append(results[model1][metric])
                values2.append(results[model2][metric])
        if len(values1) < 2:
            return None, None
        return stats.ttest_rel(values1, values2)

    def generate_summary_table(self):
        models = list(config.MODELS.keys()) + ['soft_voting', 'hard_voting', 'weighted_voting', 'stacking']
        summary = {}
        for model in models:
            summary[model] = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                mean, std = self.compute_statistics(model, metric)
                if mean is not None:
                    summary[model][metric] = {
                        'mean': mean,
                        'std': std,
                        'formatted': f"{mean*100:.2f}±{std*100:.2f}"
                    }
        return summary

    def perform_significance_tests(self):
        ensembles = ['soft_voting', 'hard_voting', 'weighted_voting', 'stacking']
        individuals = list(config.MODELS.keys())

        best_ens = max(ensembles, key=lambda x: self.compute_statistics(x, 'f1')[0] or 0)
        best_ind = max(individuals, key=lambda x: self.compute_statistics(x, 'f1')[0] or 0)

        results = {}
        for model in individuals:
            t_stat, p_val = self.paired_ttest(best_ens, model, 'f1')
            if t_stat is not None:
                results[f'{best_ens}_vs_{model}'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05,
                    'sig_level': '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                }
        return results, best_ens, best_ind

    def plot_performance(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        all_models = list(config.MODELS.keys()) + ['soft_voting', 'hard_voting', 'weighted_voting', 'stacking']

        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            data, labels = [], []
            for model in all_models:
                values = []
                for seed, results in self.all_results.items():
                    if model in results:
                        values.append(results[model][metric] * 100)
                if len(values) > 0:
                    data.append(values)
                    labels.append(model.replace('_', '\\n'))

            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = ['lightblue'] * len(config.MODELS) + ['lightgreen'] * 4
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=12)
            ax.set_title(f'{metric.upper()} Across Seeds', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def export_for_paper(self, output_file):
        summary = self.generate_summary_table()
        sig_results, best_ens, best_ind = self.perform_significance_tests()

        with open(output_file, 'w') as f:
            f.write("="*80 + "\\n")
            f.write("STATISTICAL ANALYSIS - FOR PAPER\\n")
            f.write("="*80 + "\\n\\n")

            f.write("Individual Models:\\n")
            for model in config.MODELS.keys():
                if model in summary:
                    f.write(f"  {model:15s}: F1 = {summary[model]['f1']['formatted']}%\\n")

            f.write("\\nEnsemble Methods:\\n")
            for model in ['soft_voting', 'hard_voting', 'weighted_voting', 'stacking']:
                if model in summary:
                    f.write(f"  {model:15s}: F1 = {summary[model]['f1']['formatted']}%\\n")

print("✓ StatisticalAnalyzer class defined")

class ErrorAnalyzer:
    """Comprehensive error analysis for sentiment classification"""

    def __init__(self, test_df, true_labels, predictions, probs, model_name):
        self.test_df = test_df.reset_index(drop=True)
        self.true_labels = np.array(true_labels)
        self.predictions = np.array(predictions)
        self.probs = np.array(probs)
        self.model_name = model_name

        self.false_positives = (self.predictions == 1) & (self.true_labels == 0)
        self.false_negatives = (self.predictions == 0) & (self.true_labels == 1)
        self.correct = self.predictions == self.true_labels

    def get_error_summary(self):
        total = len(self.true_labels)
        num_fp = np.sum(self.false_positives)
        num_fn = np.sum(self.false_negatives)
        num_correct = np.sum(self.correct)

        return {
            'total_samples': total,
            'correct': num_correct,
            'false_positives': num_fp,
            'false_negatives': num_fn,
            'accuracy': num_correct / total,
            'fp_rate': num_fp / total,
            'fn_rate': num_fn / total
        }

    def get_high_confidence_errors(self, confidence_threshold=0.85):
        pred_confidence = np.max(self.probs, axis=1)
        high_conf_fp = self.false_positives & (pred_confidence >= confidence_threshold)
        high_conf_fn = self.false_negatives & (pred_confidence >= confidence_threshold)

        return {
            'high_conf_fp_indices': np.where(high_conf_fp)[0],
            'high_conf_fn_indices': np.where(high_conf_fn)[0],
            'num_high_conf_fp': np.sum(high_conf_fp),
            'num_high_conf_fn': np.sum(high_conf_fn)
        }

    def analyze_label_quality(self, sample_size=50):
        high_conf = self.get_high_confidence_errors(confidence_threshold=0.85)

        fp_indices = high_conf['high_conf_fp_indices'][:sample_size//2]
        fn_indices = high_conf['high_conf_fn_indices'][:sample_size//2]

        print(f"\\n{'='*80}")
        print("POTENTIAL LABEL NOISE ANALYSIS")
        print(f"{'='*80}\\n")
        print("Examining high-confidence errors (confidence ≥ 0.85)")
        print("These may indicate mislabeled data in the dataset.\\n")

        print("\\nSUSPECTED MISLABELED AS NEGATIVE (Model says POSITIVE with high confidence)")
        for idx in fp_indices[:5]:
            pred_conf = np.max(self.probs[idx])
            review = self.test_df.iloc[idx]['review']
            print(f"Dataset: NEGATIVE | Model: POSITIVE ({pred_conf:.4f})")
            print(f"Review: {review[:300]}...\\n")

        print("\\nSUSPECTED MISLABELED AS POSITIVE (Model says NEGATIVE with high confidence)")
        for idx in fn_indices[:5]:
            pred_conf = np.max(self.probs[idx])
            review = self.test_df.iloc[idx]['review']
            print(f"Dataset: POSITIVE | Model: NEGATIVE ({pred_conf:.4f})")
            print(f"Review: {review[:300]}...\\n")

        return []

    def generate_error_report(self, output_file):
        summary = self.get_error_summary()
        high_conf = self.get_high_confidence_errors()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\\n")
            f.write(f"ERROR ANALYSIS REPORT - {self.model_name}\\n")
            f.write("="*80 + "\\n\\n")

            f.write(f"Total Samples:       {summary['total_samples']:,}\\n")
            f.write(f"Correct:             {summary['correct']:,} ({summary['accuracy']*100:.2f}%)\\n")
            f.write(f"False Positives:     {summary['false_positives']:,} ({summary['fp_rate']*100:.2f}%)\\n")
            f.write(f"False Negatives:     {summary['false_negatives']:,} ({summary['fn_rate']*100:.2f}%)\\n")
            f.write(f"\\nHigh Confidence FP:  {high_conf['num_high_conf_fp']}\\n")
            f.write(f"High Confidence FN:  {high_conf['num_high_conf_fn']}\\n")

print("✓ ErrorAnalyzer class defined")

def run_single_seed(seed):
    """
    Run complete training pipeline for a single seed

    This function:
    1. Trains all 4 models (AraBERT, MARBERT, XLM-RoBERTa, CAMeLBERT)
    2. Tests all 4 ensemble methods (Soft, Hard, Weighted, Stacking)
    3. Saves all results to Google Drive
    4. Tracks runtime automatically
    5. Returns results dictionary

    Args:
        seed (int): Random seed for reproducibility (e.g., 42, 123, 456, 789, 2024)

    Returns:
        dict: Complete results for this seed

    Example:
        results_42 = run_single_seed(42)
    """

    print(f"\n{'#'*70}")
    print(f"RUNNING SEED: {seed}")
    print(f"{'#'*70}\n")

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Track runtime for this seed
    runtime_tracker.start_seed(seed)

    # Load dataset
    print("Loading dataset...")
    data_manager = DataManager(config)
    dataset_path = os.path.join(config.DATA_DIR, config.DATASET_FILE)

    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        print("\nPlease upload 'balanced-reviews.csv' to:")
        print(f"   {config.DATA_DIR}/")
        return None

    df = data_manager.load_hard_dataset(dataset_path)
    if df is None:
        return None

    print(f"✓ Loaded {len(df)} reviews")

    # Split data
    train_df, val_df, test_df = data_manager.split_data(df, seed=seed)

    # Store predictions
    model_predictions = {}
    model_probs = {}
    model_val_accuracies = {}

    # Train individual models
    for model_key, model_name in config.MODELS.items():
        print(f"\n{'='*70}")
        print(f"Training {model_key}: {model_name}")
        print(f"{'='*70}")

        # Initialize model
        classifier = SentimentClassifier(model_name, config, config.DEVICE)

        # Create dataloaders
        train_loader, val_loader, test_loader = data_manager.create_dataloaders(
            train_df, val_df, test_df,
            classifier.tokenizer,
            config.BATCH_SIZE
        )

        # Train
        model_path = os.path.join(config.MODEL_DIR, f'{model_key}_seed{seed}.pt')
        val_acc, train_time = classifier.train(train_loader, val_loader, model_path)

        # Log training time
        runtime_tracker.log_runtime(seed, model_key, train_time)

        # Evaluate
        test_acc, preds, labels, probs = classifier.predict(test_loader)
        metrics = Evaluator.calculate_metrics(labels, preds)
        metrics['training_time'] = train_time
        metrics['val_accuracy'] = val_acc

        # Store
        model_predictions[model_key] = preds
        model_probs[model_key] = probs
        model_val_accuracies[model_key] = val_acc

        Evaluator.print_results(f"{model_key} (Test Set)", metrics)

        # Save individual results for this seed
        result_path = os.path.join(config.RESULTS_DIR, f'{model_key}_results_seed{seed}.json')
        Evaluator.save_results(metrics, result_path)

        # Free memory
        del classifier
        torch.cuda.empty_cache()
        gc.collect()

    # Apply ensemble methods
    print(f"\n{'='*70}")
    print("ENSEMBLE METHODS")
    print(f"{'='*70}")

    ensemble_results = {}
    true_labels = labels

    all_preds = [model_predictions[key] for key in config.MODELS.keys()]
    all_probs = [model_probs[key] for key in config.MODELS.keys()]

    # Soft Voting
    soft_preds, soft_probs = EnsembleMethods.soft_voting(all_probs)
    soft_metrics = Evaluator.calculate_metrics(true_labels, soft_preds)
    Evaluator.print_results("Soft Voting Ensemble", soft_metrics)
    ensemble_results['soft_voting'] = soft_metrics

    # Hard Voting
    hard_preds = EnsembleMethods.hard_voting(all_preds)
    hard_metrics = Evaluator.calculate_metrics(true_labels, hard_preds)
    Evaluator.print_results("Hard Voting Ensemble", hard_metrics)
    ensemble_results['hard_voting'] = hard_metrics

    # Weighted Voting
    weights = [model_val_accuracies[key] for key in config.MODELS.keys()]
    weighted_preds, weighted_probs = EnsembleMethods.weighted_voting(all_probs, weights)
    weighted_metrics = Evaluator.calculate_metrics(true_labels, weighted_preds)
    Evaluator.print_results("Weighted Voting Ensemble", weighted_metrics)
    ensemble_results['weighted_voting'] = weighted_metrics

    # Stacking
    val_probs_all = []
    for model_key in config.MODELS.keys():
        classifier = SentimentClassifier(config.MODELS[model_key], config, config.DEVICE)
        model_path = os.path.join(config.MODEL_DIR, f'{model_key}_seed{seed}.pt')
        classifier.model = torch.load(model_path, weights_only=False)
        classifier.model.to(config.DEVICE)

        _, val_loader, _ = data_manager.create_dataloaders(
            train_df, val_df, test_df,
            classifier.tokenizer,
            config.BATCH_SIZE
        )

        _, _, val_labels, val_probs = classifier.predict(val_loader)
        val_probs_all.append(val_probs)

        del classifier
        torch.cuda.empty_cache()
        gc.collect()

    val_labels = np.array(val_labels)
    stacking_preds, stacking_probs, meta_clf = EnsembleMethods.stacking(
        val_probs_all, val_labels, all_probs
    )
    stacking_metrics = Evaluator.calculate_metrics(true_labels, stacking_preds)
    Evaluator.print_results("Stacking Ensemble", stacking_metrics)
    ensemble_results['stacking'] = stacking_metrics

    # Prepare complete results
    seed_results = {
        'seed': seed,
        'individual_models': {k: Evaluator.calculate_metrics(true_labels, v)
                             for k, v in model_predictions.items()},
        'ensembles': ensemble_results,
        'test_data': {
            'test_df': test_df,
            'true_labels': true_labels,
            'stacking_preds': stacking_preds,
            'stacking_probs': stacking_probs
        }
    }

    # Save complete results for this seed
    seed_result_path = os.path.join(config.RESULTS_DIR, f'results_seed_{seed}.json')
    complete_seed_results = {
        **{k: Evaluator.calculate_metrics(true_labels, v)
           for k, v in model_predictions.items()},
        **ensemble_results
    }
    Evaluator.save_results(complete_seed_results, seed_result_path)

    print(f"\n{'='*70}")
    print(f"✓ SEED {seed} COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ Results saved to: {seed_result_path}")

    # Show runtime for this seed
    seed_time = runtime_tracker.get_seed_total(seed)
    print(f"✓ Seed runtime: {runtime_tracker.format_hours(seed_time):.2f} hours")

    return seed_results

def run_all_analyses():
    """
    Run comprehensive analysis across all completed seeds

    This function:
    1. Loads results from all completed seeds
    2. Performs statistical analysis (mean ± std, t-tests)
    3. Performs error analysis (FP/FN, label noise)
    4. Generates runtime analysis
    5. Creates all tables and figures
    6. Exports everything for paper

    Call this AFTER running all seeds to generate final results.
    """

    print(f"\n{'='*70}")
    print("RUNNING COMPREHENSIVE ANALYSIS")
    print(f"{'='*70}\n")

    # Find which seeds have been completed
    completed_seeds = []
    for seed in config.RANDOM_SEEDS:
        result_file = os.path.join(config.RESULTS_DIR, f'results_seed_{seed}.json')
        if os.path.exists(result_file):
            completed_seeds.append(seed)

    if len(completed_seeds) == 0:
        print("❌ No completed seeds found!")
        print("   Please run at least one seed first.")
        return

    print(f"✓ Found {len(completed_seeds)} completed seeds: {completed_seeds}\n")

    # Runtime Analysis
    print(f"{'='*70}")
    print("RUNTIME ANALYSIS")
    print(f"{'='*70}")

    runtime_tracker.generate_runtime_table()
    runtime_file = os.path.join(config.RESULTS_DIR, 'runtime_analysis.txt')
    runtime_tracker.export_runtime_analysis(runtime_file)
    print(f"\n✓ Runtime analysis saved to: {runtime_file}")

    # Statistical Analysis
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}\n")

    analyzer = StatisticalAnalyzer(config.RESULTS_DIR, completed_seeds)

    if len(analyzer.all_results) > 0:
        summary = analyzer.generate_summary_table()

        print("Individual Models (Mean ± Std):")
        for model in config.MODELS.keys():
            if model in summary and 'f1' in summary[model]:
                print(f"  {model:15s}: F1 = {summary[model]['f1']['formatted']}%")

        print("\nEnsemble Methods (Mean ± Std):")
        for model in ['soft_voting', 'hard_voting', 'weighted_voting', 'stacking']:
            if model in summary and 'f1' in summary[model]:
                print(f"  {model:15s}: F1 = {summary[model]['f1']['formatted']}%")

        sig_results, best_ens, best_ind = analyzer.perform_significance_tests()
        print(f"\n✓ Best Ensemble: {best_ens}")
        print(f"✓ Best Individual: {best_ind}")

        stat_file = os.path.join(config.RESULTS_DIR, 'statistical_analysis.txt')
        analyzer.export_for_paper(stat_file)

        plot_path = os.path.join(config.FIGURES_DIR, 'performance_boxplots.png')
        analyzer.plot_performance(plot_path)

        print(f"\n✓ Statistical analysis saved to: {stat_file}")
        print(f"✓ Box plots saved to: {plot_path}")

    # Error Analysis (using first completed seed)
    first_seed = completed_seeds[0]

    # Try to load test data if available
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS")
    print(f"{'='*70}\n")
    print(f"Note: Using data from seed {first_seed} for error analysis")

    # Load the seed results
    seed_file = os.path.join(config.RESULTS_DIR, f'results_seed_{first_seed}.json')

    # We need to re-run the seed to get test_df and predictions
    # For now, we'll create a placeholder
    print("\nTo run complete error analysis:")
    print("1. The test_df and predictions are saved during seed execution")
    print("2. Error analysis will be performed on the first seed's data")
    print("\nError analysis requires re-running or storing test data.")
    print("See error_analysis_manual.txt for instructions on manual analysis.")

    print(f"\n{'='*70}")
    print("✓ ALL ANALYSES COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - results_seed_*.json (for each completed seed)")
    print("  - statistical_analysis.txt (mean±std, t-tests, LaTeX)")
    print("  - runtime_analysis.txt (training times, LaTeX)")
    print("  - performance_boxplots.png (visualizations)")

# Initialize runtime tracker
runtime_tracker.start_experiment()

print("✓ run_single_seed() function defined")
print("✓ run_all_analyses() function defined")
print("✓ Runtime tracker initialized")
print("\nYou can now run seeds individually:")
print("  results_42 = run_single_seed(42)")
print("  results_123 = run_single_seed(123)")
print("  ... etc")
print("\nAfter running all seeds, call:")
print("  run_all_analyses()")

# =====================================
# RUN SEED 42
# =====================================
# Estimated time: 3-4 hours on A100
# This cell trains all 4 models and tests all ensembles
# Results automatically saved to Google Drive

results_seed_42 = run_single_seed(42)

# =====================================
# RUN SEED 123
# =====================================
# Estimated time: 3-4 hours on A100
# Run this AFTER seed 42 completes

results_seed_123 = run_single_seed(123)

# =====================================
# RUN SEED 456
# =====================================
# Estimated time: 3-4 hours on A100
# Run this AFTER seed 123 completes

results_seed_456 = run_single_seed(456)

# =====================================
# RUN SEED 789
# =====================================
# Estimated time: 3-4 hours on A100
# Run this AFTER seed 456 completes

results_seed_789 = run_single_seed(789)

# =====================================
# RUN SEED 2024
# =====================================
# Estimated time: 3-4 hours on A100
# Run this AFTER seed 789 completes

results_seed_2024 = run_single_seed(2024)

# =====================================
# RUN COMPREHENSIVE ANALYSIS
# =====================================
# Run this AFTER all seeds complete
# Works even if you only ran some seeds
#
# Generates:
#   - statistical_analysis.txt
#   - runtime_analysis.txt
#   - performance_boxplots.png

run_all_analyses()

# ============================================================================
# ERROR ANALYSIS - MATCHED TO YOUR ErrorAnalyzer CLASS
# ============================================================================
# Run this cell AFTER all training completes
# This version is built specifically for YOUR ErrorAnalyzer implementation
# ============================================================================

print("\n" + "="*70)
print("PERFORMING ERROR ANALYSIS ON TRAINED MODELS")
print("="*70)

# Configuration
seed = 42
error_results_dir = os.path.join(config.RESULTS_DIR, 'error_analysis')
os.makedirs(error_results_dir, exist_ok=True)

# Load test data
print("\nLoading test data...")
data_manager = DataManager(config)
dataset_path = os.path.join(config.DATA_DIR, config.DATASET_FILE)
df = data_manager.load_hard_dataset(dataset_path)
train_df, val_df, test_df = data_manager.split_data(df, seed)

print(f"Test set size: {len(test_df):,} samples")

# Analyze each model
for model_key, model_path_name in config.MODELS.items():
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {model_key}")
    print(f"{'='*70}")

    # Load trained model
    model_file = os.path.join(config.MODEL_DIR, f'{model_key}_seed{seed}.pt')

    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        continue

    print(f"Loading model from: {model_file}")
    model = torch.load(model_file, weights_only=False)
    model.to(config.DEVICE)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path_name)

    # Create test dataset and loader
    test_dataset = HARDDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        config.MAX_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Get predictions with probabilities
    print("Getting predictions...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Analyzing {model_key}"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Create ErrorAnalyzer instance
    error_analyzer = ErrorAnalyzer(
        test_df=test_df,
        true_labels=all_labels,
        predictions=all_preds,
        probs=all_probs,
        model_name=model_key
    )

    # Get error summary
    error_summary = error_analyzer.get_error_summary()

    print(f"\n📊 Error Summary:")
    print(f"  Total samples: {error_summary['total_samples']}")
    print(f"  Correct: {error_summary['correct']} ({error_summary['accuracy']*100:.2f}%)")
    print(f"  False Positives: {error_summary['false_positives']} ({error_summary['fp_rate']*100:.2f}%)")
    print(f"  False Negatives: {error_summary['false_negatives']} ({error_summary['fn_rate']*100:.2f}%)")

    # ========================================================================
    # MANUAL DATAFRAME CREATION (Your ErrorAnalyzer doesn't have get_error_df)
    # ========================================================================

    # Get high confidence errors
    high_conf = error_analyzer.get_high_confidence_errors(confidence_threshold=0.8)

    # False Positives
    fp_mask = error_analyzer.false_positives
    num_fp = np.sum(fp_mask)

    if num_fp > 0:
        fp_indices = np.where(fp_mask)[0]

        # Create FP DataFrame
        fp_data = []
        for idx in fp_indices:
            # Get the text - check available columns
            if 'text' in test_df.columns:
                text = test_df.iloc[idx]['text']
            elif 'review' in test_df.columns:
                text = test_df.iloc[idx]['review']
            else:
                text = str(test_df.iloc[idx].values[0])  # Fallback

            fp_data.append({
                'text': text,
                'true_label': int(all_labels[idx]),
                'predicted_label': int(all_preds[idx]),
                'probability': float(all_probs[idx, 1]),  # Prob of positive class
                'confidence': 'High' if all_probs[idx, 1] > 0.8 else ('Medium' if all_probs[idx, 1] > 0.6 else 'Low')
            })

        fp_df = pd.DataFrame(fp_data)

        # Save all false positives
        fp_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_false_positives.csv")
        fp_df.to_csv(fp_path, index=False, encoding='utf-8')
        print(f"\n✓ Saved: {fp_path}")
        print(f"  False Positives: {len(fp_df)}")

        # High confidence false positives
        high_conf_fp = fp_df[fp_df['probability'] > 0.8]
        if len(high_conf_fp) > 0:
            high_fp_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_HIGH_CONFIDENCE_FP.csv")
            high_conf_fp.to_csv(high_fp_path, index=False, encoding='utf-8')
            print(f"✓ Saved: {high_fp_path}")
            print(f"  High Confidence FP: {len(high_conf_fp)}")
    else:
        print(f"\n  No false positives found")

    # False Negatives
    fn_mask = error_analyzer.false_negatives
    num_fn = np.sum(fn_mask)

    if num_fn > 0:
        fn_indices = np.where(fn_mask)[0]

        # Create FN DataFrame
        fn_data = []
        for idx in fn_indices:
            # Get the text - check available columns
            if 'text' in test_df.columns:
                text = test_df.iloc[idx]['text']
            elif 'review' in test_df.columns:
                text = test_df.iloc[idx]['review']
            else:
                text = str(test_df.iloc[idx].values[0])  # Fallback

            fn_data.append({
                'text': text,
                'true_label': int(all_labels[idx]),
                'predicted_label': int(all_preds[idx]),
                'probability': float(all_probs[idx, 0]),  # Prob of negative class
                'confidence': 'High' if all_probs[idx, 0] > 0.8 else ('Medium' if all_probs[idx, 0] > 0.6 else 'Low')
            })

        fn_df = pd.DataFrame(fn_data)

        # Save all false negatives
        fn_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_false_negatives.csv")
        fn_df.to_csv(fn_path, index=False, encoding='utf-8')
        print(f"✓ Saved: {fn_path}")
        print(f"  False Negatives: {len(fn_df)}")

        # High confidence false negatives
        high_conf_fn = fn_df[fn_df['probability'] > 0.8]
        if len(high_conf_fn) > 0:
            high_fn_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_HIGH_CONFIDENCE_FN.csv")
            high_conf_fn.to_csv(high_fn_path, index=False, encoding='utf-8')
            print(f"✓ Saved: {high_fn_path}")
            print(f"  High Confidence FN: {len(high_conf_fn)}")
    else:
        print(f"  No false negatives found")

    # Save error summary as JSON
    summary_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_error_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types for JSON serialization
        summary_json = {
            'total_samples': int(error_summary['total_samples']),
            'correct': int(error_summary['correct']),
            'false_positives': int(error_summary['false_positives']),
            'false_negatives': int(error_summary['false_negatives']),
            'accuracy': float(error_summary['accuracy']),
            'fp_rate': float(error_summary['fp_rate']),
            'fn_rate': float(error_summary['fn_rate']),
            'high_conf_fp': int(high_conf['num_high_conf_fp']),
            'high_conf_fn': int(high_conf['num_high_conf_fn'])
        }
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {summary_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    print(f"\n✓ Error analysis complete for {model_key}")

print("\n" + "="*70)
print("✓ ERROR ANALYSIS COMPLETE FOR ALL MODELS")
print("="*70)
print(f"\nResults saved to: {error_results_dir}")


# ============================================================================
# AGGREGATE ERROR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("AGGREGATE ERROR ANALYSIS")
print("="*70)

# Collect summaries
all_summaries = {}
for model_key in config.MODELS.keys():
    summary_path = os.path.join(error_results_dir, f"{model_key}_seed{seed}_error_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            all_summaries[model_key] = json.load(f)

if len(all_summaries) > 0:
    print("\n📊 Error Comparison Across Models:")
    print(f"{'Model':<15} {'Accuracy':<12} {'FP':<8} {'FN':<8} {'Total Errors':<12}")
    print("-" * 70)

    for model_key, summary in all_summaries.items():
        accuracy = summary['accuracy'] * 100
        fp = summary['false_positives']
        fn = summary['false_negatives']
        total_errors = fp + fn

        print(f"{model_key:<15} {accuracy:>6.2f}%     {fp:>4d}    {fn:>4d}    {total_errors:>4d}")

    print("\n📊 High Confidence Errors:")
    print(f"{'Model':<15} {'High-Conf FP':<15} {'High-Conf FN':<15}")
    print("-" * 50)

    for model_key, summary in all_summaries.items():
        hc_fp = summary.get('high_conf_fp', 0)
        hc_fn = summary.get('high_conf_fn', 0)
        print(f"{model_key:<15} {hc_fp:<15d} {hc_fn:<15d}")

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
else:
    print("\n❌ No summaries found!")

# =====================================
# CHECK WHICH SEEDS ARE COMPLETED
# =====================================
# Run this anytime to see progress

import os

print("Checking completed seeds...\n")

for seed in [42, 123, 456, 789, 2024]:
    result_file = os.path.join(config.RESULTS_DIR, f'results_seed_{seed}.json')
    if os.path.exists(result_file):
        print(f"✓ Seed {seed} - COMPLETED")

        # Show file size and modification time
        size_kb = os.path.getsize(result_file) / 1024
        from datetime import datetime
        mod_time = datetime.fromtimestamp(os.path.getmtime(result_file))
        print(f"  Size: {size_kb:.1f} KB")
        print(f"  Completed: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"✗ Seed {seed} - NOT STARTED")
    print()

# Count completed
completed = sum(1 for seed in [42, 123, 456, 789, 2024]
                if os.path.exists(os.path.join(config.RESULTS_DIR, f'results_seed_{seed}.json')))

print(f"\nProgress: {completed}/5 seeds completed ({completed*20}%)")

if completed == 5:
    print("\n🎉 All seeds complete! Run the analysis cell above.")
elif completed >= 3:
    print(f"\n✓ {completed} seeds complete. You can run analysis now or wait for more seeds.")
elif completed > 0:
    print(f"\n⏳ {completed} seed(s) complete. Continue with remaining seeds.")
else:
    print("\n⚠️  No seeds completed yet. Start with Seed 42 above.")