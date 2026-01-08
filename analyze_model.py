#!/usr/bin/env python3
"""
Comprehensive Model Analysis for HN Success Predictor

This script analyzes the trained model to identify:
1. Where the model fails (error analysis)
2. What patterns it learned (feature importance)
3. How to improve it (actionable recommendations)

Run: python analyze_model.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
import re

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = Path("collab_download/hn_predictor")
DB_PATH = Path("hn_training_data.db")
OUTPUT_DIR = Path("model_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


class ModelAnalyzer:
    def __init__(self, model_path: Path):
        print("Loading model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
    
    def predict(self, titles: list) -> np.ndarray:
        """Predict success probability for titles."""
        inputs = self.tokenizer(
            titles, padding=True, truncation=True, 
            max_length=64, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        return probs
    
    def predict_batch(self, titles: list, batch_size: int = 64) -> np.ndarray:
        """Predict in batches for memory efficiency."""
        all_probs = []
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i+batch_size]
            probs = self.predict(batch)
            all_probs.extend(probs)
        return np.array(all_probs)


def load_test_data(db_path: Path, test_size: int = 3000) -> pd.DataFrame:
    """Load test data from the training database."""
    print(f"Loading test data from {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Get balanced sample
    df = pd.read_sql_query("""
        SELECT id, title, url, domain, points, category
        FROM posts
        WHERE title IS NOT NULL AND title != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, conn, params=(test_size * 2,))
    conn.close()
    
    df['label'] = (df['points'] >= 100).astype(int)
    
    # Balance the dataset
    hits = df[df['label'] == 1].head(test_size // 2)
    misses = df[df['label'] == 0].head(test_size // 2)
    df = pd.concat([hits, misses]).sample(frac=1, random_state=42)
    
    print(f"Loaded {len(df)} test samples ({df['label'].mean():.1%} hits)")
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for error analysis."""
    df = df.copy()
    
    # Title features
    df['title_length'] = df['title'].str.len()
    df['word_count'] = df['title'].str.split().str.len()
    df['has_question'] = df['title'].str.contains(r'\?', regex=True).astype(int)
    df['has_number'] = df['title'].str.contains(r'\d', regex=True).astype(int)
    df['has_year'] = df['title'].str.contains(r'\b(19|20)\d{2}\b', regex=True).astype(int)
    df['is_show_hn'] = df['title'].str.lower().str.startswith('show hn').astype(int)
    df['is_ask_hn'] = df['title'].str.lower().str.startswith('ask hn').astype(int)
    df['has_pdf'] = df['title'].str.lower().str.contains(r'\[pdf\]').astype(int)
    df['has_video'] = df['title'].str.lower().str.contains(r'\[video\]|video').astype(int)
    
    # Domain features
    df['is_github'] = df['domain'].str.contains('github', case=False, na=False).astype(int)
    df['is_self_post'] = (df['domain'] == 'self.hackernews').astype(int)
    
    return df


def analyze_errors(df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Analyze prediction errors."""
    df = df.copy()
    df['predicted'] = (df['prob'] >= threshold).astype(int)
    df['correct'] = (df['predicted'] == df['label']).astype(int)
    df['error_type'] = 'correct'
    df.loc[(df['predicted'] == 1) & (df['label'] == 0), 'error_type'] = 'false_positive'
    df.loc[(df['predicted'] == 0) & (df['label'] == 1), 'error_type'] = 'false_negative'
    
    results = {
        'total': len(df),
        'correct': df['correct'].sum(),
        'accuracy': df['correct'].mean(),
        'false_positives': (df['error_type'] == 'false_positive').sum(),
        'false_negatives': (df['error_type'] == 'false_negative').sum(),
    }
    
    # Error rate by feature
    feature_errors = {}
    for feature in ['is_show_hn', 'is_ask_hn', 'is_github', 'is_self_post', 
                    'has_question', 'has_pdf', 'has_number']:
        if feature in df.columns:
            mask = df[feature] == 1
            if mask.sum() > 10:
                feature_errors[feature] = {
                    'count': mask.sum(),
                    'accuracy': df.loc[mask, 'correct'].mean(),
                    'fp_rate': (df.loc[mask, 'error_type'] == 'false_positive').mean(),
                    'fn_rate': (df.loc[mask, 'error_type'] == 'false_negative').mean(),
                }
    
    results['by_feature'] = feature_errors
    
    # Worst false positives (model confident but wrong)
    fps = df[(df['error_type'] == 'false_positive')].nlargest(20, 'prob')
    results['worst_false_positives'] = fps[['title', 'prob', 'points', 'domain']].to_dict('records')
    
    # Worst false negatives (model missed these hits)
    fns = df[(df['error_type'] == 'false_negative')].nsmallest(20, 'prob')
    results['worst_false_negatives'] = fns[['title', 'prob', 'points', 'domain']].to_dict('records')
    
    return results


def analyze_by_segment(df: pd.DataFrame) -> dict:
    """Analyze performance by different segments."""
    results = {}
    
    # By title length
    df['length_bucket'] = pd.cut(df['title_length'], 
                                  bins=[0, 30, 50, 70, 100, 200],
                                  labels=['<30', '30-50', '50-70', '70-100', '100+'])
    
    length_perf = df.groupby('length_bucket').apply(
        lambda x: pd.Series({
            'count': len(x),
            'actual_hit_rate': x['label'].mean(),
            'predicted_hit_rate': x['prob'].mean(),
            'auc': roc_auc_score(x['label'], x['prob']) if len(x['label'].unique()) > 1 else 0.5
        })
    )
    results['by_title_length'] = length_perf.to_dict()
    
    # By domain (top domains)
    top_domains = df['domain'].value_counts().head(20).index
    domain_perf = {}
    for domain in top_domains:
        mask = df['domain'] == domain
        if mask.sum() >= 10:
            subset = df[mask]
            domain_perf[domain] = {
                'count': len(subset),
                'actual_hit_rate': subset['label'].mean(),
                'predicted_hit_rate': subset['prob'].mean(),
                'auc': roc_auc_score(subset['label'], subset['prob']) if len(subset['label'].unique()) > 1 else 0.5
            }
    results['by_domain'] = domain_perf
    
    # By post type
    post_types = {
        'Show HN': df['is_show_hn'] == 1,
        'Ask HN': df['is_ask_hn'] == 1,
        'GitHub': df['is_github'] == 1,
        'PDF': df['has_pdf'] == 1,
        'Regular': (df['is_show_hn'] == 0) & (df['is_ask_hn'] == 0) & (df['has_pdf'] == 0)
    }
    
    type_perf = {}
    for name, mask in post_types.items():
        if mask.sum() >= 10:
            subset = df[mask]
            type_perf[name] = {
                'count': len(subset),
                'actual_hit_rate': subset['label'].mean(),
                'predicted_hit_rate': subset['prob'].mean(),
                'auc': roc_auc_score(subset['label'], subset['prob']) if len(subset['label'].unique()) > 1 else 0.5
            }
    results['by_post_type'] = type_perf
    
    # By points bucket (for calibration analysis)
    df['points_bucket'] = pd.cut(df['points'],
                                  bins=[0, 10, 25, 50, 100, 200, 500, 10000],
                                  labels=['1-10', '11-25', '26-50', '51-100', '101-200', '201-500', '500+'])
    
    points_perf = df.groupby('points_bucket').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mean_prob': x['prob'].mean(),
            'std_prob': x['prob'].std(),
        })
    )
    results['by_points'] = points_perf.to_dict()
    
    return results


def analyze_calibration(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Analyze prediction calibration."""
    df = df.copy()
    df['prob_bucket'] = pd.cut(df['prob'], bins=n_bins, labels=False)
    
    calibration = df.groupby('prob_bucket').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mean_predicted': x['prob'].mean(),
            'mean_actual': x['label'].mean(),
            'calibration_error': abs(x['prob'].mean() - x['label'].mean())
        })
    )
    
    # Expected Calibration Error
    total = len(df)
    ece = sum(calibration['count'] * calibration['calibration_error']) / total
    
    return {
        'ece': ece,
        'bins': calibration.to_dict(),
        'is_well_calibrated': ece < 0.1
    }


def analyze_confidence(df: pd.DataFrame) -> dict:
    """Analyze model confidence patterns."""
    df = df.copy()
    
    # Confidence distribution
    high_conf_correct = ((df['prob'] > 0.8) | (df['prob'] < 0.2)) & (df['label'] == (df['prob'] > 0.5).astype(int))
    high_conf_wrong = ((df['prob'] > 0.8) | (df['prob'] < 0.2)) & (df['label'] != (df['prob'] > 0.5).astype(int))
    
    results = {
        'high_confidence_rate': ((df['prob'] > 0.8) | (df['prob'] < 0.2)).mean(),
        'high_conf_accuracy': high_conf_correct.sum() / (high_conf_correct.sum() + high_conf_wrong.sum()) if (high_conf_correct.sum() + high_conf_wrong.sum()) > 0 else 0,
        'mean_confidence_when_correct': df.loc[(df['prob'] > 0.5) == df['label'].astype(bool), 'prob'].apply(lambda x: max(x, 1-x)).mean(),
        'mean_confidence_when_wrong': df.loc[(df['prob'] > 0.5) != df['label'].astype(bool), 'prob'].apply(lambda x: max(x, 1-x)).mean(),
    }
    
    # Uncertain predictions (40-60% probability)
    uncertain = (df['prob'] >= 0.4) & (df['prob'] <= 0.6)
    results['uncertain_rate'] = uncertain.mean()
    results['uncertain_accuracy'] = accuracy_score(
        df.loc[uncertain, 'label'], 
        (df.loc[uncertain, 'prob'] >= 0.5).astype(int)
    ) if uncertain.sum() > 0 else 0
    
    return results


def generate_word_analysis(df: pd.DataFrame) -> dict:
    """Analyze which words correlate with hits/misses and model errors."""
    from collections import defaultdict
    
    # Words in titles
    def get_words(title):
        return set(re.findall(r'\b[a-z]{3,}\b', title.lower()))
    
    word_stats = defaultdict(lambda: {'hit': 0, 'miss': 0, 'fp': 0, 'fn': 0})
    
    df = df.copy()
    df['predicted'] = (df['prob'] >= 0.5).astype(int)
    
    for _, row in df.iterrows():
        words = get_words(row['title'])
        for word in words:
            if row['label'] == 1:
                word_stats[word]['hit'] += 1
            else:
                word_stats[word]['miss'] += 1
            
            if row['predicted'] == 1 and row['label'] == 0:
                word_stats[word]['fp'] += 1
            elif row['predicted'] == 0 and row['label'] == 1:
                word_stats[word]['fn'] += 1
    
    # Convert to DataFrame for analysis
    word_df = pd.DataFrame([
        {'word': w, **stats} 
        for w, stats in word_stats.items() 
        if stats['hit'] + stats['miss'] >= 10
    ])
    
    if len(word_df) == 0:
        return {}
    
    word_df['total'] = word_df['hit'] + word_df['miss']
    word_df['hit_rate'] = word_df['hit'] / word_df['total']
    word_df['fp_rate'] = word_df['fp'] / word_df['total']
    word_df['fn_rate'] = word_df['fn'] / word_df['total']
    
    return {
        'high_hit_words': word_df.nlargest(20, 'hit_rate')[['word', 'total', 'hit_rate']].to_dict('records'),
        'low_hit_words': word_df.nsmallest(20, 'hit_rate')[['word', 'total', 'hit_rate']].to_dict('records'),
        'high_fp_words': word_df.nlargest(20, 'fp_rate')[['word', 'total', 'fp_rate']].to_dict('records'),
        'high_fn_words': word_df.nlargest(20, 'fn_rate')[['word', 'total', 'fn_rate']].to_dict('records'),
    }


def plot_analysis(df: pd.DataFrame, output_dir: Path):
    """Generate analysis plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Score distribution by actual class
    ax = axes[0, 0]
    ax.hist(df[df['label']==0]['prob'], bins=50, alpha=0.6, label='Not Hit', density=True)
    ax.hist(df[df['label']==1]['prob'], bins=50, alpha=0.6, label='Hit', density=True)
    ax.axvline(0.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend()
    
    # 2. Calibration plot
    ax = axes[0, 1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(df['prob'], bin_edges[1:-1])
    
    cal_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            cal_data.append((df.loc[mask, 'prob'].mean(), df.loc[mask, 'label'].mean()))
    
    if cal_data:
        pred_probs, actual_probs = zip(*cal_data)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
        ax.scatter(pred_probs, actual_probs, s=100)
        ax.plot(pred_probs, actual_probs, 'o-', label='Model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Actual Hit Rate')
    ax.set_title('Calibration Plot')
    ax.legend()
    
    # 3. ROC Curve
    ax = axes[0, 2]
    fpr, tpr, _ = roc_curve(df['label'], df['prob'])
    auc = roc_auc_score(df['label'], df['prob'])
    ax.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    
    # 4. Performance by title length
    ax = axes[1, 0]
    length_groups = df.groupby(pd.cut(df['title_length'], bins=5))
    lengths = [str(g) for g in length_groups.groups.keys()]
    aucs = [roc_auc_score(g['label'], g['prob']) if len(g['label'].unique()) > 1 else 0.5 
            for _, g in length_groups]
    ax.bar(range(len(lengths)), aucs)
    ax.set_xticks(range(len(lengths)))
    ax.set_xticklabels(lengths, rotation=45)
    ax.set_ylabel('ROC AUC')
    ax.set_title('Performance by Title Length')
    ax.axhline(auc, color='red', linestyle='--', label=f'Overall: {auc:.3f}')
    ax.legend()
    
    # 5. Precision-Recall tradeoff
    ax = axes[1, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = (df['prob'] >= t).astype(int)
        precisions.append(precision_score(df['label'], pred, zero_division=0))
        recalls.append(recall_score(df['label'], pred, zero_division=0))
        f1s.append(f1_score(df['label'], pred, zero_division=0))
    
    ax.plot(thresholds, precisions, label='Precision')
    ax.plot(thresholds, recalls, label='Recall')
    ax.plot(thresholds, f1s, label='F1', linewidth=2)
    ax.axvline(thresholds[np.argmax(f1s)], color='gray', linestyle=':', label=f'Best F1 @ {thresholds[np.argmax(f1s)]:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend()
    
    # 6. Error distribution by points
    ax = axes[1, 2]
    df_copy = df.copy()
    df_copy['predicted'] = (df_copy['prob'] >= 0.5).astype(int)
    df_copy['error'] = df_copy['predicted'] != df_copy['label']
    
    points_groups = pd.cut(df['points'], bins=[0, 25, 50, 100, 200, 500, 5000])
    error_rates = df_copy.groupby(points_groups)['error'].mean()
    
    ax.bar(range(len(error_rates)), error_rates.values)
    ax.set_xticks(range(len(error_rates)))
    ax.set_xticklabels([str(x) for x in error_rates.index], rotation=45)
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Points Range')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'analysis_plots.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'analysis_plots.png'}")


def print_recommendations(analysis: dict):
    """Print actionable recommendations based on analysis."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("=" * 70)
    
    recommendations = []
    
    # Check calibration
    if 'calibration' in analysis:
        ece = analysis['calibration']['ece']
        if ece > 0.1:
            recommendations.append(f"❗ Model is poorly calibrated (ECE={ece:.3f}). Consider temperature scaling or Platt scaling.")
    
    # Check confidence
    if 'confidence' in analysis:
        conf = analysis['confidence']
        if conf['uncertain_rate'] > 0.3:
            recommendations.append(f"❗ {conf['uncertain_rate']:.1%} of predictions are uncertain (40-60%). Model needs more discriminative features.")
        if conf['high_conf_accuracy'] < 0.9:
            recommendations.append(f"❗ High confidence predictions only {conf['high_conf_accuracy']:.1%} accurate. Model is overconfident.")
    
    # Check segment performance
    if 'segments' in analysis:
        by_type = analysis['segments'].get('by_post_type', {})
        for name, stats in by_type.items():
            if stats.get('auc', 0.5) < 0.6:
                recommendations.append(f"❗ Poor performance on '{name}' posts (AUC={stats['auc']:.3f}). Consider type-specific features.")
    
    # Check errors
    if 'errors' in analysis:
        errors = analysis['errors']
        fp_rate = errors['false_positives'] / errors['total']
        fn_rate = errors['false_negatives'] / errors['total']
        
        if fp_rate > fn_rate * 1.5:
            recommendations.append(f"❗ High false positive rate ({fp_rate:.1%}). Model is too optimistic. Consider raising threshold or using class weights.")
        elif fn_rate > fp_rate * 1.5:
            recommendations.append(f"❗ High false negative rate ({fn_rate:.1%}). Model is too conservative. Consider lowering threshold.")
    
    # General recommendations
    recommendations.extend([
        "✅ Add domain as a feature (some domains consistently perform better)",
        "✅ Consider time-of-day features (posts submitted at optimal times do better)",
        "✅ Try ensemble with TF-IDF model (captures different patterns)",
        "✅ Experiment with RoBERTa or DeBERTa for better semantic understanding",
        "✅ Add title-level features: sentiment, reading level, clickbait score",
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")


def main():
    print("=" * 70)
    print("HN SUCCESS PREDICTOR - COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 70)
    
    # Load model and data
    analyzer = ModelAnalyzer(MODEL_PATH)
    df = load_test_data(DB_PATH)
    
    # Extract features
    print("\nExtracting features...")
    df = extract_features(df)
    
    # Get predictions
    print("\nGenerating predictions...")
    df['prob'] = analyzer.predict_batch(df['title'].tolist())
    
    # Calculate overall metrics
    print("\n" + "=" * 70)
    print("OVERALL METRICS")
    print("=" * 70)
    
    y_true = df['label'].values
    y_prob = df['prob'].values
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"ROC AUC:     {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision:   {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:      {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:    {f1_score(y_true, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    
    # Run all analyses
    analysis = {}
    
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    analysis['errors'] = analyze_errors(df)
    
    print(f"\nFalse Positives: {analysis['errors']['false_positives']} ({analysis['errors']['false_positives']/len(df):.1%})")
    print(f"False Negatives: {analysis['errors']['false_negatives']} ({analysis['errors']['false_negatives']/len(df):.1%})")
    
    print("\nPerformance by Feature:")
    for feat, stats in analysis['errors'].get('by_feature', {}).items():
        print(f"  {feat:15} n={stats['count']:4d}  acc={stats['accuracy']:.2f}  fp_rate={stats['fp_rate']:.2f}  fn_rate={stats['fn_rate']:.2f}")
    
    print("\nWorst False Positives (model said HIT but wasn't):")
    for i, fp in enumerate(analysis['errors']['worst_false_positives'][:5], 1):
        print(f"  {i}. [{fp['prob']:.0%}] {fp['points']}pts: {fp['title'][:60]}")
    
    print("\nWorst False Negatives (model missed these HITS):")
    for i, fn in enumerate(analysis['errors']['worst_false_negatives'][:5], 1):
        print(f"  {i}. [{fn['prob']:.0%}] {fn['points']}pts: {fn['title'][:60]}")
    
    print("\n" + "=" * 70)
    print("SEGMENT ANALYSIS")
    print("=" * 70)
    analysis['segments'] = analyze_by_segment(df)
    
    print("\nPerformance by Post Type:")
    for ptype, stats in analysis['segments'].get('by_post_type', {}).items():
        print(f"  {ptype:12} n={stats['count']:4d}  actual={stats['actual_hit_rate']:.2f}  pred={stats['predicted_hit_rate']:.2f}  auc={stats['auc']:.3f}")
    
    print("\nPerformance by Top Domains:")
    for domain, stats in list(analysis['segments'].get('by_domain', {}).items())[:10]:
        print(f"  {domain[:20]:20} n={stats['count']:3d}  actual={stats['actual_hit_rate']:.2f}  pred={stats['predicted_hit_rate']:.2f}  auc={stats['auc']:.3f}")
    
    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS")
    print("=" * 70)
    analysis['calibration'] = analyze_calibration(df)
    print(f"Expected Calibration Error (ECE): {analysis['calibration']['ece']:.4f}")
    print(f"Well Calibrated: {'Yes' if analysis['calibration']['is_well_calibrated'] else 'No'}")
    
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)
    analysis['confidence'] = analyze_confidence(df)
    print(f"High Confidence Rate: {analysis['confidence']['high_confidence_rate']:.1%}")
    print(f"High Confidence Accuracy: {analysis['confidence']['high_conf_accuracy']:.1%}")
    print(f"Uncertain Predictions: {analysis['confidence']['uncertain_rate']:.1%}")
    print(f"Uncertain Accuracy: {analysis['confidence']['uncertain_accuracy']:.1%}")
    
    print("\n" + "=" * 70)
    print("WORD ANALYSIS")
    print("=" * 70)
    analysis['words'] = generate_word_analysis(df)
    
    if analysis['words']:
        print("\nWords with HIGHEST hit rate:")
        for w in analysis['words'].get('high_hit_words', [])[:10]:
            print(f"  {w['word']:20} n={w['total']:3d}  hit_rate={w['hit_rate']:.2f}")
        
        print("\nWords with LOWEST hit rate:")
        for w in analysis['words'].get('low_hit_words', [])[:10]:
            print(f"  {w['word']:20} n={w['total']:3d}  hit_rate={w['hit_rate']:.2f}")
        
        print("\nWords causing FALSE POSITIVES:")
        for w in analysis['words'].get('high_fp_words', [])[:10]:
            print(f"  {w['word']:20} n={w['total']:3d}  fp_rate={w['fp_rate']:.2f}")
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_analysis(df, OUTPUT_DIR)
    
    # Print recommendations
    print_recommendations(analysis)
    
    # Save full analysis
    import json
    
    # Convert non-serializable objects
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Interval):
            return str(obj)
        return obj
    
    analysis_json = json.loads(json.dumps(analysis, default=convert))
    
    with open(OUTPUT_DIR / 'analysis.json', 'w') as f:
        json.dump(analysis_json, f, indent=2, default=str)
    print(f"\nFull analysis saved to: {OUTPUT_DIR / 'analysis.json'}")
    
    # Save predictions for further analysis
    df[['id', 'title', 'domain', 'points', 'label', 'prob', 'category']].to_csv(
        OUTPUT_DIR / 'predictions.csv', index=False
    )
    print(f"Predictions saved to: {OUTPUT_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
