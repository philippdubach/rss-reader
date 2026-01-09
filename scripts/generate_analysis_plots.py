#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for HN Success Predictor model.
Creates a multi-panel figure with 2 columns (wider plots) and as many rows as needed.

Style matches the reference image with orange/blue color scheme.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "archive/model_analysis/predictions.csv")
ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "archive/model_analysis/analysis.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "rss_reader/models/hn_model_v32/config.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "docs/model_analysis_plots.png")

# Color palette - matching reference image
COLORS = {
    'hit': '#ff7f0e',       # Orange for positive class (Hit)
    'not_hit': '#1f77b4',   # Blue for negative class (Not Hit)
    'primary': '#1f77b4',   # Blue
    'secondary': '#2ca02c', # Green
    'accent': '#d62728',    # Red for thresholds/highlights
    'neutral': '#7f7f7f',   # Gray
    'precision': '#1f77b4', # Blue
    'recall': '#ff7f0e',    # Orange
    'f1': '#2ca02c',        # Green
}


def load_data():
    """Load predictions, analysis, and config."""
    df = pd.read_csv(PREDICTIONS_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    with open(ANALYSIS_PATH) as f:
        analysis = json.load(f)
    return df, config, analysis


def setup_style():
    """Setup matplotlib style for academic-quality plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_score_distribution(ax, df, threshold):
    """Plot score distribution by class."""
    y_true = df['label'].values
    y_prob = df['prob'].values
    
    hits = y_prob[y_true == 1]
    non_hits = y_prob[y_true == 0]
    
    bins = np.linspace(0, 1, 40)
    ax.hist(non_hits, bins=bins, alpha=0.7, label='Not Hit', 
            color=COLORS['not_hit'], density=True, edgecolor='white', linewidth=0.3)
    ax.hist(hits, bins=bins, alpha=0.7, label='Hit', 
            color=COLORS['hit'], density=True, edgecolor='white', linewidth=0.3)
    
    ax.axvline(x=threshold, color=COLORS['accent'], linestyle='--', lw=2, 
               label='Threshold')
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])


def plot_calibration(ax, df, config):
    """Plot calibration curve (reliability diagram)."""
    y_true = df['label'].values
    y_prob = df['prob'].values
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--', lw=1, label='Perfect')
    ax.plot(prob_pred, prob_true, color=COLORS['primary'], lw=2, marker='o', 
            markersize=6, label='Model')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Actual Hit Rate')
    ax.set_title('Calibration Plot')
    ax.legend(loc='upper left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_aspect('equal')


def plot_roc_curve(ax, df):
    """Plot ROC curve."""
    y_true = df['label'].values
    y_prob = df['prob'].values
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color=COLORS['primary'], lw=2, 
            label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS['primary'])
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_aspect('equal')


def plot_performance_by_title_length(ax, analysis):
    """Plot AUC by title length bins."""
    title_data = analysis['segments']['by_title_length']
    
    # Get bins and AUCs
    bins = list(title_data['auc'].keys())
    aucs = list(title_data['auc'].values())
    counts = list(title_data['count'].values())
    
    # Filter out NaN values and 100+ bin (no data)
    valid_data = [(b, a, c) for b, a, c in zip(bins, aucs, counts) if not np.isnan(a) and c > 0]
    bins, aucs, counts = zip(*valid_data) if valid_data else ([], [], [])
    
    # Create labels matching reference
    labels = ['(2.915, 20.0]', '(20.0, 37.0]', '(37.0, 54.0]', '(54.0, 71.0]', '(71.0, 88.0]'][:len(bins)]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    
    # Add overall AUC line - compute from data
    overall_auc = np.mean([a for a, c in zip(aucs, counts) if c > 0])
    ax.axhline(y=overall_auc, color=COLORS['accent'], linestyle='--', lw=1.5, 
               label=f'Overall: {overall_auc:.3f}')
    
    ax.set_xlabel('Title Length Range')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Performance by Title Length')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left')
    ax.set_ylim([0, 1])


def plot_metrics_vs_threshold(ax, df):
    """Plot precision, recall, F1 vs threshold."""
    y_true = df['label'].values
    y_prob = df['prob'].values
    
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    f1s = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    ax.plot(thresholds, precisions, color=COLORS['precision'], lw=2, label='Precision')
    ax.plot(thresholds, recalls, color=COLORS['recall'], lw=2, label='Recall')
    ax.plot(thresholds, f1s, color=COLORS['f1'], lw=2, label='F1')
    
    # Mark best F1 threshold
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    ax.axvline(x=best_threshold, color=COLORS['neutral'], linestyle=':', lw=1.5,
               label=f'Best F1 @ {best_threshold:.2f}')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Metrics vs Threshold')
    ax.legend(loc='center left')
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1.05])


def plot_error_rate_by_points(ax, df):
    """Plot error rate by points range."""
    # Define bins to match reference image
    bins = [0, 25, 50, 100, 200, 500, 5000]
    labels = ['(0, 25]', '(25, 50]', '(50, 100]', '(100, 200]', '(200, 500]', '(500, 5000]']
    
    df_copy = df.copy()
    df_copy['points_bin'] = pd.cut(df_copy['points'], bins=bins, labels=labels)
    
    error_rates = []
    counts = []
    for label in labels:
        bin_data = df_copy[df_copy['points_bin'] == label]
        if len(bin_data) > 0:
            # Error is when prediction doesn't match label
            y_true = bin_data['label'].values
            y_prob = bin_data['prob'].values
            y_pred = (y_prob >= 0.3).astype(int)  # Using threshold ~0.3
            error_rate = (y_pred != y_true).mean()
            error_rates.append(error_rate)
            counts.append(len(bin_data))
        else:
            error_rates.append(0)
            counts.append(0)
    
    x = np.arange(len(labels))
    bars = ax.bar(x, error_rates, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Points Range')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Points Range')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0, 0.6])


def plot_precision_recall_curve(ax, df):
    """Plot precision-recall curve."""
    y_true = df['label'].values
    y_prob = df['prob'].values
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    
    ax.plot(recall, precision, color=COLORS['secondary'], lw=2, 
            label=f'PR (AP={ap:.3f})')
    ax.axhline(y=baseline, color=COLORS['neutral'], linestyle='--', lw=1,
               label=f'Baseline ({baseline:.2f})')
    ax.fill_between(recall, precision, alpha=0.1, color=COLORS['secondary'])
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_aspect('equal')


def plot_performance_by_domain(ax, analysis):
    """Plot performance by top domains."""
    domain_data = analysis['segments']['by_domain']
    
    # Sort by count and get top 10
    domains = [(d, data['auc'], data['count']) 
               for d, data in domain_data.items() if data['count'] >= 10]
    domains = sorted(domains, key=lambda x: x[2], reverse=True)[:10]
    
    names = [d[0] if d[0] else '(none)' for d in domains]
    aucs = [d[1] for d in domains]
    
    x = np.arange(len(names))
    colors = [COLORS['secondary'] if a > 0.75 else COLORS['primary'] for a in aucs]
    ax.barh(x, aucs, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('ROC AUC')
    ax.set_title('Performance by Domain (Top 10 by Count)')
    ax.axvline(x=0.767, color=COLORS['accent'], linestyle='--', lw=1.5, alpha=0.7)
    ax.set_xlim([0.5, 1.0])
    ax.invert_yaxis()


def plot_word_analysis(ax, analysis):
    """Plot high hit rate vs low hit rate words."""
    high_words = analysis['words']['high_hit_words'][:10]
    low_words = analysis['words']['low_hit_words'][:10]
    
    # Combine data
    all_words = [(w['word'], w['hit_rate'], 'High Hit') for w in high_words]
    all_words += [(w['word'], w['hit_rate'], 'Low Hit') for w in low_words]
    
    words = [w[0] for w in all_words]
    rates = [w[1] for w in all_words]
    types = [w[2] for w in all_words]
    
    colors = [COLORS['secondary'] if t == 'High Hit' else COLORS['accent'] for t in types]
    
    x = np.arange(len(words))
    ax.barh(x, rates, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_yticks(x)
    ax.set_yticklabels(words, fontsize=8)
    ax.set_xlabel('Hit Rate')
    ax.set_title('Word Analysis: High vs Low Hit Rate')
    ax.axvline(x=0.5, color=COLORS['neutral'], linestyle='--', lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.invert_yaxis()


def plot_confusion_matrix(ax, config):
    """Plot confusion matrix heatmap."""
    cm = config['confusion_matrix']
    matrix = np.array([[cm['tn'], cm['fp']], 
                       [cm['fn'], cm['tp']]])
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = f'{matrix[i, j]:,}'
            color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=12)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Not Hit', 'Pred: Hit'])
    ax.set_yticklabels(['Actual: Not Hit', 'Actual: Hit'])
    ax.set_title('Confusion Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_post_type_performance(ax, analysis):
    """Plot performance by post type."""
    post_data = analysis['segments']['by_post_type']
    
    types = list(post_data.keys())
    aucs = [post_data[t]['auc'] for t in types]
    counts = [post_data[t]['count'] for t in types]
    
    x = np.arange(len(types))
    bars = ax.bar(x, aucs, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Overall AUC line
    overall_auc = sum(a * c for a, c in zip(aucs, counts)) / sum(counts)
    ax.axhline(y=overall_auc, color=COLORS['accent'], linestyle='--', lw=1.5,
               label=f'Overall: {overall_auc:.3f}')
    
    ax.set_xlabel('Post Type')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Performance by Post Type')
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right')


def plot_calibration_error_by_bin(ax, analysis):
    """Plot calibration error by probability bin."""
    cal_data = analysis['calibration']['bins']
    
    bin_indices = list(cal_data['count'].keys())
    counts = [cal_data['count'][b] for b in bin_indices]
    cal_errors = [cal_data['calibration_error'][b] for b in bin_indices]
    mean_pred = [cal_data['mean_predicted'][b] for b in bin_indices]
    
    # Create bar width based on bin
    x = np.arange(len(bin_indices))
    bars = ax.bar(x, cal_errors, color=COLORS['hit'], alpha=0.8, edgecolor='white')
    
    # Create labels as probability ranges
    labels = [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)]
    
    ax.set_xlabel('Probability Bin')
    ax.set_ylabel('Calibration Error')
    ax.set_title('Calibration Error by Probability Bin')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    # Add ECE line
    ece = analysis['calibration']['ece']
    ax.axhline(y=ece, color=COLORS['accent'], linestyle='--', lw=1.5,
               label=f'ECE: {ece:.3f}')
    ax.legend(loc='upper left')


def plot_probability_by_category(ax, df):
    """Plot probability distribution by category (low/medium/high)."""
    categories = ['low', 'medium', 'high']
    colors = [COLORS['not_hit'], COLORS['neutral'], COLORS['hit']]
    
    for cat, color in zip(categories, colors):
        cat_data = df[df['category'] == cat]['prob']
        if len(cat_data) > 0:
            ax.hist(cat_data, bins=30, alpha=0.5, label=f'{cat.capitalize()} (n={len(cat_data)})', 
                    color=color, density=True, edgecolor='white', linewidth=0.3)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Category')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])


def main():
    print("Loading data...")
    df, config, analysis = load_data()
    print(f"  Predictions: {len(df):,}")
    print(f"  ROC AUC (config): {config['metrics']['roc_auc']:.4f}")
    
    threshold = config['optimal_threshold']
    
    setup_style()
    
    # Create figure with 2 columns, 6 rows (12 plots)
    fig, axes = plt.subplots(6, 2, figsize=(14, 26))
    fig.suptitle('HN Success Predictor - Model Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    print("\nGenerating plots...")
    
    # Row 1
    plot_score_distribution(axes[0, 0], df, threshold)
    plot_calibration(axes[0, 1], df, config)
    
    # Row 2
    plot_roc_curve(axes[1, 0], df)
    plot_precision_recall_curve(axes[1, 1], df)
    
    # Row 3
    plot_performance_by_title_length(axes[2, 0], analysis)
    plot_metrics_vs_threshold(axes[2, 1], df)
    
    # Row 4
    plot_error_rate_by_points(axes[3, 0], df)
    plot_performance_by_domain(axes[3, 1], analysis)
    
    # Row 5
    plot_post_type_performance(axes[4, 0], analysis)
    plot_word_analysis(axes[4, 1], analysis)
    
    # Row 6
    plot_calibration_error_by_bin(axes[5, 0], analysis)
    plot_probability_by_category(axes[5, 1], df)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    
    # Save figure
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    print("✓ Done!")


if __name__ == "__main__":
    main()
