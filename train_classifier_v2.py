#!/usr/bin/env python3
"""
HN Success Predictor v2: Training script
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, precision_recall_curve
)

# Import predictor classes from shared module
from rss_reader.predictor import FeatureExtractor, SuccessPredictor

# Paths
DB_PATH = Path(__file__).parent / "hn_training_data.db"
MODEL_PATH = Path(__file__).parent / "success_predictor_v2.pkl"


def evaluate_model(predictor: SuccessPredictor, df_test: pd.DataFrame) -> Dict:
    """Comprehensive model evaluation."""
    X = predictor.feature_extractor.transform(df_test)
    
    # Classification metrics
    y_true_class = (df_test['points'] >= predictor.hit_threshold).astype(int)
    y_pred_proba = predictor.classifier.predict_proba(X)[:, 1]
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    
    # Regression metrics
    y_true_points = df_test['points']
    y_pred_points = predictor.predict_points(df_test)
    
    # Calculate precision at different recall levels
    precision, recall, thresholds = precision_recall_curve(y_true_class, y_pred_proba)
    
    metrics = {
        # Classification
        'roc_auc': roc_auc_score(y_true_class, y_pred_proba),
        'accuracy': (y_pred_class == y_true_class).mean(),
        
        # Regression
        'mae_points': mean_absolute_error(y_true_points, y_pred_points),
        'rmse_points': np.sqrt(mean_squared_error(y_true_points, y_pred_points)),
        'r2_points': r2_score(y_true_points, y_pred_points),
        
        # Precision at different thresholds
        'precision_at_recall_0.5': precision[np.argmin(np.abs(recall - 0.5))],
        'precision_at_recall_0.3': precision[np.argmin(np.abs(recall - 0.3))],
    }
    
    return metrics


def main():
    print("=" * 60)
    print("HN SUCCESS PREDICTOR v2 - TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading training data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT id, title, url, domain, author, points, num_comments, category
        FROM posts
        WHERE title IS NOT NULL AND title != ''
    """, conn)
    conn.close()
    
    print(f"Loaded {len(df)} posts")
    print(f"Points distribution:")
    print(df['points'].describe())
    
    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"\nTraining set: {len(df_train)}")
    print(f"Test set: {len(df_test)}")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    predictor = SuccessPredictor(hit_threshold=100)
    predictor.fit(df_train)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    metrics = evaluate_model(predictor, df_test)
    
    print(f"\n📊 CLASSIFICATION (predicting hits >= 100 pts):")
    print(f"   ROC AUC:                {metrics['roc_auc']:.4f}")
    print(f"   Accuracy:               {metrics['accuracy']:.4f}")
    print(f"   Precision @ 50% recall: {metrics['precision_at_recall_0.5']:.4f}")
    print(f"   Precision @ 30% recall: {metrics['precision_at_recall_0.3']:.4f}")
    
    print(f"\n📈 REGRESSION (predicting points):")
    print(f"   MAE:  {metrics['mae_points']:.1f} points")
    print(f"   RMSE: {metrics['rmse_points']:.1f} points")
    print(f"   R²:   {metrics['r2_points']:.4f}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("TOP FEATURES (from classifier)")
    print("=" * 60)
    
    # Get TF-IDF feature names
    tfidf_names = predictor.feature_extractor.tfidf.get_feature_names_out()
    meta_names = ['is_show_hn', 'is_ask_hn', 'is_tell_hn', 'is_self_post',
                  'title_length', 'word_count', 'has_year', 'has_number',
                  'has_question', 'has_exclamation', 'has_colon', 'has_dash',
                  'has_pdf', 'has_video', 'is_github', 'is_arxiv', 'is_medium',
                  'is_twitter', 'is_youtube', 'is_substack', 'is_major_news', 'is_tech_blog']
    domain_names = [f"domain_{d}" for d in predictor.feature_extractor.top_domains]
    
    all_names = list(tfidf_names) + meta_names + domain_names
    coeffs = predictor.classifier.coef_[0]
    
    importance = pd.DataFrame({'feature': all_names[:len(coeffs)], 'coef': coeffs})
    
    print("\n🔥 POSITIVE (predict success):")
    for _, row in importance.nlargest(20, 'coef').iterrows():
        print(f"   {row['feature']:35} +{row['coef']:.3f}")
    
    print("\n❄️  NEGATIVE (predict failure):")
    for _, row in importance.nsmallest(20, 'coef').iterrows():
        print(f"   {row['feature']:35} {row['coef']:.3f}")
    
    # Demo predictions
    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)
    
    demo_titles = [
        "Show HN: I built a neural network in Rust",
        "Google announces layoffs",
        "My startup failed, here's what I learned",
        "Python 4.0 released",
        "Ask HN: Best way to learn machine learning?",
        "Why I switched from Mac to Linux",
        "The Art of Debugging [pdf]",
        "Facebook is down",
        "Claude 4 is now available",
        "Show HN: Open source alternative to Notion",
        "The History of Unix (2019)",
        "I spent 5 years building this app",
    ]
    demo_urls = [
        "https://github.com/user/project",
        "https://techcrunch.com/article",
        "https://medium.com/@author/story",
        "https://python.org/news",
        "",
        "https://blog.example.com/linux",
        "https://arxiv.org/abs/1234.5678",
        "https://twitter.com/user/status",
        "https://anthropic.com/news/claude-4",
        "https://github.com/user/notion-clone",
        "https://example.com/unix-history",
        "https://substack.com/@author/story",
    ]
    
    results = predictor.predict(demo_titles, demo_urls)
    
    print("\n")
    for _, row in results.iterrows():
        prob = row['hit_probability']
        pts = row['expected_points']
        rec = row['recommendation']
        title = row['title'][:50]
        
        # Color-coded emoji
        if rec == 'Excellent':
            emoji = '🔥'
        elif rec == 'Good':
            emoji = '✅'
        elif rec == 'Maybe':
            emoji = '🤔'
        else:
            emoji = '❌'
        
        print(f"{emoji} {rec:10} | {prob:.1%} hit | ~{pts:4.0f} pts | {title}")
    
    # Save model
    joblib.dump(predictor, MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    
    # Cross-validation for more robust estimate
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION (5-fold)")
    print("=" * 60)
    
    X_all = predictor.feature_extractor.transform(df)
    y_class = (df['points'] >= 100).astype(int)
    
    cv_scores = cross_val_score(
        LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced'),
        X_all, y_class, cv=5, scoring='roc_auc'
    )
    print(f"CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Individual folds: {[f'{s:.4f}' for s in cv_scores]}")


if __name__ == "__main__":
    main()
