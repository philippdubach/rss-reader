#!/usr/bin/env python3
"""
HN Success Predictor: Predicts how many points a post will receive.
Uses Ridge Regression with title bag-of-words + domain features.
"""

import re
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import nltk

# Paths
DB_PATH = Path(__file__).parent / "hn_training_data.db"
MODEL_PATH = Path(__file__).parent / "success_predictor.pkl"


def tokenize_title(title: str) -> str:
    """
    Tokenize and clean a title for bag-of-words features.
    """
    if not title:
        return ""
    
    # Lowercase
    title = title.lower()
    
    # Tokenize
    try:
        tokens = nltk.word_tokenize(title)
    except Exception:
        # Fallback to simple split
        tokens = title.split()
    
    # Keep alphanumeric tokens and common programming tokens
    cleaned = []
    for token in tokens:
        # Keep if alphanumeric or common programming symbols
        if token.isalnum() or token in ['++', '--', '#', '.js', '.py', '.go']:
            cleaned.append(token)
    
    return " ".join(cleaned)


def extract_domain(url: str) -> str:
    """Extract domain from URL for feature engineering."""
    if not url:
        return "no_url"
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain if domain else "no_url"
    except Exception:
        return "no_url"


def load_training_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load training data from SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT id, title, url, domain, author, points, num_comments, category
        FROM posts
        WHERE title IS NOT NULL AND title != ''
    """, conn)
    conn.close()
    
    # Clean domain
    df['domain'] = df['domain'].fillna('').apply(lambda x: x if x else 'no_url')
    
    # Tokenize titles
    df['title_clean'] = df['title'].apply(tokenize_title)
    
    return df


def create_model_pipeline(top_n_domains: int = 100) -> Pipeline:
    """
    Create sklearn pipeline with:
    - TF-IDF on title tokens
    - Ridge regression for point prediction
    """
    # We'll use a simple pipeline with TF-IDF
    # Domain will be embedded in title as a special token
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=3,
            max_df=0.95,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    return pipeline


def train_model(df: pd.DataFrame) -> Tuple[Pipeline, dict]:
    """
    Train the success predictor model.
    Returns trained pipeline and evaluation metrics.
    """
    # Prepare features: combine title with domain as a special token
    df['features'] = df.apply(
        lambda row: f"DOMAIN_{row['domain'].replace('.', '_')} {row['title_clean']}",
        axis=1
    )
    
    X = df['features']
    # Use log of points for better regression (points are right-skewed)
    y = np.log1p(df['points'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train pipeline
    pipeline = create_model_pipeline()
    
    # Cross-validation for alpha tuning
    print("\nTraining model with cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"CV R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Fit final model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    
    # Convert back from log scale for interpretable metrics
    y_test_pts = np.expm1(y_test)
    y_pred_pts = np.expm1(y_pred)
    
    metrics = {
        'r2_log': r2_score(y_test, y_pred),
        'rmse_log': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae_points': mean_absolute_error(y_test_pts, y_pred_pts),
        'rmse_points': np.sqrt(mean_squared_error(y_test_pts, y_pred_pts)),
        'r2_points': r2_score(y_test_pts, y_pred_pts),
    }
    
    return pipeline, metrics, (X_test, y_test_pts, y_pred_pts)


def analyze_feature_importance(pipeline: Pipeline, top_n: int = 30):
    """Analyze which features (words/domains) are most predictive."""
    tfidf = pipeline.named_steps['tfidf']
    regressor = pipeline.named_steps['regressor']
    
    feature_names = tfidf.get_feature_names_out()
    coefficients = regressor.coef_
    
    # Sort by absolute coefficient
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    })
    
    print("\n" + "=" * 60)
    print("MOST PREDICTIVE FEATURES")
    print("=" * 60)
    
    print("\n🔥 POSITIVE (predict HIGH points):")
    print("-" * 40)
    top_positive = importance.nlargest(top_n, 'coefficient')
    for _, row in top_positive.iterrows():
        print(f"  {row['feature']:30} +{row['coefficient']:.3f}")
    
    print("\n❄️  NEGATIVE (predict LOW points):")
    print("-" * 40)
    top_negative = importance.nsmallest(top_n, 'coefficient')
    for _, row in top_negative.iterrows():
        print(f"  {row['feature']:30} {row['coefficient']:.3f}")
    
    return importance


def predict_success(pipeline: Pipeline, titles: List[str], urls: List[str] = None) -> np.ndarray:
    """
    Predict success (points) for a list of titles.
    Returns predicted points.
    """
    if urls is None:
        urls = [''] * len(titles)
    
    features = []
    for title, url in zip(titles, urls):
        domain = extract_domain(url)
        title_clean = tokenize_title(title)
        features.append(f"DOMAIN_{domain.replace('.', '_')} {title_clean}")
    
    # Predict in log space
    log_predictions = pipeline.predict(features)
    # Convert back to points
    return np.expm1(log_predictions)


def main():
    print("=" * 60)
    print("HN SUCCESS PREDICTOR - TRAINING")
    print("=" * 60)
    
    # Load data
    print("\nLoading training data...")
    df = load_training_data()
    print(f"Loaded {len(df)} posts")
    print(f"Points range: {df['points'].min()} - {df['points'].max()}")
    print(f"Unique domains: {df['domain'].nunique()}")
    
    # Train model
    pipeline, metrics, (X_test, y_test_pts, y_pred_pts) = train_model(df)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"R² (log scale):     {metrics['r2_log']:.4f}")
    print(f"R² (points):        {metrics['r2_points']:.4f}")
    print(f"MAE (points):       {metrics['mae_points']:.1f}")
    print(f"RMSE (points):      {metrics['rmse_points']:.1f}")
    
    # Analyze feature importance
    importance = analyze_feature_importance(pipeline)
    
    # Show some example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS (from test set)")
    print("=" * 60)
    
    # Get some examples
    test_df = pd.DataFrame({
        'actual': y_test_pts.values,
        'predicted': y_pred_pts,
    })
    test_df['error'] = test_df['predicted'] - test_df['actual']
    test_df['abs_error'] = test_df['error'].abs()
    
    print("\n✅ Best predictions (lowest error):")
    best = test_df.nsmallest(10, 'abs_error')
    for idx, row in best.iterrows():
        print(f"  Actual: {row['actual']:5.0f} | Predicted: {row['predicted']:5.0f} | Error: {row['error']:+6.1f}")
    
    print("\n❌ Worst predictions (highest error):")
    worst = test_df.nlargest(10, 'abs_error')
    for idx, row in worst.iterrows():
        print(f"  Actual: {row['actual']:5.0f} | Predicted: {row['predicted']:5.0f} | Error: {row['error']:+6.1f}")
    
    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✓ Model saved to {MODEL_PATH}")
    
    # Demo predictions on new titles
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
        "The Art of Debugging",
        "Facebook is down",
    ]
    demo_urls = [
        "https://github.com/user/project",
        "https://techcrunch.com/article",
        "https://medium.com/@author/story",
        "https://python.org/news",
        "",  # self-post
        "https://blog.example.com/linux",
        "https://example.com/debugging",
        "https://twitter.com/user/status",
    ]
    
    predictions = predict_success(pipeline, demo_titles, demo_urls)
    
    for title, url, pred in zip(demo_titles, demo_urls, predictions):
        domain = extract_domain(url)
        print(f"  [{pred:5.0f} pts] {title[:50]}")
        print(f"             Domain: {domain}")


if __name__ == "__main__":
    main()
