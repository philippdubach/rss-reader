#!/usr/bin/env python3
"""
RSS Recommender: Score RSS entries using the HN success predictor.
Ranks articles by predicted success probability.
"""

import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

# Paths
RSS_DB_PATH = Path(__file__).parent / "rss_reader.db"
MODEL_PATH = Path(__file__).parent / "success_predictor_v2.pkl"


def load_model():
    """Load the trained success predictor."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_classifier_v2.py first.")
    return joblib.load(MODEL_PATH)


def get_unread_entries(db_path: Path = RSS_DB_PATH, limit: int = 100) -> pd.DataFrame:
    """Get unread entries from RSS database."""
    if not db_path.exists():
        raise FileNotFoundError(f"RSS database not found at {db_path}. Run 'python main.py refresh' first.")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT id, feed_name, title, link, published, summary, author
        FROM entries
        WHERE is_read = 0
        ORDER BY published DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    
    return df


def get_all_entries(db_path: Path = RSS_DB_PATH, limit: int = 500) -> pd.DataFrame:
    """Get all recent entries from RSS database."""
    if not db_path.exists():
        raise FileNotFoundError(f"RSS database not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT id, feed_name, title, link, published, summary, author
        FROM entries
        ORDER BY published DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    
    return df


def score_entries(entries_df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Score entries using the success predictor."""
    # Prepare data for prediction
    titles = entries_df['title'].fillna('').tolist()
    urls = entries_df['link'].fillna('').tolist()
    
    # Get predictions
    predictions = predictor.predict(titles, urls)
    
    # Combine with original data
    result = entries_df.copy()
    result['hit_probability'] = predictions['hit_probability'].values
    result['expected_points'] = predictions['expected_points'].values
    result['recommendation'] = predictions['recommendation'].values
    
    # Sort by hit probability
    result = result.sort_values('hit_probability', ascending=False)
    
    return result


def display_recommendations(scored_df: pd.DataFrame, top_n: int = 20):
    """Display top recommendations."""
    print("\n" + "=" * 80)
    print("📰 RSS FEED RECOMMENDATIONS (by predicted success)")
    print("=" * 80)
    
    # Group by recommendation level
    for rec_level in ['Excellent', 'Good', 'Maybe']:
        group = scored_df[scored_df['recommendation'] == rec_level].head(top_n // 3 + 5)
        
        if len(group) == 0:
            continue
            
        if rec_level == 'Excellent':
            emoji = '🔥'
        elif rec_level == 'Good':
            emoji = '✅'
        else:
            emoji = '🤔'
        
        print(f"\n{emoji} {rec_level.upper()} ({len(group)} articles):")
        print("-" * 80)
        
        for _, row in group.iterrows():
            prob = row['hit_probability']
            pts = row['expected_points']
            title = row['title'][:60] if row['title'] else '(no title)'
            feed = row['feed_name'][:20] if row['feed_name'] else ''
            
            print(f"  [{prob:.0%}] {title}")
            print(f"         └─ {feed}")


def main():
    """Main entry point for RSS recommendations."""
    print("=" * 80)
    print("RSS RECOMMENDER")
    print("=" * 80)
    
    # Load model
    print("\nLoading success predictor...")
    try:
        predictor = load_model()
        print("✓ Model loaded")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    # Get entries
    print("\nLoading RSS entries...")
    try:
        entries = get_all_entries(limit=200)
        print(f"✓ Loaded {len(entries)} entries")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    if len(entries) == 0:
        print("No entries found. Run 'python main.py refresh' to fetch feeds.")
        return
    
    # Score entries
    print("\nScoring entries...")
    scored = score_entries(entries, predictor)
    
    # Display recommendations
    display_recommendations(scored, top_n=30)
    
    # Summary stats
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries scored: {len(scored)}")
    print(f"Excellent:  {(scored['recommendation'] == 'Excellent').sum()}")
    print(f"Good:       {(scored['recommendation'] == 'Good').sum()}")
    print(f"Maybe:      {(scored['recommendation'] == 'Maybe').sum()}")
    print(f"Skip:       {(scored['recommendation'] == 'Skip').sum()}")
    
    # Export to CSV for easier review
    output_path = Path(__file__).parent / "recommendations.csv"
    scored[['title', 'feed_name', 'link', 'hit_probability', 'expected_points', 'recommendation']].to_csv(
        output_path, index=False
    )
    print(f"\n✓ Recommendations saved to {output_path}")


if __name__ == "__main__":
    main()
