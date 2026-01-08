#!/usr/bin/env python3
"""
Fetch balanced Hacker News training data: both high and low scoring posts.
This creates a proper dataset for training a success predictor.
"""

import requests
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

# Algolia HN Search API
ALGOLIA_API = "https://hn.algolia.com/api/v1/search_by_date"

# Database path
DB_PATH = Path(__file__).parent / "hn_training_data.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with the posts table."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT,
            domain TEXT,
            author TEXT NOT NULL,
            points INTEGER NOT NULL,
            num_comments INTEGER,
            created_at INTEGER NOT NULL,
            created_date TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            category TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points ON posts(points DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON posts(category)")
    conn.commit()
    return conn


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def fetch_posts_by_points_range(
    min_points: int,
    max_points: Optional[int],
    num_posts: int,
    months_back: int = 12,
    category: str = "unknown"
) -> list:
    """
    Fetch posts within a specific points range.
    """
    all_posts = []
    seen_ids = set()
    
    now = datetime.now()
    
    # Build points filter
    if max_points:
        points_filter = f"points>={min_points},points<{max_points}"
    else:
        points_filter = f"points>={min_points}"
    
    print(f"\n{'='*60}")
    print(f"Fetching {category} posts (points: {min_points}-{max_points or '∞'})...")
    print(f"{'='*60}")
    
    # Iterate through each month
    for month_offset in range(months_back):
        end_date = now - timedelta(days=month_offset * 30)
        start_date = now - timedelta(days=(month_offset + 1) * 30)
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        month_label = start_date.strftime("%Y-%m")
        
        page = 0
        month_posts = 0
        
        while page < 20:  # Max 20 pages per time window
            params = {
                "tags": "story",
                "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts},{points_filter}",
                "hitsPerPage": 1000,
                "page": page,
            }
            
            try:
                response = requests.get(ALGOLIA_API, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                hits = data.get("hits", [])
                nb_pages = data.get("nbPages", 0)
                
                if not hits:
                    break
                
                for hit in hits:
                    post_id = int(hit.get("objectID", 0))
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)
                    
                    url = hit.get("url", "") or ""
                    post = {
                        "id": post_id,
                        "title": hit.get("title", ""),
                        "url": url,
                        "domain": extract_domain(url),
                        "author": hit.get("author", ""),
                        "points": hit.get("points", 0) or 0,
                        "num_comments": hit.get("num_comments", 0) or 0,
                        "created_at": hit.get("created_at_i", 0),
                        "category": category,
                    }
                    if post["title"]:
                        all_posts.append(post)
                        month_posts += 1
                
                page += 1
                if page >= nb_pages:
                    break
                    
                time.sleep(0.2)
                
            except requests.RequestException as e:
                print(f"  Error: {e}")
                time.sleep(1)
                break
        
        print(f"  📅 {month_label}: +{month_posts} posts (total: {len(all_posts)})")
        
        # Stop if we have enough
        if len(all_posts) >= num_posts:
            break
            
        time.sleep(0.3)
    
    # Return requested number of posts
    return all_posts[:num_posts]


def save_to_db(conn: sqlite3.Connection, posts: list[dict]):
    """Save posts to the database."""
    now = datetime.now().isoformat()
    
    for post in posts:
        created_date = datetime.fromtimestamp(post["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO posts 
                (id, title, url, domain, author, points, num_comments, created_at, created_date, fetched_at, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post["id"],
                post["title"],
                post["url"],
                post["domain"],
                post["author"],
                post["points"],
                post["num_comments"],
                post["created_at"],
                created_date,
                now,
                post["category"],
            ))
        except sqlite3.IntegrityError:
            pass  # Skip duplicates
    
    conn.commit()


def print_stats(conn: sqlite3.Connection):
    """Print statistics about the stored posts."""
    cursor = conn.cursor()
    
    # Category breakdown
    categories = cursor.execute("""
        SELECT category, COUNT(*), MIN(points), MAX(points), AVG(points)
        FROM posts
        GROUP BY category
        ORDER BY AVG(points) DESC
    """).fetchall()
    
    total = cursor.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    
    # Domain stats
    top_domains = cursor.execute("""
        SELECT domain, COUNT(*) as cnt, AVG(points) as avg_pts
        FROM posts
        WHERE domain != ''
        GROUP BY domain
        ORDER BY cnt DESC
        LIMIT 15
    """).fetchall()
    
    print("\n" + "=" * 60)
    print("TRAINING DATA STATISTICS")
    print("=" * 60)
    print(f"Total posts: {total}")
    
    print("\n📊 BY CATEGORY:")
    print("-" * 60)
    for cat, count, min_pts, max_pts, avg_pts in categories:
        print(f"  {cat:15} | {count:5} posts | pts: {min_pts:4}-{max_pts:5} (avg: {avg_pts:.0f})")
    
    print("\n🌐 TOP DOMAINS:")
    print("-" * 60)
    for domain, count, avg_pts in top_domains:
        print(f"  {domain:30} | {count:4} posts | avg: {avg_pts:.0f} pts")


def main():
    print("=" * 60)
    print("HACKER NEWS TRAINING DATA FETCHER")
    print("=" * 60)
    
    # Initialize database
    conn = init_db(DB_PATH)
    
    # Clear existing data for fresh start
    conn.execute("DELETE FROM posts")
    conn.commit()
    
    # Fetch different categories of posts
    # Category 1: Top performers (100+ points) - "viral"
    top_posts = fetch_posts_by_points_range(
        min_points=100,
        max_points=None,
        num_posts=5000,
        months_back=12,
        category="high"
    )
    save_to_db(conn, top_posts)
    print(f"\n✓ Saved {len(top_posts)} high-scoring posts")
    
    # Category 2: Medium performers (20-99 points) - "moderate"
    medium_posts = fetch_posts_by_points_range(
        min_points=20,
        max_points=100,
        num_posts=5000,
        months_back=12,
        category="medium"
    )
    save_to_db(conn, medium_posts)
    print(f"\n✓ Saved {len(medium_posts)} medium-scoring posts")
    
    # Category 3: Low performers (1-19 points) - "low"
    low_posts = fetch_posts_by_points_range(
        min_points=1,
        max_points=20,
        num_posts=5000,
        months_back=12,
        category="low"
    )
    save_to_db(conn, low_posts)
    print(f"\n✓ Saved {len(low_posts)} low-scoring posts")
    
    # Print statistics
    print_stats(conn)
    
    conn.close()
    print(f"\n✓ Training data saved to {DB_PATH}")


if __name__ == "__main__":
    main()
