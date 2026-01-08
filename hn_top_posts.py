#!/usr/bin/env python3
"""
Fetch top Hacker News posts from the last 12 months using Algolia API.
Stores results in a SQLite database.
"""

import requests
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# Algolia HN Search API
ALGOLIA_API = "https://hn.algolia.com/api/v1/search"

# Database path
DB_PATH = Path(__file__).parent / "hn_top_posts.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the database with the posts table."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT,
            author TEXT NOT NULL,
            points INTEGER NOT NULL,
            num_comments INTEGER,
            created_at INTEGER NOT NULL,
            created_date TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points ON posts(points DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON posts(created_at DESC)")
    conn.commit()
    return conn


def fetch_top_posts(num_posts: int = 5000, months_back: int = 12) -> list[dict]:
    """
    Fetch top posts from the last N months using Algolia API.
    
    The API returns max 1000 results per query, and we can paginate up to 1000 pages.
    We use numericFilters to filter by date and sort by points.
    """
    posts = []
    
    # Calculate timestamp for N months ago
    cutoff_date = datetime.now() - timedelta(days=months_back * 30)
    cutoff_timestamp = int(cutoff_date.timestamp())
    
    print(f"Fetching top {num_posts} posts since {cutoff_date.strftime('%Y-%m-%d')}...")
    
    # Algolia limits: 1000 hits max per query, hitsPerPage max 1000
    hits_per_page = 1000
    pages_needed = (num_posts + hits_per_page - 1) // hits_per_page
    
    for page in range(pages_needed):
        params = {
            "tags": "story",
            "numericFilters": f"created_at_i>{cutoff_timestamp}",
            "hitsPerPage": hits_per_page,
            "page": page,
            # Note: Algolia HN API doesn't support sorting by points directly
            # We need to fetch more and sort locally
        }
        
        try:
            response = requests.get(ALGOLIA_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hits = data.get("hits", [])
            if not hits:
                print(f"  No more results at page {page}")
                break
                
            for hit in hits:
                post = {
                    "id": int(hit.get("objectID", 0)),
                    "title": hit.get("title", ""),
                    "url": hit.get("url", ""),
                    "author": hit.get("author", ""),
                    "points": hit.get("points", 0) or 0,
                    "num_comments": hit.get("num_comments", 0) or 0,
                    "created_at": hit.get("created_at_i", 0),
                }
                if post["title"] and post["points"] > 0:
                    posts.append(post)
            
            print(f"  Page {page + 1}/{pages_needed}: fetched {len(hits)} posts (total: {len(posts)})")
            
            # Check if we have enough
            if len(posts) >= num_posts * 2:  # Fetch extra to ensure we get top ones
                break
                
            # Be nice to the API
            time.sleep(0.5)
            
        except requests.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            time.sleep(2)
            continue
    
    # Sort by points and take top N
    posts.sort(key=lambda x: x["points"], reverse=True)
    return posts[:num_posts]


def fetch_top_posts_by_points(num_posts: int = 5000, months_back: int = 12) -> list[dict]:
    """
    Fetch top posts by iterating through monthly time windows.
    Algolia limits results to 1000 per query, so we chunk by month.
    """
    all_posts = []
    seen_ids = set()
    
    now = datetime.now()
    
    print(f"Fetching top {num_posts} posts from the last {months_back} months...")
    print("Fetching month by month to work around API limits...")
    
    api_url = "https://hn.algolia.com/api/v1/search_by_date"
    
    # Iterate through each month
    for month_offset in range(months_back):
        # Calculate time window for this month
        end_date = now - timedelta(days=month_offset * 30)
        start_date = now - timedelta(days=(month_offset + 1) * 30)
        
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        
        month_label = start_date.strftime("%Y-%m")
        print(f"\n📅 {month_label}:", end=" ")
        
        page = 0
        month_posts = 0
        
        while page < 20:  # Max 20 pages per time window
            params = {
                "tags": "story",
                "numericFilters": f"created_at_i>{start_ts},created_at_i<{end_ts},points>5",
                "hitsPerPage": 1000,
                "page": page,
            }
            
            try:
                response = requests.get(api_url, params=params, timeout=30)
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
                    
                    post = {
                        "id": post_id,
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "author": hit.get("author", ""),
                        "points": hit.get("points", 0) or 0,
                        "num_comments": hit.get("num_comments", 0) or 0,
                        "created_at": hit.get("created_at_i", 0),
                    }
                    if post["title"] and post["points"] > 0:
                        all_posts.append(post)
                        month_posts += 1
                
                page += 1
                if page >= nb_pages:
                    break
                    
                time.sleep(0.2)
                
            except requests.RequestException as e:
                print(f"Error: {e}")
                time.sleep(1)
                break
        
        print(f"{month_posts} posts")
        time.sleep(0.3)
    
    # Sort by points and take top N
    print(f"\nTotal fetched: {len(all_posts)} posts")
    all_posts.sort(key=lambda x: x["points"], reverse=True)
    return all_posts[:num_posts]


def save_to_db(conn: sqlite3.Connection, posts: list[dict]):
    """Save posts to the database."""
    now = datetime.now().isoformat()
    
    # Clear existing data
    conn.execute("DELETE FROM posts")
    
    for post in posts:
        created_date = datetime.fromtimestamp(post["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("""
            INSERT OR REPLACE INTO posts 
            (id, title, url, author, points, num_comments, created_at, created_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            post["id"],
            post["title"],
            post["url"],
            post["author"],
            post["points"],
            post["num_comments"],
            post["created_at"],
            created_date,
            now,
        ))
    
    conn.commit()
    print(f"\n✓ Saved {len(posts)} posts to {DB_PATH}")


def print_stats(conn: sqlite3.Connection):
    """Print some statistics about the stored posts."""
    cursor = conn.cursor()
    
    # Total posts
    total = cursor.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    
    # Points stats
    stats = cursor.execute("""
        SELECT 
            MIN(points) as min_pts,
            MAX(points) as max_pts,
            AVG(points) as avg_pts,
            MIN(created_date) as oldest,
            MAX(created_date) as newest
        FROM posts
    """).fetchone()
    
    # Top 10 posts
    top_posts = cursor.execute("""
        SELECT title, author, points, created_date 
        FROM posts 
        ORDER BY points DESC 
        LIMIT 10
    """).fetchall()
    
    # Top authors
    top_authors = cursor.execute("""
        SELECT author, COUNT(*) as post_count, SUM(points) as total_points
        FROM posts
        GROUP BY author
        ORDER BY total_points DESC
        LIMIT 10
    """).fetchall()
    
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total posts: {total}")
    print(f"Points range: {stats[0]} - {stats[1]} (avg: {stats[2]:.0f})")
    print(f"Date range: {stats[3]} to {stats[4]}")
    
    print("\n📈 TOP 10 POSTS:")
    print("-" * 60)
    for i, (title, author, points, date) in enumerate(top_posts, 1):
        title_short = title[:50] + "..." if len(title) > 50 else title
        print(f"{i:2}. [{points:4} pts] {title_short}")
        print(f"    by {author} on {date[:10]}")
    
    print("\n👤 TOP 10 AUTHORS (by total points):")
    print("-" * 60)
    for author, count, total_pts in top_authors:
        print(f"  {author}: {count} posts, {total_pts} total points")


def main():
    print("=" * 60)
    print("HACKER NEWS TOP POSTS FETCHER")
    print("=" * 60)
    
    # Initialize database
    conn = init_db(DB_PATH)
    
    # Fetch top posts
    posts = fetch_top_posts_by_points(num_posts=5000, months_back=12)
    
    if posts:
        # Save to database
        save_to_db(conn, posts)
        
        # Print stats
        print_stats(conn)
    else:
        print("No posts fetched!")
    
    conn.close()


if __name__ == "__main__":
    main()
