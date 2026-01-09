"""SQLite storage layer for feeds and entries."""

import csv
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .fetcher import FeedEntry, FetchResult


class FeedCSVManager:
    """Manages feeds stored in a CSV file."""
    
    def __init__(self, csv_path: str = "feeds.csv"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            self._create_empty_csv()
    
    def _create_empty_csv(self):
        """Create an empty CSV file with headers."""
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'url'])
    
    def get_all_feeds(self) -> list[dict]:
        """Read all feeds from the CSV file."""
        feeds = []
        if not self.csv_path.exists():
            return feeds
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('url') and row.get('name'):
                    feeds.append({
                        'name': row['name'].strip(),
                        'url': row['url'].strip()
                    })
        return feeds
    
    def add_feed(self, name: str, url: str) -> bool:
        """Add a new feed to the CSV file. Returns True if added, False if exists."""
        # Check if feed already exists
        existing_feeds = self.get_all_feeds()
        for feed in existing_feeds:
            if feed['url'].lower() == url.lower():
                return False
        
        # Ensure file ends with newline before appending
        if self.csv_path.exists():
            with open(self.csv_path, 'rb') as f:
                f.seek(-1, 2)  # Go to last byte
                if f.read(1) != b'\n':
                    with open(self.csv_path, 'a', encoding='utf-8') as fa:
                        fa.write('\n')
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name.strip(), url.strip()])
        return True
    
    def remove_feed(self, url: str) -> bool:
        """Remove a feed by URL. Returns True if removed."""
        feeds = self.get_all_feeds()
        original_count = len(feeds)
        feeds = [f for f in feeds if f['url'].lower() != url.lower()]
        
        if len(feeds) == original_count:
            return False
        
        # Rewrite CSV
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'url'])
            for feed in feeds:
                writer.writerow([feed['name'], feed['url']])
        return True


class FeedStorage:
    """SQLite-based storage for RSS feeds and entries."""
    
    def __init__(self, db_path: str = "rss_reader.db"):
        self.db_path = Path(db_path)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Feeds table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feeds (
                    url TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    discovered INTEGER DEFAULT 0,
                    etag TEXT,
                    last_modified TEXT,
                    last_fetched TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    feed_url TEXT NOT NULL,
                    feed_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    link TEXT,
                    published TEXT,
                    updated TEXT,
                    summary TEXT,
                    content TEXT,
                    author TEXT,
                    tags TEXT,
                    is_read INTEGER DEFAULT 0,
                    is_starred INTEGER DEFAULT 0,
                    hn_score REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (feed_url) REFERENCES feeds(url)
                )
            """)
            
            # Migration: Add hn_score column if it doesn't exist
            cursor.execute("PRAGMA table_info(entries)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'hn_score' not in columns:
                cursor.execute("ALTER TABLE entries ADD COLUMN hn_score REAL")
            
            # Index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_feed_url ON entries(feed_url)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_published ON entries(published DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_is_read ON entries(is_read)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_entries_hn_score ON entries(hn_score DESC)
            """)
    
    def save_feed(self, url: str, name: str, discovered: bool = False,
                  etag: Optional[str] = None, last_modified: Optional[str] = None):
        """Save or update a feed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feeds (url, name, discovered, etag, last_modified, last_fetched)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    name = excluded.name,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    last_fetched = excluded.last_fetched
            """, (url, name, int(discovered), etag, last_modified, datetime.now().isoformat()))
    
    def save_entries(self, entries: list[FeedEntry]) -> tuple[int, int]:
        """Save entries to the database. Returns (new_count, updated_count)."""
        new_count = 0
        updated_count = 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for entry in entries:
                # Check if entry exists
                cursor.execute("SELECT id FROM entries WHERE id = ?", (entry.id,))
                exists = cursor.fetchone()
                
                if exists:
                    # Update existing entry
                    cursor.execute("""
                        UPDATE entries SET
                            title = ?,
                            link = ?,
                            published = ?,
                            updated = ?,
                            summary = ?,
                            content = ?,
                            author = ?,
                            tags = ?
                        WHERE id = ?
                    """, (
                        entry.title,
                        entry.link,
                        entry.published.isoformat() if entry.published else None,
                        entry.updated.isoformat() if entry.updated else None,
                        entry.summary,
                        entry.content,
                        entry.author,
                        json.dumps(entry.tags),
                        entry.id,
                    ))
                    updated_count += 1
                else:
                    # Insert new entry
                    cursor.execute("""
                        INSERT INTO entries (id, feed_url, feed_name, title, link, published,
                                           updated, summary, content, author, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.id,
                        entry.feed_url,
                        entry.feed_name,
                        entry.title,
                        entry.link,
                        entry.published.isoformat() if entry.published else None,
                        entry.updated.isoformat() if entry.updated else None,
                        entry.summary,
                        entry.content,
                        entry.author,
                        json.dumps(entry.tags),
                    ))
                    new_count += 1
        
        return new_count, updated_count
    
    def update_hn_scores(self, scores: dict[str, float]):
        """Update HN scores for entries.
        
        Args:
            scores: Dict mapping entry ID to HN score
        """
        if not scores:
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for entry_id, score in scores.items():
                cursor.execute(
                    "UPDATE entries SET hn_score = ? WHERE id = ?",
                    (score, entry_id)
                )
    
    def get_entries_without_scores(self, limit: int = 1000) -> list[dict]:
        """Get entries that don't have HN scores yet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title FROM entries WHERE hn_score IS NULL LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def process_fetch_results(self, results: list[FetchResult]) -> dict:
        """Process fetch results and save to database."""
        total_new = 0
        total_updated = 0
        
        for result in results:
            if result.success and result.feed.url:
                # Save feed metadata
                self.save_feed(
                    url=result.feed.url,
                    name=result.feed.name,
                    discovered=result.feed.discovered,
                    etag=result.etag,
                    last_modified=result.last_modified,
                )
                
                # Save entries
                if result.entries:
                    new_count, updated_count = self.save_entries(result.entries)
                    total_new += new_count
                    total_updated += updated_count
        
        return {
            "new_entries": total_new,
            "updated_entries": total_updated,
        }
    
    def get_entries(
        self,
        feed_url: Optional[str] = None,
        unread_only: bool = False,
        starred_only: bool = False,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
        sort_by_score: bool = False,
        hours: Optional[int] = None,
    ) -> list[dict]:
        """Get entries with optional filtering.
        
        Args:
            feed_url: Filter by feed URL
            unread_only: Only show unread entries
            starred_only: Only show starred entries
            limit: Maximum number of entries
            offset: Offset for pagination
            search: Search term for title/summary/content
            sort_by_score: Sort by HN score (descending) instead of date
            hours: Only show entries from the last N hours
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM entries WHERE 1=1"
            params = []
            
            if feed_url:
                query += " AND feed_url = ?"
                params.append(feed_url)
            
            if unread_only:
                query += " AND is_read = 0"
            
            if starred_only:
                query += " AND is_starred = 1"
            
            if search:
                query += " AND (title LIKE ? OR summary LIKE ? OR content LIKE ?)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term])
            
            if hours:
                from datetime import datetime, timedelta
                cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
                query += " AND published > ?"
                params.append(cutoff)
            
            if sort_by_score:
                query += " ORDER BY hn_score DESC NULLS LAST, published DESC NULLS LAST LIMIT ? OFFSET ?"
            else:
                query += " ORDER BY published DESC NULLS LAST LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            entries = []
            for row in rows:
                entry = dict(row)
                entry['tags'] = json.loads(entry['tags']) if entry['tags'] else []
                entries.append(entry)
            
            return entries
    
    def get_all_entries_for_export(self) -> list[dict]:
        """Get all entries for export (JSON/HTML)."""
        return self.get_entries(limit=100000)
    
    def mark_read(self, entry_id: str, is_read: bool = True):
        """Mark an entry as read/unread."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE entries SET is_read = ? WHERE id = ?",
                (int(is_read), entry_id)
            )
    
    def mark_starred(self, entry_id: str, is_starred: bool = True):
        """Mark an entry as starred/unstarred."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE entries SET is_starred = ? WHERE id = ?",
                (int(is_starred), entry_id)
            )
    
    def mark_all_read(self, feed_url: Optional[str] = None):
        """Mark all entries as read, optionally for a specific feed."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if feed_url:
                cursor.execute(
                    "UPDATE entries SET is_read = 1 WHERE feed_url = ?",
                    (feed_url,)
                )
            else:
                cursor.execute("UPDATE entries SET is_read = 1")
    
    def get_feeds(self) -> list[dict]:
        """Get all feeds with entry counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.*, 
                       COUNT(e.id) as total_entries,
                       SUM(CASE WHEN e.is_read = 0 THEN 1 ELSE 0 END) as unread_count
                FROM feeds f
                LEFT JOIN entries e ON f.url = e.feed_url
                GROUP BY f.url
                ORDER BY f.name
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_cache_info(self) -> dict:
        """Get ETag and Last-Modified info for all feeds."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT url, etag, last_modified FROM feeds")
            return {
                row['url']: {
                    'etag': row['etag'],
                    'last_modified': row['last_modified']
                }
                for row in cursor.fetchall()
            }
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM feeds")
            feed_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM entries")
            entry_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM entries WHERE is_read = 0")
            unread_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM entries WHERE is_starred = 1")
            starred_count = cursor.fetchone()[0]
            
            return {
                "feeds": feed_count,
                "entries": entry_count,
                "unread": unread_count,
                "starred": starred_count,
            }
