"""Concurrent feed fetcher using ThreadPoolExecutor."""

import feedparser
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from hashlib import md5
import time

from .feed_parser import FeedSource


@dataclass
class FeedEntry:
    """Represents a single feed entry/article."""
    id: str  # GUID or generated hash
    feed_name: str
    feed_url: str
    title: str
    link: str
    published: Optional[datetime]
    updated: Optional[datetime]
    summary: str
    content: str  # Full content if available
    author: Optional[str]
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "id": self.id,
            "feed_name": self.feed_name,
            "feed_url": self.feed_url,
            "title": self.title,
            "link": self.link,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "summary": self.summary,
            "content": self.content,
            "author": self.author,
            "tags": self.tags,
        }


@dataclass
class FetchResult:
    """Result of fetching a single feed."""
    feed: FeedSource
    success: bool
    entries: list[FeedEntry] = field(default_factory=list)
    error: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    fetch_time: float = 0.0


def parse_date(date_struct) -> Optional[datetime]:
    """Convert feedparser date struct to datetime."""
    if date_struct:
        try:
            return datetime(*date_struct[:6])
        except (TypeError, ValueError):
            pass
    return None


def generate_entry_id(entry: dict, feed_url: str) -> str:
    """Generate a unique ID for an entry."""
    # Try to use the entry's GUID/ID
    if entry.get('id'):
        return entry['id']
    
    # Fall back to hash of title + link
    content = f"{entry.get('title', '')}{entry.get('link', '')}{feed_url}"
    return md5(content.encode()).hexdigest()


def fetch_single_feed(
    feed: FeedSource,
    timeout: int = 15,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> FetchResult:
    """Fetch and parse a single feed."""
    if not feed.url:
        return FetchResult(
            feed=feed,
            success=False,
            error="No URL available",
        )
    
    start_time = time.time()
    
    try:
        # Set up headers
        headers = {
            'User-Agent': 'RSS-Reader/1.0 (Python; +https://github.com/rss-reader)',
            'Accept': 'application/rss+xml, application/atom+xml, application/xml, text/xml',
        }
        
        # Add conditional request headers if available
        if etag:
            headers['If-None-Match'] = etag
        if last_modified:
            headers['If-Modified-Since'] = last_modified
        
        # Fetch the feed
        response = requests.get(
            feed.url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        
        # Handle 304 Not Modified
        if response.status_code == 304:
            return FetchResult(
                feed=feed,
                success=True,
                entries=[],  # No new entries
                etag=etag,
                last_modified=last_modified,
                fetch_time=time.time() - start_time,
            )
        
        response.raise_for_status()
        
        # Parse the feed
        parsed = feedparser.parse(response.content)
        
        if parsed.bozo and not parsed.entries:
            # Feed is malformed and has no entries
            return FetchResult(
                feed=feed,
                success=False,
                error=f"Malformed feed: {parsed.bozo_exception}",
                fetch_time=time.time() - start_time,
            )
        
        # Extract entries
        entries = []
        feed_title = parsed.feed.get('title', feed.name)
        
        for entry in parsed.entries:
            # Get content (prefer full content over summary)
            content = ""
            if entry.get('content'):
                content = entry.content[0].get('value', '')
            elif entry.get('description'):
                content = entry.description
            
            summary = entry.get('summary', content[:500] if content else '')
            
            # Get tags
            tags = []
            if entry.get('tags'):
                tags = [tag.get('term', '') for tag in entry.tags if tag.get('term')]
            
            feed_entry = FeedEntry(
                id=generate_entry_id(entry, feed.url),
                feed_name=feed_title,
                feed_url=feed.url,
                title=entry.get('title', 'Untitled'),
                link=entry.get('link', ''),
                published=parse_date(entry.get('published_parsed')),
                updated=parse_date(entry.get('updated_parsed')),
                summary=summary,
                content=content,
                author=entry.get('author'),
                tags=tags,
            )
            entries.append(feed_entry)
        
        return FetchResult(
            feed=feed,
            success=True,
            entries=entries,
            etag=response.headers.get('ETag'),
            last_modified=response.headers.get('Last-Modified'),
            fetch_time=time.time() - start_time,
        )
        
    except requests.exceptions.Timeout:
        return FetchResult(
            feed=feed,
            success=False,
            error="Connection timed out",
            fetch_time=time.time() - start_time,
        )
    except requests.exceptions.ConnectionError as e:
        return FetchResult(
            feed=feed,
            success=False,
            error=f"Connection error: {e}",
            fetch_time=time.time() - start_time,
        )
    except requests.exceptions.HTTPError as e:
        return FetchResult(
            feed=feed,
            success=False,
            error=f"HTTP error: {e.response.status_code}",
            fetch_time=time.time() - start_time,
        )
    except Exception as e:
        return FetchResult(
            feed=feed,
            success=False,
            error=str(e),
            fetch_time=time.time() - start_time,
        )


def fetch_all_feeds(
    feeds: list[FeedSource],
    max_workers: int = 10,
    timeout: int = 15,
    cache: Optional[dict] = None,
    progress_callback=None,
) -> list[FetchResult]:
    """Fetch all feeds concurrently using ThreadPoolExecutor."""
    results = []
    cache = cache or {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_feed = {}
        for feed in feeds:
            if feed.url:
                cached = cache.get(feed.url, {})
                future = executor.submit(
                    fetch_single_feed,
                    feed,
                    timeout,
                    cached.get('etag'),
                    cached.get('last_modified'),
                )
                future_to_feed[future] = feed
            else:
                # No URL - add a failed result immediately
                results.append(FetchResult(
                    feed=feed,
                    success=False,
                    error="No URL available",
                ))
        
        # Collect results as they complete
        completed = 0
        total = len(future_to_feed)
        
        for future in as_completed(future_to_feed):
            result = future.result()
            results.append(result)
            completed += 1
            
            if progress_callback:
                progress_callback(completed, total, result)
    
    return results


def get_fetch_summary(results: list[FetchResult]) -> dict:
    """Get a summary of fetch results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_entries = sum(len(r.entries) for r in successful)
    total_time = sum(r.fetch_time for r in results)
    
    return {
        "total_feeds": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_entries": total_entries,
        "total_time": round(total_time, 2),
        "failed_feeds": [(r.feed.name, r.error) for r in failed],
    }
