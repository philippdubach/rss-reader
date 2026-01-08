"""Parse feeds.md file and extract feed URLs, with auto-discovery for missing URLs."""

import re
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class FeedSource:
    """Represents a feed source with name and URL."""
    name: str
    url: Optional[str]
    discovered: bool = False  # True if URL was auto-discovered


# Known feed URLs for popular blogs that don't have URLs in feeds.md
KNOWN_FEEDS = {
    "joel on software": "https://www.joelonsoftware.com/feed/",
    "coding horror": "https://blog.codinghorror.com/rss/",
    "david heinemeier hansson": "https://world.hey.com/dhh/feed.atom",
    "robert martin": "https://blog.cleancoder.com/atom.xml",
    "scott hanselman": "https://feeds.hanselman.com/ScottHanselman",
    "bruce eckel": "https://www.bruceeckel.com/feed/",
}


def discover_feed_url(name: str) -> Optional[str]:
    """Try to discover a feed URL for a given blog/site name."""
    # First check known feeds
    name_lower = name.lower()
    for known_name, url in KNOWN_FEEDS.items():
        if known_name in name_lower or name_lower in known_name:
            return url
    
    # Try to search for the feed URL by constructing likely URLs
    search_terms = name.lower().replace(",", "").split()
    
    # Common blog platforms and their feed patterns
    common_patterns = [
        f"https://www.{search_terms[0]}.com/feed/",
        f"https://{search_terms[0]}.com/feed/",
        f"https://blog.{search_terms[0]}.com/rss/",
    ]
    
    for pattern_url in common_patterns:
        try:
            response = requests.head(pattern_url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                return pattern_url
        except Exception:
            continue
    
    return None


def parse_feed_line(line: str) -> Optional[FeedSource]:
    """Parse a single line from feeds.md and extract name/URL."""
    line = line.strip()
    
    # Skip empty lines
    if not line:
        return None
    
    # Pattern 1: URL only (starts with http)
    if line.startswith(('http://', 'https://')):
        # Extract a name from the URL
        name = extract_name_from_url(line)
        return FeedSource(name=name, url=line)
    
    # Pattern 2: Name: URL (e.g., "Bleeping computer: https://...")
    match = re.match(r'^(.+?):\s*(https?://.+)$', line)
    if match:
        return FeedSource(name=match.group(1).strip(), url=match.group(2).strip())
    
    # Pattern 3: Name - URL (e.g., "Ars Technica - https://...")
    match = re.match(r'^(.+?)\s+-\s+(https?://.+)$', line)
    if match:
        return FeedSource(name=match.group(1).strip(), url=match.group(2).strip())
    
    # Pattern 4: Name only (no URL) - try to discover
    discovered_url = discover_feed_url(line)
    return FeedSource(
        name=line.strip(),
        url=discovered_url,
        discovered=discovered_url is not None
    )


def extract_name_from_url(url: str) -> str:
    """Extract a readable name from a feed URL."""
    # Remove protocol
    name = re.sub(r'^https?://(www\.)?', '', url)
    # Remove common feed paths
    name = re.sub(r'/feed.*$|/rss.*$|/atom.*$|\.xml$|\.rss$', '', name, flags=re.IGNORECASE)
    # Remove trailing slashes
    name = name.rstrip('/')
    # Get the domain part
    name = name.split('/')[0]
    # Clean up
    name = name.replace('.com', '').replace('.org', '').replace('.io', '')
    return name.title()


def parse_feeds_file(file_path: str) -> list[FeedSource]:
    """Parse the feeds.md file and return a list of FeedSource objects."""
    feeds = []
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Feeds file not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            feed = parse_feed_line(line)
            if feed:
                feeds.append(feed)
    
    return feeds


def get_feeds_summary(feeds: list[FeedSource]) -> dict:
    """Get a summary of parsed feeds."""
    with_url = [f for f in feeds if f.url]
    without_url = [f for f in feeds if not f.url]
    discovered = [f for f in feeds if f.discovered]
    
    return {
        "total": len(feeds),
        "with_url": len(with_url),
        "without_url": len(without_url),
        "discovered": len(discovered),
        "missing_urls": [f.name for f in without_url],
    }


if __name__ == "__main__":
    # Test the parser
    import sys
    feeds_path = sys.argv[1] if len(sys.argv) > 1 else "feeds.md"
    feeds = parse_feeds_file(feeds_path)
    
    print(f"\nParsed {len(feeds)} feeds:")
    for feed in feeds:
        status = "✓" if feed.url else "✗"
        discovered = " (discovered)" if feed.discovered else ""
        print(f"  {status} {feed.name}: {feed.url or 'NO URL'}{discovered}")
    
    summary = get_feeds_summary(feeds)
    print(f"\nSummary: {summary['with_url']} with URLs, {summary['without_url']} missing")
    if summary['missing_urls']:
        print(f"Missing: {', '.join(summary['missing_urls'])}")
