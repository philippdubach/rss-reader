"""Check if URLs have been posted to Hacker News."""

import logging
import time
from typing import Optional
from urllib.parse import urlparse, quote
import requests

logger = logging.getLogger(__name__)

# HN Algolia Search API
HN_SEARCH_API = "https://hn.algolia.com/api/v1/search"


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (remove trailing slashes, www, etc.)."""
    if not url:
        return ""
    parsed = urlparse(url)
    # Remove www. prefix
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    # Remove trailing slash from path
    path = parsed.path.rstrip("/")
    return f"{host}{path}"


def check_hn_submission(url: str, delay: float = 0.1) -> Optional[dict]:
    """Check if a URL has been submitted to HN.
    
    Args:
        url: The URL to check
        delay: Delay between API calls (be nice to the API)
        
    Returns:
        Dict with HN submission info if found, None otherwise
        {
            'id': HN story ID,
            'title': HN submission title,
            'points': Current points,
            'num_comments': Number of comments,
            'created_at': Submission timestamp,
            'url': 'https://news.ycombinator.com/item?id=...'
        }
    """
    if not url:
        return None
    
    try:
        # Search by URL
        params = {
            "query": url,
            "restrictSearchableAttributes": "url",
            "hitsPerPage": 5,
        }
        
        time.sleep(delay)  # Rate limiting
        response = requests.get(HN_SEARCH_API, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        hits = data.get("hits", [])
        
        if not hits:
            return None
        
        # Find exact or close match
        normalized_target = normalize_url(url)
        
        for hit in hits:
            hit_url = hit.get("url", "")
            if normalize_url(hit_url) == normalized_target:
                return {
                    "id": hit.get("objectID"),
                    "title": hit.get("title"),
                    "points": hit.get("points", 0),
                    "num_comments": hit.get("num_comments", 0),
                    "created_at": hit.get("created_at"),
                    "hn_url": f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                }
        
        return None
        
    except requests.RequestException as e:
        logger.warning(f"HN API error for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error checking HN for {url}: {e}")
        return None


def check_hn_batch(urls: list[str], progress_callback=None) -> dict[str, Optional[dict]]:
    """Check multiple URLs for HN submissions.
    
    Args:
        urls: List of URLs to check
        progress_callback: Optional callback(completed, total) for progress
        
    Returns:
        Dict mapping URL to HN submission info (or None)
    """
    results = {}
    total = len(urls)
    
    for i, url in enumerate(urls):
        results[url] = check_hn_submission(url)
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results
