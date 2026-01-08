"""Export functionality for JSON and HTML formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from html import escape


def export_to_json(
    entries: list[dict],
    output_path: str,
    pretty: bool = True,
) -> str:
    """Export entries to JSON format for sentiment analysis."""
    # Prepare export data with metadata
    export_data = {
        "export_date": datetime.now().isoformat(),
        "total_entries": len(entries),
        "entries": entries,
    }
    
    # Add summary statistics
    feeds = set(e.get('feed_name', 'Unknown') for e in entries)
    export_data["metadata"] = {
        "unique_feeds": len(feeds),
        "feeds": list(feeds),
        "date_range": {
            "earliest": min((e.get('published') for e in entries if e.get('published')), default=None),
            "latest": max((e.get('published') for e in entries if e.get('published')), default=None),
        }
    }
    
    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        else:
            json.dump(export_data, f, ensure_ascii=False, default=str)
    
    return str(path.absolute())


def export_to_jsonl(entries: list[dict], output_path: str) -> str:
    """Export entries to JSON Lines format (one JSON object per line).
    
    This format is ideal for sentiment analysis pipelines that process
    entries one at a time. Text is cleaned of HTML for ML processing.
    """
    from html import unescape
    import re
    
    def clean_html(text: str) -> str:
        """Remove HTML tags and clean text."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            content = entry.get('content') or entry.get('summary', '')
            cleaned_text = clean_html(content)
            
            # Create a flat structure optimized for ML processing
            flat_entry = {
                "id": entry.get('id'),
                "feed": entry.get('feed_name'),
                "title": clean_html(entry.get('title', '')),
                "text": cleaned_text,
                "link": entry.get('link'),
                "published": entry.get('published'),
                "author": entry.get('author'),
                "tags": entry.get('tags', []),
                "word_count": len(cleaned_text.split()) if cleaned_text else 0,
            }
            f.write(json.dumps(flat_entry, ensure_ascii=False, default=str) + '\n')
    
    return str(path.absolute())


def export_to_html(
    entries: list[dict],
    output_path: str,
    title: str = "RSS Feed Export",
) -> str:
    """Export entries to HTML format for reading."""
    
    # Group entries by feed
    feeds_dict = {}
    for entry in entries:
        feed_name = entry.get('feed_name', 'Unknown')
        if feed_name not in feeds_dict:
            feeds_dict[feed_name] = []
        feeds_dict[feed_name].append(entry)
    
    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape(title)}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --text-muted: #a0a0a0;
            --accent: #e94560;
            --link-color: #4da8ff;
            --border-color: #0f3460;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        
        h1 {{
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
        }}
        
        h2 {{
            color: var(--link-color);
            margin-top: 40px;
            padding: 10px;
            background: var(--border-color);
            border-radius: 5px;
        }}
        
        .stats {{
            background: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        
        .entry {{
            background: var(--card-bg);
            border-left: 4px solid var(--accent);
            margin: 15px 0;
            padding: 15px;
            border-radius: 0 8px 8px 0;
        }}
        
        .entry-title {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        
        .entry-title a {{
            color: var(--text-color);
            text-decoration: none;
        }}
        
        .entry-title a:hover {{
            color: var(--link-color);
        }}
        
        .entry-meta {{
            color: var(--text-muted);
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        
        .entry-summary {{
            color: var(--text-color);
            line-height: 1.7;
        }}
        
        .entry-summary img {{
            max-width: 100%;
            height: auto;
        }}
        
        .tags {{
            margin-top: 10px;
        }}
        
        .tag {{
            display: inline-block;
            background: var(--border-color);
            color: var(--link-color);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-right: 5px;
        }}
        
        .toc {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .toc h3 {{
            margin-top: 0;
            color: var(--accent);
        }}
        
        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .toc li {{
            margin: 8px 0;
        }}
        
        .toc a {{
            color: var(--link-color);
            text-decoration: none;
        }}
        
        .toc a:hover {{
            text-decoration: underline;
        }}
        
        .count {{
            color: var(--text-muted);
            font-size: 0.9em;
        }}
        
        @media (prefers-color-scheme: light) {{
            :root {{
                --bg-color: #f5f5f5;
                --card-bg: #ffffff;
                --text-color: #333;
                --text-muted: #666;
                --accent: #e94560;
                --link-color: #0066cc;
                --border-color: #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{escape(title)}</h1>
        
        <div class="stats">
            <strong>Export Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Total Entries:</strong> {len(entries)}<br>
            <strong>Feeds:</strong> {len(feeds_dict)}
        </div>
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
"""]
    
    # Add TOC entries
    for feed_name, feed_entries in sorted(feeds_dict.items()):
        safe_id = escape(feed_name.replace(' ', '-').lower())
        html_parts.append(f'                <li><a href="#{safe_id}">{escape(feed_name)}</a> <span class="count">({len(feed_entries)} entries)</span></li>\n')
    
    html_parts.append("            </ul>\n        </div>\n")
    
    # Add entries by feed
    for feed_name, feed_entries in sorted(feeds_dict.items()):
        safe_id = escape(feed_name.replace(' ', '-').lower())
        html_parts.append(f'        <h2 id="{safe_id}">{escape(feed_name)}</h2>\n')
        
        for entry in feed_entries:
            title = entry.get('title', 'Untitled')
            link = entry.get('link', '#')
            published = entry.get('published', '')
            author = entry.get('author', '')
            summary = entry.get('summary', '')
            tags = entry.get('tags', [])
            
            # Format date
            date_str = ''
            if published:
                try:
                    if isinstance(published, str):
                        dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d %H:%M')
                except (ValueError, AttributeError):
                    date_str = str(published)
            
            meta_parts = []
            if date_str:
                meta_parts.append(date_str)
            if author:
                meta_parts.append(f"by {escape(author)}")
            
            html_parts.append(f"""        <article class="entry">
            <h3 class="entry-title"><a href="{escape(link)}" target="_blank">{escape(title)}</a></h3>
            <div class="entry-meta">{' | '.join(meta_parts)}</div>
            <div class="entry-summary">{summary}</div>
""")
            
            if tags:
                html_parts.append('            <div class="tags">')
                for tag in tags[:5]:  # Limit to 5 tags
                    html_parts.append(f'<span class="tag">{escape(tag)}</span>')
                html_parts.append('</div>\n')
            
            html_parts.append("        </article>\n")
    
    html_parts.append("""    </div>
</body>
</html>
""")
    
    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))
    
    return str(path.absolute())


def export_for_sentiment_analysis(
    entries: list[dict],
    output_path: str,
    include_content: bool = True,
) -> str:
    """Export entries in a format optimized for sentiment analysis.
    
    Creates a JSON file with cleaned text data ready for NLP processing.
    """
    from html import unescape
    import re
    
    def clean_html(text: str) -> str:
        """Remove HTML tags and clean text."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode HTML entities
        text = unescape(text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned_entries = []
    for entry in entries:
        # Get the best text content available
        content = entry.get('content', '') or entry.get('summary', '')
        cleaned_content = clean_html(content)
        cleaned_title = clean_html(entry.get('title', ''))
        cleaned_summary = clean_html(entry.get('summary', ''))
        
        cleaned_entry = {
            "id": entry.get('id'),
            "feed_name": entry.get('feed_name'),
            "title": cleaned_title,
            "text": cleaned_content if include_content else cleaned_summary,
            "link": entry.get('link'),
            "published": entry.get('published'),
            "author": entry.get('author'),
            "tags": entry.get('tags', []),
            # Additional metadata for analysis
            "word_count": len(cleaned_content.split()) if cleaned_content else 0,
            "has_content": bool(cleaned_content),
        }
        cleaned_entries.append(cleaned_entry)
    
    export_data = {
        "export_type": "sentiment_analysis",
        "export_date": datetime.now().isoformat(),
        "total_entries": len(cleaned_entries),
        "entries": cleaned_entries,
    }
    
    path = Path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    return str(path.absolute())
