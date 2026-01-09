"""Generate HTML dashboard for top RSS entries with HN status."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# HTML template for the dashboard
DASHBOARD_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top Stories</title>
    <style>
        :root {
            --bg: #fff;
            --bg-card: #f9f9f9;
            --text: #222;
            --text-dim: #888;
            --green: #16a34a;
            --yellow: #ca8a04;
            --orange: #ea580c;
            --border: #e5e5e5;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            font-size: 14px;
            line-height: 1.4;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        header {
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }
        
        h1 {
            font-size: 1rem;
            font-weight: 600;
        }
        
        .stats {
            display: flex;
            gap: 1.5rem;
            font-size: 0.8rem;
            color: var(--text-dim);
        }
        
        .stat b { color: var(--text); }
        
        .entries { display: flex; flex-direction: column; gap: 2px; }
        
        .entry {
            display: grid;
            grid-template-columns: 42px 1fr auto;
            gap: 0.75rem;
            padding: 0.6rem 0.75rem;
            background: var(--bg-card);
            border-radius: 4px;
            align-items: center;
        }
        
        .entry:hover { background: #f0f0f0; }
        
        .score {
            font-weight: 600;
            font-size: 0.85rem;
            text-align: center;
        }
        
        .score-high { color: var(--green); }
        .score-med { color: var(--yellow); }
        .score-low { color: var(--text-dim); }
        
        .content { min-width: 0; }
        
        .title {
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .title a {
            color: var(--text);
            text-decoration: none;
        }
        
        .title a:hover { color: #000; }
        
        .meta {
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 2px;
        }
        
        .feed { color: #666; }
        
        .hn {
            font-size: 0.75rem;
            white-space: nowrap;
        }
        
        .hn a {
            text-decoration: none;
        }
        
        .hn-posted a {
            color: var(--orange);
        }
        
        .hn-new a {
            color: var(--green);
        }
        
        footer {
            margin-top: 1.5rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border);
            text-align: center;
            color: var(--text-dim);
            font-size: 0.7rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Top Stories ({{hours}}h)</h1>
            <div class="stats">
                <span><b>{{total_entries}}</b> entries</span>
                <span><b>{{already_posted}}</b> on HN</span>
                <span><b>{{not_posted}}</b> new</span>
            </div>
        </header>
        
        <div class="entries">
{{entries_html}}
        </div>
        
        <footer>{{generated_at}}</footer>
    </div>
</body>
</html>
'''

ENTRY_TEMPLATE = '''            <div class="entry">
                <div class="score {{score_class}}">{{score_pct}}</div>
                <div class="content">
                    <div class="title"><a href="{{link}}" target="_blank">{{title}}</a></div>
                    <div class="meta"><span class="feed">{{feed_name}}</span> · {{published}}</div>
                </div>
                <div class="hn {{hn_class}}">{{hn_html}}</div>
            </div>'''


def get_score_class(score: float) -> str:
    """Get CSS class for score."""
    if score >= 0.5:
        return "score-high"
    elif score >= 0.3:
        return "score-med"
    return "score-low"


def format_relative_time(dt_str: str) -> str:
    """Format datetime as relative time."""
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt
        
        if diff.days == 0:
            hours = diff.seconds // 3600
            if hours == 0:
                minutes = diff.seconds // 60
                return f"{minutes}m"
            return f"{hours}h"
        elif diff.days < 7:
            return f"{diff.days}d"
        else:
            return dt.strftime('%b %d')
    except:
        return str(dt_str)[:10] if dt_str else ""


def generate_hn_html(hn_info: Optional[dict], link: str) -> tuple[str, str]:
    """Generate HTML for HN status. Returns (html, css_class)."""
    if hn_info:
        points = hn_info.get('points', 0)
        hn_id = hn_info.get('id', '')
        hn_url = hn_info.get('hn_url', '')
        return f'<a href="{hn_url}" target="_blank">#{hn_id}</a> {points}pts', "hn-posted"
    else:
        submit_url = f"https://news.ycombinator.com/submitlink?u={quote(link)}" if link else "#"
        return f'<a href="{submit_url}" target="_blank">submit</a>', "hn-new"


def quote(url: str) -> str:
    """URL-encode a string."""
    from urllib.parse import quote as url_quote
    return url_quote(url, safe='')


def generate_dashboard(
    entries: list[dict],
    hn_results: dict[str, Optional[dict]],
    hours: int = 24,
    output_path: str = "dashboard.html",
) -> Path:
    """Generate HTML dashboard.
    
    Args:
        entries: List of entry dicts with hn_score
        hn_results: Dict mapping URL to HN submission info
        hours: Hours window for the dashboard
        output_path: Output file path
        
    Returns:
        Path to generated file
    """
    entries_html_parts = []
    already_posted = 0
    not_posted = 0
    
    for entry in entries:
        score = entry.get('hn_score', 0) or 0
        link = entry.get('link', '')
        title = entry.get('title', 'Untitled')
        feed_name = entry.get('feed_name', 'Unknown')
        published = format_relative_time(entry.get('published'))
        
        hn_info = hn_results.get(link)
        if hn_info:
            already_posted += 1
        else:
            not_posted += 1
        
        hn_html, hn_class = generate_hn_html(hn_info, link)
        
        entry_html = ENTRY_TEMPLATE.replace('{{score_pct}}', f"{score*100:.0f}%")
        entry_html = entry_html.replace('{{score_class}}', get_score_class(score))
        entry_html = entry_html.replace('{{title}}', title.replace('<', '&lt;').replace('>', '&gt;'))
        entry_html = entry_html.replace('{{link}}', link)
        entry_html = entry_html.replace('{{feed_name}}', feed_name.replace('<', '&lt;').replace('>', '&gt;'))
        entry_html = entry_html.replace('{{published}}', published)
        entry_html = entry_html.replace('{{hn_html}}', hn_html)
        entry_html = entry_html.replace('{{hn_class}}', hn_class)
        
        entries_html_parts.append(entry_html)
    
    # Generate full HTML
    html = DASHBOARD_TEMPLATE.replace('{{hours}}', str(hours))
    html = html.replace('{{total_entries}}', str(len(entries)))
    html = html.replace('{{already_posted}}', str(already_posted))
    html = html.replace('{{not_posted}}', str(not_posted))
    html = html.replace('{{entries_html}}', '\n'.join(entries_html_parts))
    html = html.replace('{{generated_at}}', datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    # Write to file
    output = Path(output_path)
    output.write_text(html, encoding='utf-8')
    
    return output
