# RSS Feed Reader

A Python-based RSS/Atom feed reader with concurrent fetching, SQLite storage, and export capabilities for sentiment analysis.

## Features

- **Multi-format feed parsing**: Parses `feeds.md` with various formats (Name - URL, Name: URL, URL-only, Name-only)
- **Auto-discovery**: Automatically discovers feed URLs for popular blogs without explicit URLs
- **Concurrent fetching**: Uses ThreadPoolExecutor for fast parallel feed fetching
- **Smart caching**: Stores ETags and Last-Modified headers to avoid re-downloading unchanged feeds
- **SQLite storage**: Persistent storage with read/starred state, deduplication by GUID
- **Rich terminal display**: Color-coded feeds, formatted tables, and progress indicators
- **Export formats**: JSON, JSON Lines, HTML, and sentiment-analysis-optimized JSON

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Refresh Feeds

Fetch all feeds from `feeds.md`:

```bash
python main.py refresh
```

Options:
- `--feeds/-f`: Path to feeds file (default: feeds.md)
- `--workers/-w`: Number of worker threads (default: 10)
- `--timeout/-t`: Request timeout in seconds (default: 15)

### List Entries

```bash
# List recent entries
python main.py list

# List unread entries only
python main.py list --unread

# List starred entries
python main.py list --starred

# Search entries
python main.py list --search "AI"

# Limit results
python main.py list --limit 50
```

### Read an Entry

```bash
# Read entry #3 from the list
python main.py read 3
```

### View Feeds & Stats

```bash
# List all subscribed feeds
python main.py feeds

# Show statistics
python main.py stats
```

### Export for Sentiment Analysis

```bash
# Export to JSON (with metadata)
python main.py export-json -o my_export.json

# Export to JSON Lines (one entry per line, cleaned text)
python main.py export-jsonl -o feed_data.jsonl

# Export optimized for sentiment analysis (cleaned text, word counts)
python main.py export-sentiment -o sentiment_data.json

# Export to HTML for reading
python main.py export-html -o feed_export.html --title "My Feeds"
```

### Interactive Mode

```bash
python main.py interactive
# or
python main.py i
```

Commands in interactive mode:
- `refresh` - Fetch all feeds
- `list` - List recent entries
- `unread` - List unread entries
- `feeds` - List all feeds
- `stats` - Show statistics
- `read N` - Read entry number N
- `star N` - Star entry number N
- `export` - Export to JSON
- `html` - Export to HTML
- `quit` - Exit

### Mark Entries as Read

```bash
# Mark all entries as read
python main.py mark-read --all

# Mark all entries in a specific feed as read
python main.py mark-read --all --feed "https://example.com/feed"
```

## Feed File Format

The `feeds.md` file supports multiple formats:

```markdown
# With name and URL
Ars Technica - https://feeds.arstechnica.com/arstechnica/technology-lab
Bleeping computer: https://www.bleepingcomputer.com/feed/

# URL only (name extracted from URL)
https://hackaday.com/blog/feed/

# Name only (auto-discovered)
Joel on Software
Coding Horror
```

## Export Formats

### JSON (`export-json`)
Full export with metadata, suitable for archival:
```json
{
  "export_date": "2026-01-08T...",
  "total_entries": 2396,
  "metadata": { "unique_feeds": 41, ... },
  "entries": [...]
}
```

### JSON Lines (`export-jsonl`)
One cleaned entry per line, ideal for ML pipelines:
```json
{"id": "...", "feed": "TechCrunch", "title": "...", "text": "cleaned text...", "word_count": 150}
```

### Sentiment Analysis (`export-sentiment`)
Optimized for NLP processing with cleaned HTML and word counts:
```json
{
  "export_type": "sentiment_analysis",
  "entries": [{
    "id": "...",
    "title": "cleaned title",
    "text": "cleaned content without HTML",
    "word_count": 150,
    "has_content": true
  }]
}
```

### HTML (`export-html`)
Readable HTML with dark theme, table of contents, grouped by feed.

## Database

Data is stored in `rss_reader.db` (SQLite). Use `--db` to specify a different location:

```bash
python main.py --db /path/to/my.db refresh
```

## Project Structure

```
rss-reader/
├── feeds.md              # Your feed subscriptions
├── main.py               # CLI application
├── requirements.txt      # Python dependencies
├── rss_reader.db         # SQLite database (created on first run)
└── rss_reader/
    ├── __init__.py
    ├── feed_parser.py    # Parse feeds.md and auto-discover URLs
    ├── fetcher.py        # Concurrent feed fetching with feedparser
    ├── storage.py        # SQLite storage layer
    ├── exporter.py       # JSON/HTML export functions
    └── display.py        # Rich terminal display
```

## License

MIT
