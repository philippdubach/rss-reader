# RSS Feed Reader

A Python-based RSS/Atom feed reader with concurrent fetching, SQLite storage, AI-powered HN success prediction, and export capabilities.

## Quick Start

```bash
cp feeds.example.csv feeds.csv  # Set up your feeds
./run.sh                        # Refresh + dashboard
```

## Features

- **Concurrent fetching**: Fast parallel feed fetching with ETag caching
- **HN Success Prediction**: ML-powered scoring of articles likely to perform well on Hacker News
- **HN Status Checking**: Checks if articles have already been posted to HN
- **Interactive Dashboard**: HTML dashboard with score visualization
- **SQLite storage**: Persistent storage with read/starred state
- **Export formats**: JSON, JSON Lines, HTML

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your feeds
cp feeds.example.csv feeds.csv
# Edit feeds.csv with your favorite RSS feeds

# Download the HN prediction model (~500MB)
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('philippdubach/hn-success-predictor', local_dir='rss_reader/models/hn_model_v7')"
```

## Usage

```bash
# Refresh feeds and score entries
python main.py refresh
python main.py refresh --no-score   # Skip ML scoring (faster)

# List entries
python main.py list
python main.py top                  # Top entries by HN score
python main.py list --search "AI"   # Search

# Generate dashboard
python main.py dashboard --open

# Export
python main.py export-jsonl -o out.jsonl
```

## Project Structure

```
rss-reader/
├── main.py                 # CLI application
├── run.sh                  # Quick-start script
├── feeds.example.csv       # Example feed subscriptions
├── requirements.txt        # Python dependencies
├── rss_reader/
│   ├── feed_parser.py      # Parse CSV feeds
│   ├── fetcher.py          # Concurrent feed fetching
│   ├── storage.py          # SQLite storage layer
│   ├── hn_predictor.py     # RoBERTa-based HN prediction
│   ├── hn_checker.py       # HN status via Algolia API
│   ├── dashboard.py        # HTML dashboard generation
│   ├── exporter.py         # Multi-format export
│   └── models/             # ML model (download separately)
└── docs/
    └── HN_PREDICTOR_RETROSPECTIVE.md
```

## HN Success Predictor

Predicts the probability that an article title would achieve ≥100 points on Hacker News.

| Metric | Value |
|--------|-------|
| Architecture | RoBERTa-base (regularized) |
| Test ROC AUC | 0.685 |
| Calibration (ECE) | 0.043 |
| Model Size | ~500MB |

### Model Download

**[📦 Download from HuggingFace](https://huggingface.co/philippdubach/hn-success-predictor)**

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('philippdubach/hn-success-predictor', local_dir='rss_reader/models/hn_model_v7')"
```

## Feed File Format

Copy `feeds.example.csv` to `feeds.csv` and add your feeds:

```csv
name,url
Hacker News,https://news.ycombinator.com/rss
Ars Technica,https://feeds.arstechnica.com/arstechnica/technology-lab
TechCrunch,https://techcrunch.com/feed/
```

## License

MIT
