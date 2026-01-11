# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Quick pipeline: refresh feeds + generate dashboard + open in browser
./run.sh                    # 24h window, 50 entries
./run.sh 48 100             # 48h window, 100 entries

# Individual operations
python main.py refresh              # Fetch and score all feeds
python main.py refresh --no-score   # Fetch without ML scoring (faster)
python main.py list --show-scores   # List entries with HN scores
python main.py top                  # Show top entries by HN score
python main.py dashboard --open     # Generate and open HTML dashboard

# Export
python main.py export-jsonl -o out.jsonl     # One entry per line for ML
python main.py export-sentiment -o out.json  # Cleaned text for NLP
```

## Architecture

### Core Data Flow

```
feeds.csv → FeedParser → Fetcher (ThreadPool) → Storage (SQLite) → HNPredictor → Display/Export
```

### Key Modules in `rss_reader/`

| Module | Responsibility |
|--------|----------------|
| `feed_parser.py` | Parse CSV feeds |
| `fetcher.py` | Concurrent feed fetching with ETag caching |
| `storage.py` | SQLite layer, deduplication by GUID |
| `hn_predictor.py` | RoBERTa-based HN success prediction |
| `hn_checker.py` | Algolia API for HN status |
| `dashboard.py` | HTML dashboard generation |
| `exporter.py` | Multi-format export |

### ML Model (V7)

Located at `rss_reader/models/hn_model_v7/` (download from HuggingFace).

Model: [philippdubach/hn-success-predictor](https://huggingface.co/philippdubach/hn-success-predictor)

- ROC AUC: 0.685
- Calibration ECE: 0.043
- Threshold: 0.302
