#!/usr/bin/env python3
"""RSS Feed Reader - Main CLI Application."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt, Confirm

from rss_reader.feed_parser import parse_feeds_file, get_feeds_summary, FeedSource
from rss_reader.fetcher import fetch_all_feeds, get_fetch_summary
from rss_reader.storage import FeedStorage, FeedCSVManager
from rss_reader.exporter import (
    export_to_json,
    export_to_jsonl,
    export_to_html,
    export_for_sentiment_analysis,
)
from rss_reader.display import (
    console,
    display_entries,
    display_entry_detail,
    display_feeds_list,
    display_stats,
    display_fetch_progress,
    display_parse_summary,
    display_fetch_summary,
    display_error,
    display_success,
    display_info,
)


def get_default_feeds_path() -> Path:
    """Get the default feeds.md path."""
    # Look in current directory first
    local_path = Path("feeds.md")
    if local_path.exists():
        return local_path
    
    # Look in script directory
    script_dir = Path(__file__).parent
    script_path = script_dir / "feeds.md"
    if script_path.exists():
        return script_path
    
    return local_path


def get_default_csv_path() -> Path:
    """Get the default feeds.csv path."""
    local_path = Path("feeds.csv")
    if local_path.exists():
        return local_path
    
    script_dir = Path(__file__).parent
    script_path = script_dir / "feeds.csv"
    if script_path.exists():
        return script_path
    
    return local_path


def cmd_refresh(args, storage: FeedStorage):
    """Refresh all feeds."""
    # Check for CSV feeds first
    csv_path = Path(args.csv) if hasattr(args, 'csv') and args.csv else get_default_csv_path()
    
    if csv_path.exists():
        # Use CSV feeds
        display_info(f"Loading feeds from {csv_path}...")
        csv_manager = FeedCSVManager(str(csv_path))
        csv_feeds = csv_manager.get_all_feeds()
        feeds = [FeedSource(name=f['name'], url=f['url']) for f in csv_feeds]
        
        summary = {
            "total": len(feeds),
            "with_url": len(feeds),
            "without_url": 0,
            "discovered": 0,
            "missing_urls": [],
        }
        display_parse_summary(summary)
    else:
        # Fall back to feeds.md
        feeds_path = Path(args.feeds) if hasattr(args, 'feeds') and args.feeds else get_default_feeds_path()
        
        if not feeds_path.exists():
            display_error(f"No feeds file found! Create feeds.csv or feeds.md")
            return
        
        # Parse feeds file
        display_info(f"Parsing feeds from {feeds_path}...")
        feeds = parse_feeds_file(str(feeds_path))
        summary = get_feeds_summary(feeds)
        display_parse_summary(summary)
    
    # Filter to only feeds with URLs
    feeds_with_urls = [f for f in feeds if f.url]
    
    if not feeds_with_urls:
        display_error("No feeds with valid URLs found!")
        return
    
    # Get cache info for conditional requests
    cache = storage.get_cache_info()
    
    # Fetch feeds with progress
    display_info(f"Fetching {len(feeds_with_urls)} feeds...")
    
    with display_fetch_progress() as progress:
        task = progress.add_task("Fetching feeds...", total=len(feeds_with_urls), status="")
        
        def update_progress(completed, total, result):
            status = f"✓ {result.feed.name}" if result.success else f"✗ {result.feed.name}"
            progress.update(task, completed=completed, status=status[:30])
        
        results = fetch_all_feeds(
            feeds_with_urls,
            max_workers=args.workers,
            timeout=args.timeout,
            cache=cache,
            progress_callback=update_progress,
        )
    
    # Display fetch summary
    fetch_summary = get_fetch_summary(results)
    display_fetch_summary(fetch_summary)
    
    # Save to database
    display_info("Saving to database...")
    save_result = storage.process_fetch_results(results)
    display_success(f"Saved {save_result['new_entries']} new entries, updated {save_result['updated_entries']}")


def cmd_list(args, storage: FeedStorage):
    """List entries."""
    entries = storage.get_entries(
        feed_url=args.feed,
        unread_only=args.unread,
        starred_only=args.starred,
        limit=args.limit,
        search=args.search,
    )
    
    title = "Feed Entries"
    if args.unread:
        title = "Unread Entries"
    elif args.starred:
        title = "Starred Entries"
    elif args.search:
        title = f"Search: {args.search}"
    
    display_entries(entries, title=title)
    
    if entries:
        console.print(f"\n[dim]Showing {len(entries)} entries. Use --limit to see more.[/dim]")


def cmd_read(args, storage: FeedStorage):
    """Read a specific entry."""
    entries = storage.get_entries(limit=args.limit)
    
    if args.number < 1 or args.number > len(entries):
        display_error(f"Invalid entry number. Choose between 1 and {len(entries)}")
        return
    
    entry = entries[args.number - 1]
    display_entry_detail(entry)
    
    # Mark as read
    storage.mark_read(entry['id'])


def cmd_feeds(args, storage: FeedStorage):
    """List all feeds."""
    feeds = storage.get_feeds()
    display_feeds_list(feeds)


def cmd_stats(args, storage: FeedStorage):
    """Show statistics."""
    stats = storage.get_stats()
    display_stats(stats)


def cmd_export_json(args, storage: FeedStorage):
    """Export entries to JSON."""
    entries = storage.get_all_entries_for_export()
    
    if not entries:
        display_error("No entries to export. Run 'refresh' first.")
        return
    
    output_path = args.output or "rss_export.json"
    path = export_to_json(entries, output_path)
    display_success(f"Exported {len(entries)} entries to {path}")


def cmd_export_jsonl(args, storage: FeedStorage):
    """Export entries to JSON Lines format."""
    entries = storage.get_all_entries_for_export()
    
    if not entries:
        display_error("No entries to export. Run 'refresh' first.")
        return
    
    output_path = args.output or "rss_export.jsonl"
    path = export_to_jsonl(entries, output_path)
    display_success(f"Exported {len(entries)} entries to {path}")


def cmd_export_html(args, storage: FeedStorage):
    """Export entries to HTML."""
    entries = storage.get_all_entries_for_export()
    
    if not entries:
        display_error("No entries to export. Run 'refresh' first.")
        return
    
    output_path = args.output or "rss_export.html"
    path = export_to_html(entries, output_path, title=args.title or "RSS Feed Export")
    display_success(f"Exported {len(entries)} entries to {path}")


def cmd_export_sentiment(args, storage: FeedStorage):
    """Export entries for sentiment analysis."""
    entries = storage.get_all_entries_for_export()
    
    if not entries:
        display_error("No entries to export. Run 'refresh' first.")
        return
    
    output_path = args.output or "rss_sentiment.json"
    path = export_for_sentiment_analysis(entries, output_path)
    display_success(f"Exported {len(entries)} entries for sentiment analysis to {path}")


def cmd_mark_read(args, storage: FeedStorage):
    """Mark entries as read."""
    if args.all:
        storage.mark_all_read(feed_url=args.feed)
        display_success("Marked all entries as read")
    else:
        display_error("Specify --all to mark all entries as read")


def cmd_web(args, storage: FeedStorage):
    """Start the web interface."""
    from rss_reader.web_server import run_server
    csv_path = args.csv if hasattr(args, 'csv') and args.csv else str(get_default_csv_path())
    run_server(
        host=args.host,
        port=args.port,
        csv_path=csv_path,
        db_path=args.db,
    )


def cmd_interactive(args, storage: FeedStorage):
    """Interactive mode."""
    console.print("\n[bold cyan]RSS Feed Reader - Interactive Mode[/bold cyan]")
    console.print("[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")
    
    while True:
        try:
            cmd = Prompt.ask("[bold]rss>[/bold]").strip().lower()
            
            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                console.print("""
[bold]Commands:[/bold]
  refresh    - Fetch all feeds
  list       - List recent entries
  unread     - List unread entries
  feeds      - List all feeds
  stats      - Show statistics
  read N     - Read entry number N
  star N     - Star entry number N
  export     - Export to JSON
  html       - Export to HTML
  quit       - Exit
""")
            elif cmd == 'refresh':
                args.feeds = None
                args.workers = 10
                args.timeout = 15
                cmd_refresh(args, storage)
            elif cmd == 'list':
                args.feed = None
                args.unread = False
                args.starred = False
                args.limit = 20
                args.search = None
                cmd_list(args, storage)
            elif cmd == 'unread':
                args.feed = None
                args.unread = True
                args.starred = False
                args.limit = 20
                args.search = None
                cmd_list(args, storage)
            elif cmd == 'feeds':
                cmd_feeds(args, storage)
            elif cmd == 'stats':
                cmd_stats(args, storage)
            elif cmd.startswith('read '):
                try:
                    args.number = int(cmd.split()[1])
                    args.limit = 100
                    cmd_read(args, storage)
                except (ValueError, IndexError):
                    display_error("Usage: read <number>")
            elif cmd.startswith('star '):
                try:
                    num = int(cmd.split()[1])
                    entries = storage.get_entries(limit=100)
                    if 1 <= num <= len(entries):
                        storage.mark_starred(entries[num-1]['id'])
                        display_success(f"Starred entry {num}")
                    else:
                        display_error("Invalid entry number")
                except (ValueError, IndexError):
                    display_error("Usage: star <number>")
            elif cmd == 'export':
                args.output = None
                cmd_export_json(args, storage)
            elif cmd == 'html':
                args.output = None
                args.title = None
                cmd_export_html(args, storage)
            elif cmd:
                display_error(f"Unknown command: {cmd}")
        
        except KeyboardInterrupt:
            console.print("\n")
            continue
        except EOFError:
            break
    
    console.print("\n[dim]Goodbye![/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RSS Feed Reader - Fetch, store, and export RSS feeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--db', '-d',
        default='rss_reader.db',
        help='Database file path (default: rss_reader.db)',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Refresh command
    refresh_parser = subparsers.add_parser('refresh', help='Fetch all feeds')
    refresh_parser.add_argument('--feeds', '-f', help='Path to feeds.md file')
    refresh_parser.add_argument('--csv', '-c', help='Path to feeds.csv file')
    refresh_parser.add_argument('--workers', '-w', type=int, default=10, help='Number of worker threads')
    refresh_parser.add_argument('--timeout', '-t', type=int, default=15, help='Request timeout in seconds')
    
    # Web server command
    web_parser = subparsers.add_parser('web', help='Start web interface for managing feeds')
    web_parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    web_parser.add_argument('--port', '-p', type=int, default=8080, help='Port to listen on (default: 8080)')
    web_parser.add_argument('--csv', '-c', help='Path to feeds.csv file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List entries')
    list_parser.add_argument('--feed', help='Filter by feed URL')
    list_parser.add_argument('--unread', '-u', action='store_true', help='Show only unread')
    list_parser.add_argument('--starred', '-s', action='store_true', help='Show only starred')
    list_parser.add_argument('--limit', '-l', type=int, default=20, help='Maximum entries to show')
    list_parser.add_argument('--search', help='Search in title/content')
    
    # Read command
    read_parser = subparsers.add_parser('read', help='Read a specific entry')
    read_parser.add_argument('number', type=int, help='Entry number from list')
    read_parser.add_argument('--limit', '-l', type=int, default=100, help='Maximum entries to consider')
    
    # Feeds command
    subparsers.add_parser('feeds', help='List all feeds')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # Export commands
    json_parser = subparsers.add_parser('export-json', help='Export to JSON')
    json_parser.add_argument('--output', '-o', help='Output file path')
    
    jsonl_parser = subparsers.add_parser('export-jsonl', help='Export to JSON Lines')
    jsonl_parser.add_argument('--output', '-o', help='Output file path')
    
    html_parser = subparsers.add_parser('export-html', help='Export to HTML')
    html_parser.add_argument('--output', '-o', help='Output file path')
    html_parser.add_argument('--title', help='HTML page title')
    
    sentiment_parser = subparsers.add_parser('export-sentiment', help='Export for sentiment analysis')
    sentiment_parser.add_argument('--output', '-o', help='Output file path')
    
    # Mark read command
    mark_parser = subparsers.add_parser('mark-read', help='Mark entries as read')
    mark_parser.add_argument('--all', '-a', action='store_true', help='Mark all as read')
    mark_parser.add_argument('--feed', help='Mark all in specific feed as read')
    
    # Interactive mode
    subparsers.add_parser('interactive', aliases=['i'], help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize storage
    storage = FeedStorage(args.db)
    
    # Execute command
    if args.command == 'refresh':
        cmd_refresh(args, storage)
    elif args.command == 'list':
        cmd_list(args, storage)
    elif args.command == 'read':
        cmd_read(args, storage)
    elif args.command == 'feeds':
        cmd_feeds(args, storage)
    elif args.command == 'stats':
        cmd_stats(args, storage)
    elif args.command == 'export-json':
        cmd_export_json(args, storage)
    elif args.command == 'export-jsonl':
        cmd_export_jsonl(args, storage)
    elif args.command == 'export-html':
        cmd_export_html(args, storage)
    elif args.command == 'export-sentiment':
        cmd_export_sentiment(args, storage)
    elif args.command == 'mark-read':
        cmd_mark_read(args, storage)
    elif args.command == 'web':
        cmd_web(args, storage)
    elif args.command in ('interactive', 'i'):
        cmd_interactive(args, storage)
    else:
        # Default: show help or run interactive
        if not args.command:
            parser.print_help()
            console.print("\n[dim]Run 'python main.py refresh' to fetch feeds[/dim]")


if __name__ == "__main__":
    main()
