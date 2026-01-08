"""Terminal display using Rich library."""

from datetime import datetime
from html import unescape
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich.style import Style
from rich import box
import re


console = Console()


def strip_html(text: str) -> str:
    """Remove HTML tags from text and decode entities."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities
    text = unescape(text)
    return text


def truncate(text: str, max_length: int = 100) -> str:
    """Truncate text to max length."""
    text = strip_html(text)
    text = ' '.join(text.split())  # Normalize whitespace
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_date(date_str: Optional[str]) -> str:
    """Format a date string for display."""
    if not date_str:
        return "-"
    try:
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            diff = now - dt
            
            if diff.days == 0:
                hours = diff.seconds // 3600
                if hours == 0:
                    minutes = diff.seconds // 60
                    return f"{minutes}m ago"
                return f"{hours}h ago"
            elif diff.days == 1:
                return "Yesterday"
            elif diff.days < 7:
                return f"{diff.days}d ago"
            else:
                return dt.strftime('%b %d')
    except (ValueError, AttributeError):
        pass
    return str(date_str)[:10]


def get_feed_color(feed_name: str) -> str:
    """Get a consistent color for a feed name."""
    colors = [
        "cyan", "green", "yellow", "blue", "magenta", 
        "bright_cyan", "bright_green", "bright_yellow",
        "bright_blue", "bright_magenta",
    ]
    # Use hash to get consistent color
    index = hash(feed_name) % len(colors)
    return colors[index]


def display_entries(entries: list[dict], title: str = "Feed Entries"):
    """Display entries in a formatted table."""
    if not entries:
        console.print("[yellow]No entries to display.[/yellow]")
        return
    
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    
    table.add_column("#", style="dim", width=4)
    table.add_column("Feed", style="bold", width=20)
    table.add_column("Title", width=50)
    table.add_column("Date", width=12)
    table.add_column("Status", width=6)
    
    for i, entry in enumerate(entries, 1):
        feed_name = truncate(entry.get('feed_name', 'Unknown'), 18)
        feed_color = get_feed_color(feed_name)
        title_text = truncate(entry.get('title', 'Untitled'), 48)
        date_str = format_date(entry.get('published'))
        
        # Status indicators
        status = ""
        if entry.get('is_starred'):
            status += "★"
        if not entry.get('is_read'):
            status += "●"
            title_style = "bold"
        else:
            title_style = "dim"
        
        table.add_row(
            str(i),
            Text(feed_name, style=feed_color),
            Text(title_text, style=title_style),
            date_str,
            Text(status, style="yellow"),
        )
    
    console.print(table)


def display_entry_detail(entry: dict):
    """Display a single entry in detail."""
    title = entry.get('title', 'Untitled')
    feed_name = entry.get('feed_name', 'Unknown')
    link = entry.get('link', '')
    published = format_date(entry.get('published'))
    author = entry.get('author', '')
    content = strip_html(entry.get('content', '') or entry.get('summary', ''))
    tags = entry.get('tags', [])
    
    # Header
    console.print()
    console.print(Panel(
        Text(title, style="bold white"),
        title=f"[{get_feed_color(feed_name)}]{feed_name}[/]",
        border_style="cyan",
    ))
    
    # Metadata
    meta = []
    if published:
        meta.append(f"[dim]Published:[/] {published}")
    if author:
        meta.append(f"[dim]Author:[/] {author}")
    if link:
        meta.append(f"[dim]Link:[/] [link={link}]{link}[/link]")
    if tags:
        meta.append(f"[dim]Tags:[/] {', '.join(tags[:5])}")
    
    if meta:
        console.print("  " + "  |  ".join(meta))
    
    # Content
    console.print()
    if content:
        # Word wrap the content
        console.print(Panel(
            content,
            title="Content",
            border_style="dim",
            padding=(1, 2),
        ))
    else:
        console.print("[dim]No content available.[/dim]")
    
    console.print()


def display_feeds_list(feeds: list[dict]):
    """Display list of feeds."""
    if not feeds:
        console.print("[yellow]No feeds found.[/yellow]")
        return
    
    table = Table(
        title="Subscribed Feeds",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("#", style="dim", width=4)
    table.add_column("Feed Name", width=35)
    table.add_column("URL", width=40)
    table.add_column("Entries", justify="right", width=8)
    table.add_column("Unread", justify="right", width=8)
    
    for i, feed in enumerate(feeds, 1):
        name = truncate(feed.get('name', 'Unknown'), 33)
        url = truncate(feed.get('url', ''), 38)
        total = feed.get('total_entries', 0)
        unread = feed.get('unread_count', 0)
        
        unread_style = "bold yellow" if unread else "dim"
        
        table.add_row(
            str(i),
            Text(name, style=get_feed_color(name)),
            Text(url, style="dim"),
            str(total),
            Text(str(unread), style=unread_style),
        )
    
    console.print(table)


def display_stats(stats: dict):
    """Display database statistics."""
    table = Table(
        title="Feed Reader Statistics",
        box=box.ROUNDED,
        show_header=False,
    )
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold white", justify="right")
    
    table.add_row("Total Feeds", str(stats.get('feeds', 0)))
    table.add_row("Total Entries", str(stats.get('entries', 0)))
    table.add_row("Unread Entries", str(stats.get('unread', 0)))
    table.add_row("Starred Entries", str(stats.get('starred', 0)))
    
    console.print(table)


def display_fetch_progress():
    """Create a progress display for fetching feeds."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        console=console,
    )


def display_parse_summary(summary: dict):
    """Display summary of parsed feeds."""
    console.print()
    console.print(Panel(
        f"[bold]Total feeds:[/] {summary['total']}\n"
        f"[green]With URL:[/] {summary['with_url']}\n"
        f"[yellow]Auto-discovered:[/] {summary['discovered']}\n"
        f"[red]Missing URL:[/] {summary['without_url']}",
        title="Feed Parse Summary",
        border_style="cyan",
    ))
    
    if summary['missing_urls']:
        console.print(f"\n[yellow]Feeds without URLs:[/] {', '.join(summary['missing_urls'])}")


def display_fetch_summary(summary: dict):
    """Display summary of fetch results."""
    console.print()
    console.print(Panel(
        f"[bold]Total feeds:[/] {summary['total_feeds']}\n"
        f"[green]Successful:[/] {summary['successful']}\n"
        f"[red]Failed:[/] {summary['failed']}\n"
        f"[cyan]Total entries:[/] {summary['total_entries']}\n"
        f"[dim]Time:[/] {summary['total_time']}s",
        title="Fetch Summary",
        border_style="green",
    ))
    
    if summary['failed_feeds']:
        console.print("\n[red]Failed feeds:[/]")
        for name, error in summary['failed_feeds'][:10]:
            console.print(f"  • {name}: {error}")


def display_error(message: str):
    """Display an error message."""
    console.print(f"[bold red]Error:[/] {message}")


def display_success(message: str):
    """Display a success message."""
    console.print(f"[bold green]✓[/] {message}")


def display_info(message: str):
    """Display an info message."""
    console.print(f"[cyan]ℹ[/] {message}")
