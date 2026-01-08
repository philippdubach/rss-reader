"""Web server for RSS Feed Reader management interface."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from pathlib import Path

from .storage import FeedCSVManager, FeedStorage


class FeedManagerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for feed management."""
    
    csv_manager: FeedCSVManager = None
    storage: FeedStorage = None
    
    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[WebServer] {args[0]}")
    
    def send_json_response(self, data: dict, status: int = 200):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_html_response(self, html: str, status: int = 200):
        """Send an HTML response."""
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/' or parsed.path == '/index.html':
            self.serve_main_page()
        elif parsed.path == '/api/feeds':
            self.get_feeds_api()
        elif parsed.path == '/api/stats':
            self.get_stats_api()
        else:
            self.send_error(404, 'Not Found')
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/feeds':
            self.add_feed_api()
        elif parsed.path == '/api/feeds/delete':
            self.delete_feed_api()
        else:
            self.send_error(404, 'Not Found')
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def get_feeds_api(self):
        """API: Get all feeds."""
        feeds = self.csv_manager.get_all_feeds()
        self.send_json_response({'feeds': feeds, 'count': len(feeds)})
    
    def get_stats_api(self):
        """API: Get statistics."""
        stats = self.storage.get_stats()
        self.send_json_response(stats)
    
    def add_feed_api(self):
        """API: Add a new feed."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body)
            name = data.get('name', '').strip()
            url = data.get('url', '').strip()
            
            if not name or not url:
                self.send_json_response({'error': 'Name and URL are required'}, 400)
                return
            
            if not url.startswith(('http://', 'https://')):
                self.send_json_response({'error': 'URL must start with http:// or https://'}, 400)
                return
            
            if self.csv_manager.add_feed(name, url):
                self.send_json_response({'success': True, 'message': f'Feed "{name}" added successfully'})
            else:
                self.send_json_response({'error': 'Feed with this URL already exists'}, 409)
        
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, 400)
    
    def delete_feed_api(self):
        """API: Delete a feed."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body)
            url = data.get('url', '').strip()
            
            if not url:
                self.send_json_response({'error': 'URL is required'}, 400)
                return
            
            if self.csv_manager.remove_feed(url):
                self.send_json_response({'success': True, 'message': 'Feed removed successfully'})
            else:
                self.send_json_response({'error': 'Feed not found'}, 404)
        
        except json.JSONDecodeError:
            self.send_json_response({'error': 'Invalid JSON'}, 400)
    
    def serve_main_page(self):
        """Serve the main HTML page."""
        html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RSS Feed Reader - Manage Feeds</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #71717a;
            margin-bottom: 2rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: #a1a1aa;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #a1a1aa;
            font-size: 0.875rem;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem 1rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 0.5rem;
            background: rgba(0, 0, 0, 0.2);
            color: #e4e4e7;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #60a5fa;
            box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.2);
        }
        .form-group input::placeholder {
            color: #52525b;
        }
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        .btn-danger {
            background: #ef4444;
            color: white;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
        .btn-danger:hover {
            background: #dc2626;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }
        .stat-item {
            text-align: center;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 0.5rem;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #60a5fa;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .feed-list {
            max-height: 500px;
            overflow-y: auto;
        }
        .feed-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            transition: background 0.2s;
        }
        .feed-item:hover {
            background: rgba(255, 255, 255, 0.02);
        }
        .feed-item:last-child {
            border-bottom: none;
        }
        .feed-info {
            flex: 1;
            min-width: 0;
        }
        .feed-name {
            font-weight: 500;
            color: #e4e4e7;
            margin-bottom: 0.25rem;
        }
        .feed-url {
            font-size: 0.75rem;
            color: #52525b;
            word-break: break-all;
        }
        .message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: none;
        }
        .message.success {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #86efac;
        }
        .message.error {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
        }
        .feed-count {
            color: #71717a;
            font-size: 0.875rem;
        }
        .loading {
            text-align: center;
            padding: 2rem;
            color: #71717a;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .spinner {
            display: inline-block;
            width: 1.5rem;
            height: 1.5rem;
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-top-color: #60a5fa;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📡 RSS Feed Reader</h1>
        <p class="subtitle">Manage your RSS feed subscriptions</p>
        
        <div id="message" class="message"></div>
        
        <div class="card">
            <h2>📊 Statistics</h2>
            <div class="stats-grid" id="stats">
                <div class="loading"><div class="spinner"></div></div>
            </div>
        </div>
        
        <div class="card">
            <h2>➕ Add New Feed</h2>
            <form id="addFeedForm">
                <div class="form-group">
                    <label for="feedName">Feed Name</label>
                    <input type="text" id="feedName" placeholder="e.g., Hacker News" required>
                </div>
                <div class="form-group">
                    <label for="feedUrl">Feed URL</label>
                    <input type="url" id="feedUrl" placeholder="e.g., https://news.ycombinator.com/rss" required>
                </div>
                <button type="submit" class="btn btn-primary">Add Feed</button>
            </form>
        </div>
        
        <div class="card">
            <h2>📋 Your Feeds <span class="feed-count" id="feedCount"></span></h2>
            <div class="feed-list" id="feedList">
                <div class="loading"><div class="spinner"></div></div>
            </div>
        </div>
    </div>
    
    <script>
        const API_BASE = '';
        
        function showMessage(text, type) {
            const msg = document.getElementById('message');
            msg.textContent = text;
            msg.className = 'message ' + type;
            msg.style.display = 'block';
            setTimeout(() => { msg.style.display = 'none'; }, 5000);
        }
        
        async function loadStats() {
            try {
                const res = await fetch(API_BASE + '/api/stats');
                const data = await res.json();
                document.getElementById('stats').innerHTML = `
                    <div class="stat-item">
                        <div class="stat-value">${data.feeds || 0}</div>
                        <div class="stat-label">Feeds</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.entries || 0}</div>
                        <div class="stat-label">Entries</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.unread || 0}</div>
                        <div class="stat-label">Unread</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${data.starred || 0}</div>
                        <div class="stat-label">Starred</div>
                    </div>
                `;
            } catch (err) {
                document.getElementById('stats').innerHTML = '<p>Failed to load stats</p>';
            }
        }
        
        async function loadFeeds() {
            try {
                const res = await fetch(API_BASE + '/api/feeds');
                const data = await res.json();
                const feedList = document.getElementById('feedList');
                document.getElementById('feedCount').textContent = `(${data.count})`;
                
                if (data.feeds.length === 0) {
                    feedList.innerHTML = '<p style="padding: 1rem; color: #71717a;">No feeds yet. Add one above!</p>';
                    return;
                }
                
                feedList.innerHTML = data.feeds.map(feed => `
                    <div class="feed-item">
                        <div class="feed-info">
                            <div class="feed-name">${escapeHtml(feed.name)}</div>
                            <div class="feed-url">${escapeHtml(feed.url)}</div>
                        </div>
                        <button class="btn btn-danger" onclick="deleteFeed('${escapeHtml(feed.url)}')">Remove</button>
                    </div>
                `).join('');
            } catch (err) {
                document.getElementById('feedList').innerHTML = '<p>Failed to load feeds</p>';
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function deleteFeed(url) {
            if (!confirm('Remove this feed?')) return;
            
            try {
                const res = await fetch(API_BASE + '/api/feeds/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await res.json();
                
                if (data.success) {
                    showMessage('Feed removed successfully', 'success');
                    loadFeeds();
                } else {
                    showMessage(data.error || 'Failed to remove feed', 'error');
                }
            } catch (err) {
                showMessage('Error removing feed', 'error');
            }
        }
        
        document.getElementById('addFeedForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const name = document.getElementById('feedName').value.trim();
            const url = document.getElementById('feedUrl').value.trim();
            
            try {
                const res = await fetch(API_BASE + '/api/feeds', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, url })
                });
                const data = await res.json();
                
                if (data.success) {
                    showMessage(data.message, 'success');
                    document.getElementById('feedName').value = '';
                    document.getElementById('feedUrl').value = '';
                    loadFeeds();
                } else {
                    showMessage(data.error || 'Failed to add feed', 'error');
                }
            } catch (err) {
                showMessage('Error adding feed', 'error');
            }
        });
        
        // Initial load
        loadStats();
        loadFeeds();
    </script>
</body>
</html>'''
        self.send_html_response(html)


def create_handler(csv_manager: FeedCSVManager, storage: FeedStorage):
    """Create a handler class with the managers attached."""
    class Handler(FeedManagerHandler):
        pass
    Handler.csv_manager = csv_manager
    Handler.storage = storage
    return Handler


def run_server(host: str = 'localhost', port: int = 8080, 
               csv_path: str = 'feeds.csv', db_path: str = 'rss_reader.db'):
    """Run the web server."""
    csv_manager = FeedCSVManager(csv_path)
    storage = FeedStorage(db_path)
    
    handler = create_handler(csv_manager, storage)
    server = HTTPServer((host, port), handler)
    
    print(f"🌐 RSS Feed Reader Web Interface")
    print(f"   Server running at http://{host}:{port}")
    print(f"   Press Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        server.shutdown()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RSS Feed Reader Web Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--csv', default='feeds.csv', help='Path to feeds CSV file')
    parser.add_argument('--db', default='rss_reader.db', help='Path to database file')
    args = parser.parse_args()
    
    run_server(args.host, args.port, args.csv, args.db)
