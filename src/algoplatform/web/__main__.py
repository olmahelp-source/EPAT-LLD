import argparse

from .server import serve

ap = argparse.ArgumentParser(description="algoplatform web UI")
ap.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1 unless behind TLS + auth proxy)")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--db-dir", default=None, help="persist users and paper sessions here so they survive a restart")
a = ap.parse_args()
from .server import App
serve(a.host, a.port, App(db_dir=a.db_dir))
