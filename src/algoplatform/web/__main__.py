"""``python -m algoplatform.web [--host] [--port] [--db-dir]``: the same as ``python -m algoplatform web``."""
import sys

from ..__main__ import run

raise SystemExit(run(["web", *sys.argv[1:]]))
