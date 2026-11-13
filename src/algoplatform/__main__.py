"""``python -m algoplatform <command>`` and the installed ``algoplatform`` command.

The command-line interface lives in ``app.cli``. Only the ``web`` command needs the web layer, which the application
layer may not import, so this top-level launcher (outside every job group) supplies it."""
from __future__ import annotations

import argparse
from typing import List, Optional

from .application.app.cli import main


def start_web(args: argparse.Namespace) -> int:
    from .application.web.server import App, serve                       # imported only when the web UI is started
    serve(args.host, args.port, App(db_dir=args.db_dir))
    return 0


def run(argv: Optional[List[str]] = None) -> int:
    return main(argv, web_runner=start_web)


if __name__ == "__main__":
    raise SystemExit(run())
