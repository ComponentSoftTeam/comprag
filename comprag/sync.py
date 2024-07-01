from __future__ import annotations

import logging
from argparse import Namespace

from asgiref.sync import async_to_sync
from comprag.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def main(args: Namespace):
    dm = DatabaseManager()

    if not args.dry_run:
        dm.sync()

    if args.list:
        dm.print_stats()
