from __future__ import annotations

import logging
from argparse import Namespace

from asgiref.sync import async_to_sync
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

def main(args: Namespace):
    dm = DatabaseManager()

    if args.upload:

        @async_to_sync
        async def upload():
            print("hey")
            return await dm.upload(path=args.upload)

        files = upload()

        print("Succesfully uploaded files:")
        for file in files:
            print(file)


    print(dm.registry.get_missing_chunks())
    dm.sync()
    print(dm.registry.get_missing_chunks())
