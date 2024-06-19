import logging
from argparse import Namespace

from asgiref.sync import async_to_sync
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

def main(args: Namespace):
    dm = DatabaseManager()

    if args.input:
        @async_to_sync
        async def upload():
            return await dm.upload(path=args.input)

        files = upload()

        for file in files:
            if file:
                logger.info(f"File uploaded: {file}")
            else:
                logger.warning(f"One file has failed to upload")
