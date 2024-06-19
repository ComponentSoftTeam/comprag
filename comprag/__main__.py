import argparse
import logging.handlers

import sync
import upload
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def main(
    args: argparse.Namespace,
) -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format=("[%(asctime)s] " "[%(levelname)s] " "[%(name)s.%(funcName)s:%(lineno)d]: " "%(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.handlers.TimedRotatingFileHandler(
                "log/comprag.log",
                when="D",
                interval=1,
                backupCount=365,
            ),
            logging.StreamHandler(),
        ],
    )

    match args.subcommand:
        case "upload":
            return upload.main(args)
        case "sync":
            return sync.main(args)
        case command_name:
            logger.error(f"Not supported command '{command_name}'.")
            raise RuntimeError("The command is not yet implemented")


if __name__ == "__main__":
    # Lets have a few options
    # Sync database
    # Check healt
    # Run benchmark
    # Serve application
    parser = argparse.ArgumentParser(add_help=True)

    subcommand_parsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
    )

    web_parser = subcommand_parsers.add_parser(name="web")
    web_parser.add_argument(
        "--port",
        "-u",
    )

    upload_parser = subcommand_parsers.add_parser(name="upload")
    upload_parser.add_argument("--input", "-i", required=True, help="The input path can be any file, directory or URI")

    sync_parser = subcommand_parsers.add_parser(name="sync")
    sync_parser.add_argument(
        "--list",
        "-ls",
        help="List the awailable databases and their statuses",
    )

    args = parser.parse_args()

    main(args)
