import argparse
import logging.handlers

import search
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
        case "search":
            return search.main(args)
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

    upload_parser = subcommand_parsers.add_parser(name="upload")
    upload_parser.add_argument("--input", "-i", required=True, help="The input path can be any file, directory or URI")

    sync_parser = subcommand_parsers.add_parser(name="sync")
    sync_parser.add_argument(
        "--list",
        "-ls",
        help="List the awailable databases and their statuses",
    )

    search_parser = subcommand_parsers.add_parser(name="search")
    search_parser.add_argument("--query", "-q", help="The query to search for", required=True)
    search_parser.add_argument("--k", "-k", help="The number of results to return", default=4, type=int)

    search_method = search_parser.add_mutually_exclusive_group(required=True)
    search_method.add_argument("--rerank", action="store_true", help="Use the rerank combiner")
    search_method.add_argument("--dp", action="store_true", help="Use the dp combiner")

    search_method.add_argument("--knn", action="store_true", help="Use only the the KNN search tool")
    search_method.add_argument("--mmr", action="store_true", help="Use only the the MMR search tool")
    search_method.add_argument("--bm25", action="store_true", help="Use only the the BM25 search tool")

    args = parser.parse_args()

    main(args)
