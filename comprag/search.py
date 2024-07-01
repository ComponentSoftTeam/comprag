import logging
from argparse import Namespace
from pprint import pprint

from asgiref.sync import async_to_sync
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

# search_parser = subcommand_parsers.add_parser(name="search")
# search_parser.add_argument("--query", "-q", help="The query to search for", required=True)
# search_method = search_parser.add_mutually_exclusive_group(required=True)
# search_method.add_argument("--rerank", action="store_true", help="Use the rerank combiner")
# search_method.add_argument("--dp", action="store_true", help="Use the dp combiner")
# search_method.add_argument("--knn", action="store_true", help="Use only the the KNN search tool")
# search_method.add_argument("--mmr", action="store_true", help="Use only the the MMR search tool")
# search_method.add_argument("--bm25", action="store_true", help="Use only the the BM25 search tool")


@async_to_sync
async def async_main(args: Namespace):
    dm = DatabaseManager()

    query = args.query
    k = args.k
    vector_store_id = "bge m3"

    if args.rerank:
        # TODO: Add the ability to change the weights
        results = await dm.search_by_database(
            database_id=vector_store_id,
            query=query,
            combination_method="rerank",
            k=k,
        )

    elif args.dp:
        raise NotImplementedError("Not implemented yet")
    elif args.knn:
        results = await dm.similarity_search_by_database(
            database_id=vector_store_id,
            query=query,
            k=k,
        )
    elif args.mmr:
        results = await dm.mmr_search_by_database(
            database_id=vector_store_id,
            query=query,
            k=k,
        )
    elif args.bm25:
        results = await dm.bm25_search_by_database(
            database_id=vector_store_id,
            query=query,
            k=k,
        )
    else:
        raise RuntimeError("No search method specified")

    for result in results:
        pprint(result)


def main(args: Namespace):
    async_main(args)
