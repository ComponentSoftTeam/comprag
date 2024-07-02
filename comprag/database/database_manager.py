from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Future
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pprint import pformat
from typing import Callable, Literal

import colorama
from database.database_config import Database, Loader, Reranker, get_databases
from database.registry import ChunkId, FileId, FileMetaData, Registry, VectorStoreId
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    KNN = "knn"
    MMR = "mmr"
    BM25 = "bm25"


@dataclass
class RRFAccumulator:
    chunk_id: ChunkId
    document: Document
    methods: list[SearchMethod]
    score: float
    methods_by_db: list[tuple[VectorStoreId, list[SearchMethod]]]


class DatabaseManager(metaclass=SingletonMeta):
    def __init__(self):
        self.databases: list[Database] = get_databases()
        self.loader = Loader()
        self.reranker = Reranker()
        self.registry = Registry()

        registered = self.registry.get_vector_store_ids()
        present = [database.id for database in self.databases]
        missing = set(present) - set(registered)
        for database_id in missing:
            logger.info(f"Database '{database_id}' not registered, registering...")
            self.registry.register_vector_store(database_id)
            logger.info(f"Database '{database_id}' registered")

    def _upload_dir(self, path) -> Future[list[FileId | None]]:
        logger.info(f"Trying to upload directory '{path}'")

        files: list[Awaitable[str | None]] = []
        for dirpath, _, filenames in os.walk(path, followlinks=True):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                # Handle invalid node
                if not os.path.isfile(file_path):
                    logger.error(f"Invaild entry found '{file_path}'")
                    continue

                files.append(self._upload_file(file_path))

        return asyncio.gather(*files, return_exceptions=False)

    def _upload_uri(self, uri: str) -> bytes | None: ...

    def _chunk_file(self, pages: list[Document], ext) -> list[Document]:
        # TODO: Should factor out into a saparate configuration

        # XML SPLITTER https://www.restack.io/docs/langchain-knowledge-langchain-xml-splitter
        character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)

        return character_splitter.split_documents(pages)

    async def _upload_file(self, path: str) -> FileId | None:
        logger.info(f"Trying to upload file '{path}'")

        # Load the file
        _, ext = os.path.splitext(path)
        file_future = self.loader.aload(path)
        if not file_future:
            return None

        # Chunk the file's content
        file = await file_future
        chunks = self._chunk_file(file, ext)

        file_name = os.path.basename(path)
        file_id = self.registry.get_file_id(chunks)

        if self.registry.has_file(file_id):
            logger.info(f"File '{path}' already uploaded")
            return file_id

        def _get_chunk_metadata(chunk: Document) -> FileMetaData:
            nonlocal file_name
            nonlocal file_id
            nonlocal path

            return FileMetaData(
                chunk_id=self.registry.get_chunk_id(chunk),
                file_id=file_id,
                file_name=file_name,
                vector_store_entries={},
                chunk_metadata=chunk.metadata,
                chunk_content=self.registry.get_content(chunk),
            )

        # Compute chunk checksums
        try:
            chunks_metadata = [_get_chunk_metadata(chunk) for chunk in chunks]
        except Exception as e:
            logger.error(f"Failed to compute checksums for chunk in file: '{path}'\n{e}")
            return None

        # Upload the chunks
        for database in self.databases:
            chunk_ids = database.vector_store.add_documents(chunks)
            for chunk_id, chunk_metadata in zip(chunk_ids, chunks_metadata):
                chunk_metadata.vector_store_entries[database.id] = chunk_id

        self.registry.add_chunks(chunks_metadata)

        return file_id

    async def upload(self, path: str) -> list[FileId | None]:

        # Hanlde generic URI
        if not os.path.isfile(path) and not os.path.isdir(path):
            protocol_end = path.find("://")
            if protocol_end == -1:
                logger.error(f"Invalid path '{path}'")
                return [None]

            return [None]
            # return self._upload_uri(path)

        # Garantee that that it is a file or a directory
        if os.path.isdir(path):
            return await self._upload_dir(path)
        elif os.path.isfile(path):
            return [await self._upload_file(path)]
        else:
            logger.error(f"Unexpected path '{path}'")
            return [None]

    def sync(self):

        missing_chunks = self.registry.get_missing_chunks()
        database_map = {database.id: database for database in self.databases}

        for vector_store_id, chunk_ids in missing_chunks.items():

            database = database_map.get(vector_store_id)
            if not database:
                logger.error(f"Database '{vector_store_id}' not found")
                continue

            chunks = self.registry.get_chunks(chunk_ids)
            docs = [Document(chunk.chunk_content, metadata=chunk.chunk_metadata) for chunk in chunks]
            entry_ids = database.vector_store.add_documents(docs)
            vector_store_entries = list(zip(entry_ids, [chunk.chunk_id for chunk in chunks]))

            self.registry.add_chunk_ids(vector_store_id, vector_store_entries)

    def get_database(self, database_id: VectorStoreId) -> Database | None:
        return next((db for db in self.databases if db.id == database_id), None)

    def mmr_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[tuple[Document, SearchMethod]]]:
        """
        Returns the top K most similar entries in the specified vector store
        Internali MMR is used.

        This aims to return diverse documents, while still returning relevant docs.

        The returned tuple contains the relevance score [0-1] and the document itself
        """

        database = self.get_database(database_id)
        if not database:
            logger.error(f"Database '{database_id}' not found")

            async def default_value():
                return []

            return default_value()

        async def search() -> list[tuple[Document, SearchMethod]]:
            # TODO: get the sorted variant
            results = await database.vector_store.amax_marginal_relevance_search(k=k, fetch_k=k * 5, query=query)
            return [(res, SearchMethod.MMR) for res in results]

        return search()

    def bm25_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[tuple[Document, SearchMethod]]]:
        """
        Returns the top K most similar entries by keyword matches.
        Internali bm25 is used.

        The returned tuple contains the relevance score [0-1] and the document itself
        """

        database = self.get_database(database_id)
        if not database:
            logger.error(f"Database '{database_id}' not found")

            async def default_value():
                return []

            return default_value()

        # TODO: use an existing implementation
        async def default_value():
            return []

        return default_value()

    def similarity_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[tuple[Document, SearchMethod]]]:
        """
        Returns the top K most similar entries in the specified vector store
        Internaly an approximate KNN is used.

        The returned tuple contains the relevance score [0-1] and the document itself
        """

        database = self.get_database(database_id)
        if not database:
            logger.error(f"Database '{database_id}' not found")

            async def default_value():
                return []

            return default_value()

        # TODO: Check if the databases actulaly support this functionality and if they implement it correctly
        async def search() -> list[tuple[Document, SearchMethod]]:
            results = await database.vector_store.asimilarity_search_with_relevance_scores(query=query, k=k)

            results.sort(key=lambda x: x[1], reverse=True)  # Sort by the relevance score

            return [(res, SearchMethod.KNN) for res, _ in results]

        return search()

    async def search(
        self,
        query: str,
        # rerank_method: Literal["cross-encoder", "rrf"] = "rrf",
        sub_rerank_method: Literal["cross-encoder", "rrf"] = "rrf",
        knn: float = 1,
        mmr: float = 1,
        bm25: float = 1,
        k: int = 4,
        return_all: bool = False,
    ) -> list[tuple[ChunkId, Document, list[SearchMethod], VectorStoreId]]:
        """
        For all databases run the search and return the results
        """

        results_futures = [
            self.search_by_database(
                database_id=database.id,
                query=query,
                rerank_method=sub_rerank_method,
                knn=knn,
                mmr=mmr,
                bm25=bm25,
                k=k,
                return_all=return_all,
            )
            for database in self.databases
        ]

        results_by_database = await asyncio.gather(*results_futures, return_exceptions=False)

        database_ids = [db.id for db in self.databases]

        results_rrf: dict[ChunkId, RRFAccumulator] = {}
        for database_id, results in zip(database_ids, results_by_database):
            for rank, (chunk_id, doc, methods) in enumerate(results):
                if chunk_id not in results_rrf:
                    results_rrf[chunk_id] = RRFAccumulator(chunk_id, doc, [], 0, [])

                results_rrf[chunk_id].methods_by_db.append((database_id, methods))
                additional_score = 1 / (rank + 60)
                results_rrf[chunk_id].score += additional_score

        results_rrfacc_list = list(results_rrf.values())
        results_rrfacc_list.sort(key=lambda x: x.score, reverse=True)

        results = []
        for rrf in results_rrfacc_list:
            rrf.methods_by_db.sort(key=lambda x: x[0])
            results.extend([(rrf.chunk_id, rrf.document, methods, db) for db, methods in rrf.methods_by_db])

        return results[:k] if not return_all else results

    async def search_by_database(
        self,
        database_id: VectorStoreId,
        query: str,
        rerank_method: Literal["cross-encoder", "rrf"] = "rrf",
        knn: float = 1,
        mmr: float = 1,
        bm25: float = 1,
        k: int = 4,
        return_all: bool = False,
    ) -> list[tuple[ChunkId, Document, list[SearchMethod]]]:
        """
        Combines the knn, mmr, and bm25 search results to create a better search function for the given database

        You may specify the linear contributions of each
        The relevance for the original query will be determined by a reranker, but you may infulecne this score with the linear factors
        """

        knn_future = self.similarity_search_by_database(database_id, query=query, k=k)
        mmr_future = self.mmr_search_by_database(database_id, query=query, k=k)
        bm25_future = self.bm25_search_by_database(database_id, query=query, k=k)

        knn_results, mmr_results, bm25_results = await asyncio.gather(knn_future, mmr_future, bm25_future, return_exceptions=False)

        # Collection

        # Two algorithms are considered
        # 1. Rerank with the cross-encoder
        # 2. Rerank using the reciprocal rank fusion algorithm

        if rerank_method == "cross-encoder":

            # Dedupe and attribute in O(H*n) Where H is the cost of the Hash funciton and n is the total number of documents
            results_ce: dict[ChunkId, tuple[ChunkId, Document, list[SearchMethod]]] = {}
            for result in [knn_results, mmr_results, bm25_results]:
                for doc, method in result:
                    chunk_id = self.registry.get_chunk_id(doc)
                    if chunk_id not in results_ce:
                        results_ce[chunk_id] = (chunk_id, doc, [])

                    results_ce[chunk_id][2].append(method)

            results_list = list(results_ce.values())

            reranked = self.reranker.rerank(
                query=query,
                documents=[self.registry.get_content(doc) for _, doc, _ in results_list],
            )

            logger.debug(f"fetched documents: {pformat(reranked)}")
            ordered = [results_list[index] for _, index in reranked]
            top_k = ordered[:k] if not return_all else ordered

            # Get the actual file ids to ba able to collect telemetry
            return top_k

        elif rerank_method == "rrf":
            # RRS: Rank Reciprocal Score
            total_weight = knn + mmr + bm25
            weights: dict[SearchMethod, float] = {}
            weights[SearchMethod.KNN] = knn / total_weight
            weights[SearchMethod.MMR] = mmr / total_weight
            weights[SearchMethod.BM25] = bm25 / total_weight

            # Dedupe and attribute in O(H*n) Where H is the cost of the Hash funciton and n is the total number of documents
            results_rrf: dict[ChunkId, RRFAccumulator] = {}
            for result in [knn_results, mmr_results, bm25_results]:
                for rank, (doc, method) in enumerate(result):
                    chunk_id = self.registry.get_chunk_id(doc)
                    if chunk_id not in results_rrf:
                        results_rrf[chunk_id] = RRFAccumulator(chunk_id, doc, [], 0, [])

                    additional_score = weights[method] / (rank + 60)
                    results_rrf[chunk_id].score += additional_score
                    results_rrf[chunk_id].methods.append(method)

            results_rrfacc_list = list(results_rrf.values())
            results_rrfacc_list.sort(key=lambda x: x.score, reverse=True)
            results_rrf_list = [(rrf.chunk_id, rrf.document, rrf.methods) for rrf in results_rrfacc_list]

            return results_rrf_list[:k] if not return_all else results_rrf_list

        elif rerank_method == "dp":
            # TODO: Implement the dynamic programming algorithm
            raise NotImplementedError("WIP")

    def print_stats(self):
        # Print the number of uploaded files
        # Print the number of total chunk
        # Print vectordb :
        # Print the number of chunks uploaded into each vector db

        total_files = self.registry.get_total_files()
        total_chunks = self.registry.get_total_chunks()
        vector_db_stats = self.registry.get_vector_db_stats()

        # Print {Dark blue}: {Dark orange}

        # Lets try to make it 40 wide with the stats left aligned
        # And the names right and with ellipsis if too long
        def line(lhs: str, rhs: str, width=40, padding=1) -> str:
            right_len = len(rhs)
            left_len = len(lhs)
            if right_len + left_len + padding > width:
                # Truncate the left side with 3 dots
                lhs = lhs[: width - right_len - padding - 3] + "..."
                left_len = len(lhs)

            padding_len = width - left_len - right_len
            return f"{colorama.Fore.BLUE}{lhs}{colorama.Fore.RESET}{' ' * padding_len}{colorama.Fore.YELLOW}{rhs}{colorama.Fore.RESET}"

        print(line("Total:", f"{total_files} files"))
        print(line("Total:", f"{total_chunks} chunks"))

        # print line --- colored
        print(f"{colorama.Fore.BLUE}{'-' * 40}{colorama.Fore.RESET}")
        for vector_db, stats in vector_db_stats.items():
            print(line(f"{vector_db}:", f"{stats} chunks"))
