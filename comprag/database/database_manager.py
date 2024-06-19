from __future__ import annotations

import asyncio
import logging
import os
from asyncio import Future
from collections.abc import Awaitable
from hashlib import sha256
from typing import Literal

from database.database_config import Database, Loader, get_databases
from database.registry import FileId, FileMetaData, Registry, VectorStoreId
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class DatabaseManager(metaclass=SingletonMeta):
    def __init__(self):
        self.databases: list[Database] = get_databases()
        self.loader = Loader()
        self.registry = Registry()

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

        try:
            file_checksum = sha256(b"\n\n".join(chunk.page_content.encode() for chunk in chunks))
        except Exception as e:
            logger.error(f"Failed to encode file: '{path}'\n{e}")
            return None

        file_name = os.path.basename(path)
        file_id = f"{file_name}-file-{file_checksum.hexdigest()}"

        if self.registry.has_file(file_id):
            logger.info(f"File '{path}' already uploaded")
            return file_id

        def _get_chunk_metadata(chunk: Document) -> FileMetaData:
            nonlocal file_checksum
            nonlocal file_name
            nonlocal file_id
            nonlocal path

            chunk_checksum = sha256(chunk.page_content.encode())

            return FileMetaData(
                chunk_id=f"{file_name}-chunk-{chunk_checksum.hexdigest()}",
                file_id=file_id,
                file_name=file_name,
                vector_store_entries={},
                chunk_metadata=chunk.metadata,
                chunk_content=chunk.page_content,
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


    def mmr_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[Document]]:
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

        return database.vector_store.amax_marginal_relevance_search(k=k, fetch_k=k*5, query=query)

    def bm25_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[Document]]:
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

    def similarity_search_by_database(self, database_id: VectorStoreId, query: str, k: int) -> Awaitable[list[Document]]:
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

        database.vector_store.asimilarity_search_with_relevance_scores
        
        # TODO: Check if the databases actulaly support this functionality and if they implement it correctly
        return database.vector_store.asimilarity_search(query=query, k=k)

    def batch_similarity(self, pairs: list[tuple[str, str]]) -> list[float]:
        return len(pairs)*[0.5]

    def get_content(self, doc: Document):
        return doc.page_content

    async def search_by_database(self, database_id: VectorStoreId, query: str, combination_method: Literal["rerank", "dp"] = "rerank", k: int = 4, knn: float = 1, mmr: float = 1, mb25: float = 1) -> Awaitable[list[tuple[Document, float]]]:
        """
            Combines the knn, mmr, and mb25 search results to create a better search function for the given database

            You may specify the linear contributions of each
            The relevance for the original query will be determined by a reranker, but you may infulecne this score with the linear factors
        """

        # TODO: these parameters for the linear contributions should be automaticly set, based on the queried topic

        knn_future = self.similarity_search_by_database(database_id, query=query, k=k)
        mmr_future = self.mmr_search_by_database(database_id, query=query, k=k)
        mb25_future = self.mmr_search_by_database(database_id, query=query, k=k)

        knn_results, mmr_results, mb25_results = await asyncio.gather(knn_future, mmr_future, mb25_future, return_exceptions=False)

        # Create tags for the different methods, to collect metadata 
        knn_tag: int = 0
        mmr_tag: int = 1
        mb25_tag: int = 2

        tags = [knn_tag, mmr_tag, mb25_tag]
        tags_size = max(tags) + 1
        weights = [0.0]*tags_size

        weights[knn_tag] = knn
        weights[mmr_tag] = mmr
        weights[knn_tag] = mb25

        # Tag the results
        results: list[tuple[Document, int]]= []
        results.extend([(doc, knn_tag) for doc in knn_results])
        results.extend([(doc, mmr_tag) for doc in mmr_results])
        results.extend([(doc, mb25_tag) for doc in mb25_results])

        # Two algorithms are considered
        # 1. Append all of the results together and sort by relevance to the original query
        # 2. Create a cross product of relevances and pick the best k out of them (like the mmr does)

        # The (1.) is O(k) the second is O(k^2)

        if combination_method == "rerank":
            # TODO: Make use of metadata such for citing and additional context
            pairs = [(query, self.get_content(doc)) for doc, _ in results]
            relevances = self.batch_similarity(pairs)
            reranked = [(sim * weights[tag], doc) for sim, (doc, tag) in zip(relevances, results)]

            reranked.sort(reverse=True) # Desc
            top_k = [doc for _, doc in reranked][:k]

            # Get the actual file ids to ba able to collect telemetry
            return [(self.registry.get_chunk_id(doc), self.get_content(doc)) for doc in top_k]


        elif combination_method == "dp":
            raise NotImplementedError("WIP")





