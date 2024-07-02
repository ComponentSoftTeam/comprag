import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document
from util.singleton import SingletonMeta

VectorStoreId = str
VectorStoreEntryId = str
ChunkId = str
FileId = str


@dataclass
class FileMetaData:
    chunk_id: ChunkId
    file_id: FileId
    file_name: str
    vector_store_entries: dict[VectorStoreId, VectorStoreEntryId]
    chunk_metadata: dict
    chunk_content: str


class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self.REGISTRY_PATH = "db/file_metadata.db"
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    file_name TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_stores (
                vector_store_id TEXT PRIMARY KEY
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_store_entries (
                    entry_id TEXT,
                    vector_store_id TEXT,
                    chunk_id TEXT,
                    PRIMARY KEY (entry_id, vector_store_id)
                    FOREIGN KEY (vector_store_id) REFERENCES vector_stores(vector_store_id)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS file_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    chunk_metadata TEXT,
                    content TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES vector_store_entries(chunk_id),
                    FOREIGN KEY(file_id) REFERENCES files(file_id)
                )
            """
            )

            conn.commit()

    def register_vector_store(self, vector_store_id: VectorStoreId):
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            INSERT OR REPLACE INTO vector_stores (vector_store_id) VALUES (?)
            """,
                (vector_store_id,),
            )
            conn.commit()

    def get_vector_store_ids(self) -> list[VectorStoreId]:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT vector_store_id FROM vector_stores
            """
            )
            return [vector_store_id for vector_store_id, in cursor.fetchall()]

    def add_chunk_ids(self, vector_store_id: VectorStoreId, chunk_ids: list[tuple[VectorStoreEntryId, ChunkId]]):
        # We can assume that the vector store exists
        # We can assume that the chunk exists

        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            for vector_store_entry_id, chunk_id in chunk_ids:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO vector_store_entries (entry_id, vector_store_id, chunk_id)
                    VALUES (?, ?, ?)
                """,
                    (vector_store_entry_id, vector_store_id, chunk_id),
                )
            conn.commit()

    def add_chunks(self, chunks_metadata: list[FileMetaData]):

        file_id_name = list(set((chunk.file_id, chunk.file_name) for chunk in chunks_metadata))

        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            # Inset files
            for file_id, file_name in file_id_name:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO files (file_id, file_name)
                    VALUES (?, ?)
                """,
                    (file_id, file_name),
                )

            # Insert Vector Store Entries
            for chunk_metadata in chunks_metadata:
                for vector_store_id, entry_id in chunk_metadata.vector_store_entries.items():
                    # Check if the vector store exists
                    cursor.execute(
                        """
                    SELECT vector_store_id FROM vector_stores WHERE vector_store_id = ?
                    """,
                        (vector_store_id,),
                    )

                    if not cursor.fetchone():
                        # Insert the vector store
                        cursor.execute(
                            """
                        INSERT OR REPLACE INTO vector_stores (vector_store_id) VALUES (?)
                        """,
                            (vector_store_id,),
                        )

                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO vector_store_entries (entry_id, vector_store_id, chunk_id)
                        VALUES (?, ?, ?)
                    """,
                        (entry_id, vector_store_id, chunk_metadata.chunk_id),
                    )

            # Insert chunks
            for chunk_metadata in chunks_metadata:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO file_chunks (chunk_id, file_id, chunk_metadata, content)
                    VALUES (?, ?, ?, ?)
                """,
                    (chunk_metadata.chunk_id, chunk_metadata.file_id, json.dumps(chunk_metadata.chunk_metadata), chunk_metadata.chunk_content),
                )

            conn.commit()

    def get_content(self, doc: Document) -> str:
        return doc.page_content

    def get_chunk_id(self, doc: Document) -> ChunkId:
        content = self.get_content(doc) + json.dumps(doc.metadata)

        return sha256(content.encode()).hexdigest()

    def get_file_id(self, docs: list[Document]) -> FileId:
        content = "\n\n".join(self.get_content(doc) + json.dumps(doc.metadata) for doc in docs)
        return sha256(content.encode()).hexdigest()

    def get_file_id_form_chunk_doc(self, doc: Document) -> FileId | None:
        chunk_id = self.get_chunk_id(doc)
        file_id = self.get_file_from_chunk_id(chunk_id)
        return file_id

    def get_file_from_chunk_id(self, chunk_id: ChunkId) -> FileId | None:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT file_id FROM file_chunks WHERE chunk_id = ?
            """,
                (chunk_id,),
            )

            result = cursor.fetchone()

            if not result:
                return None
            return result[0]

    def has_file(self, file_id: FileId) -> bool:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT file_id FROM files WHERE file_id = ?
            """,
                (file_id,),
            )
            return bool(cursor.fetchone())

    # def get_chunk_content(self, chunk_id: ChunkId) -> str:
    #     with sqlite3.connect(self.REGISTRY_PATH) as conn:
    #         cursor = conn.cursor()
    #
    #         cursor.execute(
    #             """
    #             SELECT content FROM file_chunks WHERE chunk_id = ?
    #             """,
    #             (chunk_id,),
    #         )
    #
    #         return cursor.fetchone()[0]

    def get_total_files(self) -> int:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(file_id) FROM files
            """
            )
            return cursor.fetchone()[0]

    def get_total_chunks(self) -> int:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(chunk_id) FROM file_chunks
            """
            )
            return cursor.fetchone()[0]

    def get_vector_db_stats(self) -> dict[VectorStoreId, int]:
        """
        Returns the number of chunks in each vector store
        """

        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT vector_store_id, COUNT(chunk_id) FROM vector_store_entries GROUP BY vector_store_id
            """
            )
            return {vector_store_id: count for vector_store_id, count in cursor.fetchall()}

    def get_chunk(self, chunk_id: ChunkId) -> FileMetaData:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    files.file_name, files.file_id, file_chunks.chunk_id, chunk_metadata, content
                FROM
                    file_chunks
                JOIN 
                    files ON file_chunks.file_id = files.file_id
                WHERE file_chunks.chunk_id = ?
            """,
                (chunk_id,),
            )

            file_name, file_id, chunk_id, chunk_metadata, content = cursor.fetchone()

            file_metadata = FileMetaData(
                chunk_id=chunk_id,
                file_id=file_id,
                file_name=file_name,
                vector_store_entries={},
                chunk_metadata=json.loads(chunk_metadata),
                chunk_content=content,
            )

            cursor.execute(
                """
                SELECT
                    entry_id, vector_store_id
                FROM
                    vector_store_entries
                WHERE chunk_id = ?
            """,
                (chunk_id,),
            )

            vector_store_entries = {v: k for k, v in cursor.fetchall()}
            file_metadata.vector_store_entries = vector_store_entries

            return file_metadata

    def get_chunks(self, chunk_ids: list[ChunkId]) -> list[FileMetaData]:
        return [self.get_chunk(chunk_id) for chunk_id in chunk_ids]

    def get_number_of_matches(self, query: str) -> int:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
            SELECT COUNT(*) FROM file_chunks WHERE content LIKE ?
            """,
                (f"%{query}%",),
            )
            return cursor.fetchone()[0]

    def get_all_files(self, query: str = "", limit: int = 10) -> list[FileMetaData]:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    files.file_id, files.file_name, file_chunks.chunk_id, chunk_metadata, content
                FROM
                    files
                JOIN
                    file_chunks ON files.file_id = file_chunks.file_id
                WHERE content LIKE ?
                LIMIT ?
                            """,
                (f"%{query}%", limit),
            )

            return [
                FileMetaData(
                    file_id=file_id,
                    file_name=file_name,
                    chunk_id=chunk_id,
                    chunk_content=content,
                    chunk_metadata=json.loads(chunk_metadata),
                    vector_store_entries={},
                )
                for (file_id, file_name, chunk_id, chunk_metadata, content) in cursor.fetchall()
            ]

    def get_file(self, file_id: FileId) -> list[FileMetaData]:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            # Get the chunks with the vector store entries
            cursor.execute(
                """
                SELECT
                    files.file_name, file_chunks.chunk_id, chunk_metadata, content
                FROM
                    file_chunks
                JOIN 
                    files ON file_chunks.file_id = files.file_id
                WHERE file_chunks.file_id = ?
            """,
                (file_id,),
            )

            file_chunks = [
                FileMetaData(
                    chunk_id=chunk_id,
                    file_id=file_id,
                    file_name=file_name,
                    vector_store_entries={},
                    chunk_metadata=json.loads(chunk_metadata),
                    chunk_content=content,
                )
                for file_name, chunk_id, chunk_metadata, content in cursor.fetchall()
            ]

            # Fetch the vector store entries
            cursor.execute(
                """
                SELECT
                    vector_store_entries.chunk_id, entry_id, vector_store_id
                FROM
                    vector_store_entries
                JOIN
                    file_chunks ON vector_store_entries.chunk_id = file_chunks.chunk_id
                WHERE file_chunks.file_id = ?
            """,
                (file_id,),
            )

            vector_store_entries_by_chunk = {}
            for chunk_id, entry_id, vector_store_id in cursor.fetchall():
                if chunk_id not in vector_store_entries_by_chunk:
                    vector_store_entries_by_chunk[chunk_id] = []
                vector_store_entries_by_chunk[chunk_id].append((vector_store_id, entry_id))

            for chunk in file_chunks:
                vector_store_entries = vector_store_entries_by_chunk.get(chunk.chunk_id, [])
                chunk.vector_store_entries = dict(vector_store_entries)

        return file_chunks

    def get_missing_chunks(self) -> dict[VectorStoreId, list[ChunkId]]:
        """

        Returns multiple chunks that are missing from the database

        We expect each chunk id to be present in each and every vector store
        This returns the vectorstore - chunkids where it is not true
        """

        # Get the existing (vector store, chunk_id) pairs
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
            SELECT 
                vs.vector_store_id, 
                fc.chunk_id
            FROM 
                vector_stores vs
            CROSS JOIN 
                file_chunks fc
            LEFT JOIN 
                vector_store_entries vse 
            ON 
                vs.vector_store_id = vse.vector_store_id 
                AND fc.chunk_id = vse.chunk_id
            WHERE 
                vse.chunk_id IS NULL;
            """
            )

            missing_chunks = {}
            for vector_store_id, chunk_id in cursor.fetchall():
                if vector_store_id not in missing_chunks:
                    missing_chunks[vector_store_id] = []
                missing_chunks[vector_store_id].append(chunk_id)

        return missing_chunks
