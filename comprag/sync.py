from __future__ import annotations

import json
import logging
import os
import sqlite3
from argparse import Namespace
from collections.abc import Awaitable
from dataclasses import dataclass
from hashlib import sha256

from asgiref.sync import async_to_sync, sync_to_async
from langchain_chroma import Chroma
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class FileMetaData:
    chunk_id: str
    file_id: str
    file_name: str
    vector_store_entries: dict[str, str]  # Database id to chunk id
    chunk_metadata: dict
    chunk_content: str


class SingletonMeta(type):
    """
    A metaclass for the singleton pattern
    """

    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


@dataclass
class Database:
    id: str
    vector_store: VectorStore


class Registry(metaclass=SingletonMeta):

    def __init__(self):
        self.REGISTRY_PATH = 'db/file_metadata.db'
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    file_name TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vector_store_entries (
                    entry_id TEXT,
                    vector_store_id TEXT,
                    chunk_id TEXT,
                    PRIMARY KEY (entry_id, vector_store_id)
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS file_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_id TEXT,
                    chunk_metadata TEXT,
                    content TEXT,
                    FOREIGN KEY(chunk_id) REFERENCES vector_store_entries(chunk_id),
                    FOREIGN KEY(file_id) REFERENCES files(file_id)
                )
            ''')

            conn.commit()

    def add_chunks(self, chunks_metadata: list[FileMetaData]):

        file_id_name = list(set((chunk.file_id, chunk.file_name) for chunk in chunks_metadata))

        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()

            # Inset files
            for file_id, file_name in file_id_name:
                cursor.execute('''
                    INSERT OR REPLACE INTO files (file_id, file_name)
                    VALUES (?, ?)
                ''', (file_id, file_name))

            # Insert Vector Store Entries
            for chunk_metadata in chunks_metadata:
                for vector_store_id, entry_id in chunk_metadata.vector_store_entries.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO vector_store_entries (entry_id, vector_store_id, chunk_id)
                        VALUES (?, ?, ?)
                    ''', (entry_id, vector_store_id, chunk_metadata.chunk_id))

            # Insert chunks
            for chunk_metadata in chunks_metadata:
                cursor.execute('''
                    INSERT OR REPLACE INTO file_chunks (chunk_id, file_id, chunk_metadata, content)
                    VALUES (?, ?, ?, ?)
                ''', (chunk_metadata.chunk_id, chunk_metadata.file_id, json.dumps(chunk_metadata.chunk_metadata), chunk_metadata.chunk_content))

            conn.commit()

    def has_file(self, file_id: str) -> bool:
        with sqlite3.connect(self.REGISTRY_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT file_id FROM files WHERE file_id = ?
            ''', (file_id,))
            return bool(cursor.fetchone())


class DatabaseManager(metaclass=SingletonMeta):
    def __init__(self):
        self.databases: list[Database] = [
            Database(id="chroma openai", vector_store=Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./db/chroma_openai")),
        ]

        self.registry = Registry()

    def _upload_dir(self, path):
        logger.info(f"Trying to upload directory '{path}'")

        files = []
        for dirpath, _, filenames in os.walk(path, followlinks=True):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                # Handle invalid node
                if not os.path.isfile(file_path):
                    logger.error(f"Invaild entry found '{file_path}'")
                    continue

                files.append(self._upload_file(file_path))

    def _upload_uri(self, uri: str) -> bytes | None: ...

    def _is_text_based(self, file_path, chunk_size=1024):
        try:
            with open(file_path, "rb") as file:
                chunk = file.read(chunk_size)
                if not chunk:
                    return False
                try:
                    chunk.decode("utf-8")
                    return True
                except UnicodeDecodeError:
                    return False
        except:
            return False

    async def _load_file(self, path: str) -> Awaitable[list[Document]] | None:
        # Match on file extension
        _, ext = os.path.splitext(path)

        # File with no extension
        if not ext:
            is_text_based = self._is_text_based(path)
            if is_text_based:
                loader = TextLoader(path, autodetect_encoding=True)
                return loader.aload()
            else:
                logger.error(f"Unsupported file extension for file '{path}'")
                return None

        # docx, txt, rtf, pdf, html, xml, json, csv, tsv, md, odt, tex, log, ini, yaml, yml, cfg, properties, php, asp, jsp, aspx, htm, xhtml, rss, atom, srt, vtt, pptx, xlsx, dat, sql, h, cpp, py, java, js, rb, pl, sh, bat, ps1, c, go, swift, kt, cs, scala, ini, toml
        match ext:
            case _:
                logger.error(f"Unsupported file extension '{ext}' for file '{path}'")
                return None

    def _chunk_file(self, pages: list[Document], ext) -> list[Document]:
        character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=200)

        return character_splitter.split_documents(pages)

    async def _upload_file(self, path: str) -> str | None:
        logger.info(f"Trying to upload file '{path}'")

        # Load the file
        _, ext = os.path.splitext(path)
        file_future = await self._load_file(path)
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

    async def upload(self, path: str):

        if not os.path.isfile(path) and not os.path.isdir(path):
            protocol_end = path.find("://")
            if protocol_end == -1:
                logger.error(f"Invalid path '{path}'")
                return None

            return self._upload_uri(path)

        # Garantee that that it is a file or a directory
        if os.path.isdir(path):
            self._upload_dir(path)
        elif os.path.isfile(path):
            return await self._upload_file(path)
        else:
            logger.error(f"Unexpected path '{path}'")
            return None


def main(args: Namespace):
    dm = DatabaseManager()

    @async_to_sync
    async def upload():
        print("hey")
        return await dm.upload(path=args.upload)

    upload()
