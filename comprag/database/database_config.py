from __future__ import annotations

import logging
import os
from asyncio import Future
from collections.abc import Awaitable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from database.registry import FileId, FileMetaData, Registry, VectorStoreId
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from util.singleton import SingletonMeta

logger = logging.getLogger(__name__)

@dataclass
class Database:
    id: VectorStoreId
    vector_store: VectorStore

def get_databases() -> list[Database]:
    return [
        Database(
            id="chroma openai",
            vector_store=Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory="./db/chroma_openai"
            )
        ),
        Database(
            id="chroma mistral",
            vector_store=Chroma(
                embedding_function=MistralAIEmbeddings(),
                persist_directory="./db/chroma_mistral"
            )
        ),
    ]


class Loader(metaclass=SingletonMeta):
    def _is_text_based(self, file_path, chunk_size=1024) -> bool:
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

    def aload(self, path) -> Awaitable[list[Document]] | None: 
        _, ext = os.path.splitext(path)

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
                logger.error(f"Unsupported file extension for file '{path}'")
                return None

