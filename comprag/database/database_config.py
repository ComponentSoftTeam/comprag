from __future__ import annotations

import logging
import os
from asyncio import Future
from collections.abc import Awaitable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

import numpy as np
from chromadb.api.types import D
from database.registry import FileId, FileMetaData, Registry, VectorStoreId
from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy.typing import NDArray
from sentence_transformers import CrossEncoder, SentenceTransformer
from util.singleton import SingletonMeta

logger = logging.getLogger(__name__)

DatabaseTag = int


@dataclass
class DatabaseConf:
    id: VectorStoreId
    vector_store: VectorStore


@dataclass
class Database(DatabaseConf):
    tag: DatabaseTag


def get_databases() -> list[DatabaseConf]:
    # I want to use the SentenceTransformer as the embedder
    bge_m3_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return [
        # DatabaseConf(id="chroma openai", vector_store=Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./db/chroma_openai")),
        # DatabaseConf(id="chroma mistral", vector_store=Chroma(embedding_function=MistralAIEmbeddings(), persist_directory="./db/chroma_mistral")),
        DatabaseConf(id="bge m3", vector_store=Chroma(embedding_function=bge_m3_embedding, persist_directory="./db/bge_m3")),
    ]


class Reranker(metaclass=SingletonMeta):
    def __init__(self):
        self.bge_v2_m3_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(self, query: str, documents: list[str], reweight: list[float] | None = None) -> list[tuple[float, int]]:
        """
        Returns the reranked documents based on the query and the initial ranking of the documents.

        The returned list has the rewighted scores and the index of the document in the initial list (The order).
        """
        pairs = [[query, doc] for doc in documents]
        scores = self.bge_v2_m3_reranker.predict(
            sentences=pairs,
            batch_size=32,
            show_progress_bar=False,
            num_workers=0,
            apply_softmax=True,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )

        # Creating a flat float array using np.array

        scores = cast(NDArray[np.float_], scores)

        if reweight:
            scores = np.array([s * r for s, r in zip(scores, reweight)])
            print("Scores after reweighting: ", scores)
            print("Weights: ", reweight)
            scores = scores / np.sum(scores)

        order: list[int] = np.argsort(scores)[::-1].tolist()
        return [(scores[i], i) for i in order]


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

        # XML Loader: https://python.langchain.com/v0.2/docs/integrations/document_loaders/xml/

        # docx, txt, rtf, pdf, html, xml, json, csv, tsv, md, odt, tex, log, ini, yaml, yml, cfg, properties, php, asp, jsp, aspx, htm, xhtml, rss, atom, srt, vtt, pptx, xlsx, dat, sql, h, cpp, py, java, js, rb, pl, sh, bat, ps1, c, go, swift, kt, cs, scala, ini, toml
        match ext:
            case _:
                logger.error(f"Unsupported file extension for file '{path}'")
                return None
