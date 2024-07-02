from __future__ import annotations

import json
import logging
import os
from argparse import Namespace
from typing import Any, Literal, cast

import gradio as gr
from database.database_manager import DatabaseManager, SearchMethod
from database.registry import ChunkId
from langchain_core.documents import Document
from numpy import minimum

from comprag.database.registry import VectorStoreId

logger = logging.getLogger(__name__)


def database_tab(args: Namespace, dm: DatabaseManager) -> None:
    """
    Create the database tab on a gradio Block component.

    Args:
        args (Namespace): The arguments passed to the main function.
        dm (DatabaseManager): The database manager instance.
    """

    def get_database_stats() -> list[list[Any]]:
        """
        Get the statistics of the database.

        Returns the total number of files and chunks in the database.

        Returns:
            list[list[Any]]: A list of lists containing the description and the value of the statistics.
        """

        num_files = dm.registry.get_total_files()
        num_chunks = dm.registry.get_total_chunks()

        return [
            ["Total Number of Files", num_files],
            ["Total Number of Chunks", num_chunks],
        ]

    def get_vector_db_stats() -> list[tuple[VectorStoreId, int]]:
        """
        Get the statistics of the vector database.

        Returns the name of the vector database and the number of chunks.

        Returns:
            list[tuple[VectorStoreId, int]]: A list of tuple containing the name / id of the vector database and the number of chunks present in that vector store.
        """

        vector_db_stats = dm.registry.get_vector_db_stats()
        return list(vector_db_stats.items())

    def get_dataset(query: str = "") -> tuple[list[list[Any]], int]:
        """
        Get a subset of the uploaded chunks, that match the query.

        Args:
            query (str): The query to search for.

        Returns:
            tuple[list[list[Any]], int]: A tuple containing the list of chunks and the total number of results.
        """

        results = dm.registry.get_all_files(query, 10)
        total_results = dm.registry.get_number_of_matches(query)

        results_list = [
            [
                result.file_name,
                result.chunk_id[0:7],
                result.chunk_content,
                json.dumps(result.chunk_metadata),
                result.file_id[0:7],
            ]
            for result in results
        ]

        return (results_list, total_results)

    # The headers of the dataset
    _dataset_headers = ["File name", "Chunk id", "Content", "Metadata", "File id"]

    with gr.Row():
        """
        Statistics component
        """

        with gr.Accordion("statistics"):
            gr.Markdown("# Database Statistics")

            with gr.Row():
                with gr.Column():
                    """
                    Database statistics component
                    """

                    aggregate_stats_component = gr.DataFrame(
                        value=get_database_stats,
                        interactive=False,
                        headers=["Description", "Value"],
                    )

                with gr.Column():
                    """
                    Vector database statistics component
                    """

                    vectordb_stats_component = gr.DataFrame(value=get_vector_db_stats, interactive=False, headers=["VectorDB Name", "Chunks"])

    with gr.Row():
        """
        Inspect component

        Allows the user to search for chunks in the database, by keywords.
        """

        with gr.Accordion("Inspect"):
            with gr.Row():
                dataset_search = gr.Textbox(label="Search", placeholder="Search for keywords")

                total_results = gr.Textbox(label="Total Results", value=lambda: get_dataset()[1])

            with gr.Row():
                dataset = gr.DataFrame(value=lambda: get_dataset()[0], interactive=False, headers=_dataset_headers)

                dataset_search.submit(
                    fn=get_dataset,
                    inputs=dataset_search,
                    outputs=[dataset, total_results],
                )

    with gr.Row():
        """
        Upload component

        Allows the user to upload files to the database.
        """

        with gr.Accordion("Upload"):
            select_button = gr.File(label="Select Files", file_count="multiple")
            upload_button = gr.Button(value="Upload")

            async def upload_files(files: list[str] = [], query: str = ""):
                """
                Upload the selected files to the database.

                Sends a popup notification of the result of the upload.
                Args:
                    files (list[str]): The list of files_paths to upload.
                    query (str): The search query to update the dataset.

                Returns:
                    tuple[list[str], list[list[Any]], list[list[Any]]]: remaining files, the database statistics and the vector database statistics.
                """

                for file in files:
                    base_name = os.path.basename(file)
                    try:
                        res = await dm.upload(file)
                        if len(res) != 1 or res[0] == None:
                            gr.Error(f"Failed to upload {base_name}")

                        gr.Info(f"Successfully uploaded {base_name}")
                    except Exception as e:
                        gr.Error(f"Unexpected error! Failed to upload {base_name}")
                        logger.error(f"Failed to upload {base_name}\n{e}")

                dataset, total = get_dataset(query)
                return ([], dataset, total, get_database_stats(), get_vector_db_stats())

            upload_button.click(
                fn=upload_files,
                inputs=[select_button, dataset_search],
                outputs=[select_button, dataset, total_results, aggregate_stats_component, vectordb_stats_component],
            )

    with gr.Row():
        """
        Sync component
        """

        refresh_button = gr.Button(value="Refresh")
        """
        Reloads the components with the current data, if a concurent user has made changes.
        """

        sync_button = gr.Button(value="Sync")
        """
        Sync the vector databases with each other, so they have the same chunks.
        """

        refresh_button.click(
            fn=lambda query: (get_database_stats(), get_vector_db_stats(), *get_dataset(query)),
            inputs=dataset_search,
            outputs=[aggregate_stats_component, vectordb_stats_component, dataset, total_results],
        )

        def sync_and_refresh():
            dm.sync()
            return [get_database_stats(), get_vector_db_stats()]

        sync_button.click(
            fn=sync_and_refresh,
            outputs=[aggregate_stats_component, vectordb_stats_component],
        )


def search_tab(args: Namespace, dm: DatabaseManager) -> None:
    """
    Create the search tab on a gradio Block component.

    Args:
        args (Namespace): The arguments passed to the main function.
        dm (DatabaseManager): The database manager instance.
    """

    async def search(database_id: str, query: str, k: int, rerank_method: str):

        rerank_method = cast(Literal["cross-encoder", "rrf"], rerank_method)

        def into_md_json(dictionary: dict[str, Any]) -> str:
            return f"```json\n{json.dumps(dictionary, indent=4)}\n```"

        def into_ext_md(doc: Document) -> str:
            path = doc.metadata.get("source", "")
            _, ext = os.path.splitext(path)

            match ext:
                case "":
                    ext = "txt"

                case "md":
                    return doc.page_content

                case _:
                    ext = "txt"

            return f"```{ext}\n{doc.page_content}\n```"

        def format_serach_result(doc: Document, search_methods: list[SearchMethod], database_id: str) -> list[str]:
            search_methods.sort(key=lambda x: x.value)

            return [
                into_ext_md(doc),
                into_md_json(doc.metadata),
                ", ".join([method.value for method in search_methods]),
                database_id,
            ]

        if database_id == "all":
            results = await dm.search(
                query=query,
                # rerank_method="rrf",
                sub_rerank_method=rerank_method,
                k=k,
            )

            return [format_serach_result(doc, search_methods, db_id) for _, doc, search_methods, db_id in results]

        else:
            results = await dm.search_by_database(
                database_id=database_id,
                query=query,
                rerank_method=rerank_method,
                k=k,
            )

            return [format_serach_result(doc, search_methods, database_id) for _, doc, search_methods in results]

    with gr.Row():
        """
        Search settings component
        """

        with gr.Accordion("Search Settings"):
            with gr.Row():
                choices = ["all"] + [db.id for db in dm.databases]
                search_database = gr.Dropdown(label="Database", choices=list(choices), value="all")
                search_k = gr.Slider(label="Number of returned results / db", minimum=1, maximum=10, step=1, value=4)
                search_rerank_method = gr.Radio(label="Rerank Method", choices=["cross-encoder", "rrf"], value="cross-encoder")

    with gr.Row():
        """
        Search component
        """

        search_input = gr.Textbox(label="Search", placeholder="Search for keywords")

    with gr.Row():
        """
        Search results component
        """

        search_results = gr.DataFrame(value=None, datatype="markdown", interactive=False, headers=["Content", "Metadata", "DB", "Method"])

        search_input.submit(
            fn=search,
            inputs=[search_database, search_input, search_k, search_rerank_method],
            outputs=[search_results],
        )


def main(args: Namespace):
    dm = DatabaseManager()

    with gr.Blocks() as web_ui:
        """
        Gradio web UI
        """

        with gr.Tab(label="Chatbot"):
            pass

        with gr.Tab(label="Search"):
            search_tab(args, dm)

        with gr.Tab(label="Database"):
            database_tab(args, dm)

    web_ui.launch()


# demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=(GRADIO_USER, GRADIO_PASSWORD), max_threads=20, show_error=True, favicon_path="favicon.ico", state_session_capacity=20)
