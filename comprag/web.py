from __future__ import annotations

import asyncio
import json
import logging
import os
from argparse import Namespace
from dataclasses import asdict

import gradio as gr
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def main(args: Namespace):
    dm = DatabaseManager()

    # Parameters for the web interface

    # Has a chatbot tab, a retriever tab adn a database tab
    # For the tiem being the implementation fo the tabs are ...

    def get_database_stats():
        num_files = dm.registry.get_total_files()
        num_chunks = dm.registry.get_total_chunks()

        return gr.DataFrame(
            value=[
                ["Total Number of Files", num_files],
                ["Total Number of Chunks", num_chunks],
            ],
            interactive=False,
            headers=["Description", "Value"],
        )

    def get_vector_db_stats():
        vector_db_stats = dm.registry.get_vector_db_stats()
        return gr.DataFrame(value=list(vector_db_stats.items()), interactive=False, headers=["VectorDB Name", "Chunks"])

    def get_dataset(query: str = ""):
        results = dm.registry.get_all_files(query, 10)
        total_results = dm.registry.get_number_of_matches(query)

        results_list = [[result.file_name, result.chunk_id[0:7], result.chunk_content, json.dumps(result.chunk_metadata), result.file_id[0 : result.file_id.rfind("-") + 1 + 7]] for result in results]
        return (results_list, f"Total Results: {total_results}")

    _dataset, _total_results = get_dataset()
    _dataset_headers = ["File name", "Chunk id", "Content", "Metadata", "File id"]
    with gr.Blocks() as web_ui:
        with gr.Tab(label="Chatbot"):
            pass

        with gr.Tab(label="Search"):
            pass

        with gr.Tab(label="Database"):
            # Make a layout for files: some number
            # chunks: some number
            # N times
            # Vectordb name: some number
            # Display it like a talbe

            with gr.Row():
                with gr.Accordion("statistics"):
                    gr.Markdown("# Database Statistics")

                    with gr.Row():
                        with gr.Column():
                            aggregate_stats_component = get_database_stats()

                        with gr.Column():
                            vectordb_stats_component = get_vector_db_stats()

            with gr.Row():
                with gr.Accordion("Inspect"):
                    with gr.Row():
                        dataset_search = gr.Textbox(label="Search", placeholder="Search for keywords")

                        total_results = gr.Label(value=f"Total Results: {_total_results}")

                    with gr.Row():
                        dataset = gr.DataFrame(value=_dataset, interactive=False, headers=_dataset_headers)

                        dataset_search.submit(
                            fn=get_dataset,
                            inputs=dataset_search,
                            outputs=[dataset, total_results],
                        )

            with gr.Row():
                with gr.Accordion("Upload"):
                    select_button = gr.File(label="Select Files", file_count="multiple")
                    upload_button = gr.Button(value="Upload")

                    async def upload_files(files: list[str] = []):
                        print("uploading files", files)
                        for file in files:
                            print("uploading file", file)
                            base_name = os.path.basename(file)
                            print("base_name", base_name)
                            try:
                                res = await dm.upload(file)
                                print("res", res)
                                if len(res) != 1 or res[0] == None:
                                    gr.Error(f"Failed to upload {base_name}")

                                gr.Info(f"Successfully uploaded {base_name}")
                            except Exception as e:
                                gr.Error(f"Unexpected error! Failed to upload {base_name}")
                                logger.error(f"Failed to upload {base_name}\n{e}")

                        return ([], get_database_stats(), get_vector_db_stats())

                    upload_button.click(
                        fn=upload_files,
                        inputs=[select_button],
                        outputs=[select_button, aggregate_stats_component, vectordb_stats_component],
                    )

            with gr.Row():

                refresh_button = gr.Button(value="Refresh")
                sync_button = gr.Button(value="Sync")

                refresh_button.click(
                    fn=lambda: (get_database_stats(), get_vector_db_stats()),
                    outputs=[aggregate_stats_component, vectordb_stats_component],
                )

                def sync_and_refresh():
                    dm.sync()
                    return [get_database_stats(), get_vector_db_stats()]

                sync_button.click(
                    fn=sync_and_refresh,
                    outputs=[aggregate_stats_component, vectordb_stats_component],
                )

    web_ui.launch()


# demo.launch(share=True, share_server_address="gradio.componentsoft.ai:7000", share_server_protocol="https", auth=(GRADIO_USER, GRADIO_PASSWORD), max_threads=20, show_error=True, favicon_path="favicon.ico", state_session_capacity=20)
