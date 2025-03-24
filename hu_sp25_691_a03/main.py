import os
# import json
import logging
import argparse
from pathlib import Path
from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader
from classes.llm_client import LLMClient
from classes.chromadb_retriever import ChromaDBRetriever
from classes.rag_query_processor import RAGQueryProcessor
from classes.utilities import delete_directory

from datetime import datetime

CONFIG_FILE = "config.json"
config = ConfigManager(CONFIG_FILE)  # Use ConfigManager for configuration loading


def setup_logging(log_level):
    """ Configures logging to console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
    log_filename = f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M_%S%f')[:-4]}"  # Remove last 4 digits of microseconds
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(module)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_filename)
        ]
    )
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("pdfplumber").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce excessive API logs


def ensure_directories_exist(config):
    """Ensures necessary directories exist, creating them if needed."""
    for key in config.get_directory_names():
        dir_path = Path(config.get(key, key))  # Use key name as default
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


def step01_ingest_documents(args):
    """ Step 01: Reads and preprocesses documents."""
    logging.info("[Step 01] Document ingestion started.")

    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("raw_input_directory"))
    ingestor = DocumentIngestor(file_list=file_list,
                                input_dir=config.get("raw_input_directory"),
                                output_dir=config.get("cleaned_text_directory"),
                                embedding_model_name=config.get("embedding_model_name"))
    ingestor.process_files()

    logging.info("[Step 01] Document ingestion completed.")

def step02_generate_embeddings(args):
    """ Step 02: Generates vector embeddings from text chunks."""
    logging.info("[Step 02] Embedding generation started.")

    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("cleaned_text_directory"))
    preparer = EmbeddingPreparer(file_list=file_list,
                                 input_dir=config.get("cleaned_text_directory"),
                                 output_dir=config.get("embeddings_directory"),
                                 embedding_model_name=config.get("embedding_model_name"))
    preparer.process_files()

    logging.info("[Step 02] Embedding generation completed.")


def step03_store_vectors(args):
    """ Step 03: Stores embeddings in a vector database."""
    logging.info("[Step 03] Vector storage started.")

    # delete existing vectordb
    if Path(config.get("vectordb_directory")).exists():
        logging.info("deleting existing vectordb")
        delete_directory(config.get("vectordb_directory"))

    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(config.get("cleaned_text_directory"))
    loader = EmbeddingLoader(cleaned_text_file_list=file_list,
                             cleaned_text_dir=config.get("cleaned_text_directory"),
                             embeddings_dir=config.get("embeddings_directory"),
                             vectordb_dir=config.get("vectordb_directory"),
                             collection_name=config.get("collection_name"))
    loader.process_files()

    logging.info("[Step 03] Vector storage completed.")


def step04_retrieve_relevant_chunks(args):
    """ Step 04: Retrieves relevant text chunks based on a query."""
    logging.info("[Step 04] Retrieval started.")

    logging.info( f"Query arguments: {args.query_args}")

    retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                                 embedding_model_name=config.get("embedding_model_name"),
                                 collection_name=config.get("collection_name"),
                                 score_threshold=float(config.get("retriever_min_score_threshold")))

    search_results = retriever.query(args.query_args, top_k=3)

    if not search_results:
        logging.info("*** No relevant documents found.")
    else:
        for idx, result in enumerate(search_results):

            logging.info(f"Result {idx + 1}:")

            # metadata = result.get('metadata', {})
            doc_text = result.get('text', '')
            preview_text = (doc_text[:150] + "...") if len(doc_text) > 250 else doc_text

            logging.info(f"ID: {result.get('id', 'N/A')}")  # Handle missing ID
            logging.info(f"Score: {result.get('score', 'N/A')}")
            logging.info(f"Document: {preview_text}")
            logging.info(f"Context: {result.get('context', '')}")
            logging.info("-" * 50)

    logging.info("[Step 04] Retrieval completed.")


def step05_generate_response(args):
    """ Step 05: Uses LLM to generate an augmented response."""
    logging.info("[Step 05] Response generation started.")

    llm_client = LLMClient(llm_api_url=config.get("llm_api_url"),
                           llm_model_name=config.get("llm_model_name"))

    # llm_response = llm_client.query(args.query_args)
    # logging.info("\nLLM Response:\n", llm_response)
    # print("\nLLM Response:\n", llm_response)

    retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                             embedding_model_name=config.get("embedding_model_name"),
                             collection_name=config.get("collection_name"))

    processor = RAGQueryProcessor(llm_client=llm_client,
                                  retriever=retriever,
                                  use_rag=args.use_rag)
    response = processor.query(args.query_args)
    print("\nResponse:\n", response)

    logging.info("[Step 05] Response generation completed.")


def main():

    print("rag_pipeline starting...")
    """ Command-line interface for the RAG pipeline."""
    parser = argparse.ArgumentParser(description="CLI for RAG pipeline.")
    parser.add_argument("step",
                        choices=["step01_ingest",
                                 "step02_generate_embeddings",
                                 "step03_store_vectors",
                                 "step04_retrieve_chunks",
                                 "step05_generate_response"],
                        help="Specify the pipeline step.")

    parser.add_argument("--input_filename",
                        nargs="?",
                        default=None,
                        help="Specify filename or 'all' to process all files in the input directory. (Optional)")

    parser.add_argument("--query_args",
                        nargs="?",
                        default=None,
                        help="Specify search query arguments (enclosed in quotes) for step 4. (Optional, required for step04_retrieve_chunks)")

    parser.add_argument("--use_rag",
                        action="store_true",
                        help="Call vectordb for RAG before sending to LLM (Optional, required for step05_generate_response)")

    args = parser.parse_args()

    # Ensure that query_args is required only when using step04_retrieve_chunks
    if args.step in ["step04_retrieve_chunks", "step05_generate_response"] and args.query_args is None:
        parser.error("The 'query_args' parameter is required when using step04_retrieve_chunks or step05_generate_response.")

    if args.step == "step05_generate_response" and args.use_rag is None:
        parser.error("The 'use_rag' parameter is required when using step05_generate_response.")

    #
    setup_logging(config.get("log_level", "DEBUG"))

    logging.info("------ Command line arguments -------")
    logging.info(f"{'step':<50}: {args.step}")
    logging.info(f"{'input_filename':<50}: {args.input_filename}")
    logging.info(f"{'query_args':<50}: {args.query_args}")
    logging.info(f"{'use_rag':<50}: {args.use_rag}")
    logging.info("------ Config Settings -------")
    for key in sorted(config.to_dict().keys()):
        logging.info(f"{key:<50}: {config.get(key)}")
    logging.info("------------------------------")
    ensure_directories_exist(config)

    steps = {
        "step01_ingest": step01_ingest_documents,
        "step02_generate_embeddings": step02_generate_embeddings,
        "step03_store_vectors": step03_store_vectors,
        "step04_retrieve_chunks": step04_retrieve_relevant_chunks,
        "step05_generate_response": step05_generate_response
    }

    steps[args.step](args)

    logging.info(f"RAG pipeline done")

def check_things():
    print("checking things")
    print("1. creating ChromaDBRetriever")
    retriever = ChromaDBRetriever(vectordb_dir=config.get("vectordb_directory"),
                                  embedding_model_name=config.get("embedding_model_name"),
                                  collection_name=config.get("collection_name"))

    print(f"Collection count: {retriever.collection.count()}")

if __name__ == "__main__":
    # print hi
    # check_things()
    main()
