import os
import logging
import argparse
import csv
import io
from pathlib import Path
import openai
from datetime import datetime
from transformers import AutoTokenizer

from classes.config_manager import ConfigManager
from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader
from classes.chromadb_retriever import ChromaDBRetriever
from classes.rag_query_processor import RAGQueryProcessor
from classes.utilities import delete_directory

# Use the correct path for the config.json file
CONFIG_FILE = r"C:/Users/vikramp/OneDrive - School Health Corporation/Desktop/Assignment Files CISC 691/A04/A04_A-Simple-RAG-Design-Implementation/hu_sp25_691_a03/config.json"

config = ConfigManager(CONFIG_FILE)  # Use ConfigManager for configuration loading

# Set up OpenAI API key for GPT-4 (replace with your actual API key)
openai.api_key = "OPENAI_API_KEY"  # Replace with your actual OpenAI API key

# Path for the Amazon TXT file as specified in the provided file path
raw_input_directory = r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A04\A04_A-Simple-RAG-Design-Implementation\hu_sp25_691_a03\data\raw_input"
cleaned_text_directory = raw_input_directory  # Save cleaned text in the same directory
embeddings_directory = r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A04\A04_A-Simple-RAG-Design-Implementation\hu_sp25_691_a03\data\embeddings"
vectordb_directory = r"C:\Users\vikramp\OneDrive - School Health Corporation\Desktop\Assignment Files CISC 691\A04\A04_A-Simple-RAG-Design-Implementation\hu_sp25_691_a03\data\vectordb"
collection_name = "product_reviews"  # Collection name for ChromaDB


def setup_logging(log_level):
    """Configures logging to console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
    log_filename = f"rag_pipeline_{datetime.now().strftime('%Y%m%d_%H%M_%S%f')[:-4]}"
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(module)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / log_filename)
        ]
    )
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("chromadb").setLevel(logging.INFO)


def ensure_directories_exist(config):
    """Ensures necessary directories exist, creating them if needed."""
    for key in config.get_directory_names():
        dir_path = Path(config.get(key, key))
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


def step01_ingest_documents(args):
    """
    Step 01: Reads and preprocesses documents.
    If a file's tokenized content exceeds the maximum sequence length,
    it splits the text into chunks (with overlap) to avoid exceeding model limits.
    This step also checks if the file is CSV-like and converts each row into a structured format.
    """
    logging.info("[Step 01] Document ingestion started.")

    # Get the list of files to process.
    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(raw_input_directory)

    # Initialize tokenizer using the embedding model name from config (or a default model)
    model_name = config.get("embedding_model_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define maximum sequence length and overlap for chunking
    max_length = 512
    overlap = 50

    def split_text_into_chunks(text, max_length=max_length, overlap=overlap):
        tokens = tokenizer.tokenize(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_length
            chunk_tokens = tokens[start:end]
            chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk)
            start += max_length - overlap
        return chunks

    def preprocess_csv(text):
        """
        Checks if the file appears to be CSV-like (by inspecting the header)
        and converts each row into a structured text format.
        This version uses csv.Sniffer to auto-detect the delimiter so that it
        works even if commas are not used.
        """
        lines = text.splitlines()
        if not lines:
            return text
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=[',', '\t', ';', '|'])
            delimiter = dialect.delimiter
        except Exception as e:
            delimiter = ','  # Fallback to comma if detection fails

        header = lines[0].split(delimiter)
        if any("product" in col.lower() for col in header):
            logging.info(f"CSV-like file detected. Preprocessing structured data using delimiter '{delimiter}'.")
            f = io.StringIO(text)
            reader = csv.DictReader(f, delimiter=delimiter)
            structured_rows = []
            for row in reader:
                row_str = "\n".join(
                    f"{(key.strip() if key is not None else 'Unknown')}: {(value.strip() if value is not None else 'N/A')}"
                    for key, value in row.items()
                )
                structured_rows.append(row_str)
            return "\n\n".join(structured_rows)
        else:
            return text

    # Process each file
    for filename in file_list:
        input_file_path = os.path.join(raw_input_directory, filename)
        logging.info(f"Processing file: {input_file_path}")
        try:
            with open(input_file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading file {input_file_path}: {e}")
            continue

        # Preprocess CSV if applicable
        text = preprocess_csv(text)

        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_length:
            logging.info(f"File {filename} is too long ({len(tokens)} tokens). Splitting into chunks.")
            chunks = split_text_into_chunks(text)
            for i, chunk in enumerate(chunks):
                out_filename = f"{Path(filename).stem}_chunk_{i}{Path(filename).suffix}"
                out_filepath = os.path.join(cleaned_text_directory, out_filename)
                try:
                    with open(out_filepath, "w", encoding="utf-8") as out_f:
                        out_f.write(chunk)
                    logging.info(f"Created chunk file: {out_filepath}")
                except Exception as e:
                    logging.error(f"Error writing chunk file {out_filepath}: {e}")
        else:
            out_filepath = os.path.join(cleaned_text_directory, filename)
            try:
                with open(out_filepath, "w", encoding="utf-8") as out_f:
                    out_f.write(text)
                logging.info(f"Copied file to cleaned text directory: {out_filepath}")
            except Exception as e:
                logging.error(f"Error writing file {out_filepath}: {e}")

    logging.info("[Step 01] Document ingestion completed.")


def step02_generate_embeddings(args):
    """Step 02: Generates vector embeddings from text chunks."""
    logging.info("[Step 02] Embedding generation started.")

    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(cleaned_text_directory)
    preparer = EmbeddingPreparer(file_list=file_list,
                                 input_dir=cleaned_text_directory,
                                 output_dir=embeddings_directory,
                                 embedding_model_name=config.get("embedding_model_name"))
    preparer.process_files()

    logging.info("[Step 02] Embedding generation completed.")


def step03_store_vectors(args):
    """Step 03: Stores embeddings in a vector database."""
    logging.info("[Step 03] Vector storage started.")

    if Path(vectordb_directory).exists():
        logging.info("Deleting existing vectordb")
        delete_directory(vectordb_directory)

    file_list = [args.input_filename] if args.input_filename != "all" else os.listdir(cleaned_text_directory)
    loader = EmbeddingLoader(cleaned_text_file_list=file_list,
                             cleaned_text_dir=cleaned_text_directory,
                             embeddings_dir=embeddings_directory,
                             vectordb_dir=vectordb_directory,
                             collection_name=collection_name)
    loader.process_files()

    logging.info("[Step 03] Vector storage completed.")


def step04_retrieve_relevant_chunks(args):
    """Step 04: Retrieves relevant text chunks based on a query."""
    logging.info("[Step 04] Retrieval started.")
    logging.info(f"Query arguments: {args.query_args}")

    retriever = ChromaDBRetriever(vectordb_dir=vectordb_directory,
                                  embedding_model_name=config.get("embedding_model_name"),
                                  collection_name=collection_name,
                                  score_threshold=float(config.get("retriever_min_score_threshold")))
    search_results = retriever.query(args.query_args, top_k=5)

    if not search_results:
        logging.info("*** No relevant documents found.")
    else:
        for idx, result in enumerate(search_results):
            logging.info(f"Result {idx + 1}:")
            doc_text = result.get('text', '')
            preview_text = (doc_text[:150] + "...") if len(doc_text) > 250 else doc_text
            logging.info(f"ID: {result.get('id', 'N/A')}")
            logging.info(f"Score: {result.get('score', 'N/A')}")
            logging.info(f"Document: {preview_text}")
            logging.info(f"Context: {result.get('context', '')}")
            logging.info("-" * 50)
    logging.info("[Step 04] Retrieval completed.")


def generate_response_with_openai(query_args):
    """Generate a response from GPT-4 using OpenAI's API."""
    pass  # This helper function is not directly used in this implementation.


class RAGQueryProcessor:
    def __init__(self, retriever, llm_client):
        self.retriever = retriever
        self.llm_client = llm_client

    def query(self, question):
        search_results = self.retriever.query(question)
        context = "\n".join([result.get('text', 'No text found') for result in search_results])
        final_prompt = f"Context: {context}\nQuestion: {question}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Replace with "gpt-4" if available or your chosen model.
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return None


def step05_generate_response(args):
    """Step 05: Uses OpenAI GPT-4 to generate an augmented response."""
    logging.info("[Step 05] Response generation started.")
    retriever = ChromaDBRetriever(
        vectordb_dir=vectordb_directory,
        embedding_model_name=config.get("embedding_model_name"),
        collection_name=collection_name
    )
    llm_client = openai
    processor = RAGQueryProcessor(retriever=retriever, llm_client=llm_client)
    response = processor.query(args.query_args)
    if response:
        print("\nResponse:\n", response)
    else:
        print("No response generated.")
    logging.info("[Step 05] Response generation completed.")


def main():
    print("rag_pipeline starting...")
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
                        default="amazon.txt",
                        help="Specify filename or 'all' to process all files in the input directory. (Optional)")
    parser.add_argument("--query_args",
                        nargs="?",
                        default=None,
                        help="Specify search query arguments (enclosed in quotes) for step 4. (Optional, required for step04_retrieve_chunks)")
    args = parser.parse_args()

    if args.step in ["step04_retrieve_chunks", "step05_generate_response"] and args.query_args is None:
        parser.error("The 'query_args' parameter is required when using step04_retrieve_chunks or step05_generate_response.")

    setup_logging(config.get("log_level", "DEBUG"))
    logging.info("------ Command line arguments -------")
    logging.info(f"{'step':<50}: {args.step}")
    logging.info(f"{'input_filename':<50}: {args.input_filename}")
    logging.info(f"{'query_args':<50}: {args.query_args}")
    logging.info("------ Config Settings -------")
    for key in sorted(config.to_dict().keys()):
        logging.info(f"{key:<50}: {config.get(key)}")
    logging.info("------------------------------")

    steps = {
        "step01_ingest": step01_ingest_documents,
        "step02_generate_embeddings": step02_generate_embeddings,
        "step03_store_vectors": step03_store_vectors,
        "step04_retrieve_chunks": step04_retrieve_relevant_chunks,
        "step05_generate_response": step05_generate_response
    }
    steps[args.step](args)
    logging.info("RAG pipeline done")


if __name__ == "__main__":
    main()
