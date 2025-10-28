"""
Script to build the RAG database: parses all documents in a corpus directory, chunks them, and outputs a list of chunks ready for embedding.

Usage:
    python build_rag_chunks.py --corpus_dir <path_to_corpus> --chunk_size <int> --chunk_overlap <int>

Dependencies:
    - docling
    - chonkie
"""
import logging
import os
import argparse
from pathlib import Path
from typing import List
from google import genai
import pickle
from collections import defaultdict

from logging_config import configure_logging

# Configure program-wide logging from the entry point before importing modules
configure_logging()

from rag_sqlite import RagSqliteDB
from document_parser_chunker import DocumentParserChunker

# module logger
logger = logging.getLogger(__name__)

BATCH_SIZE = 32  # Tune for speed/memory

def get_document_paths(corpus_dir: str) -> List[str]:
    """Return a list of document file paths in the corpus directory."""
    exts = ['.docx', '.md', '.pdf', '.txt']
    files = []
    for root, _, filenames in os.walk(corpus_dir):
        for fname in filenames:
            if any(fname.lower().endswith(ext) for ext in exts):
                files.append(os.path.join(root, fname))
    return files

def batch_embed_chunks(chunks, client):
    """
    Embed chunks in batches using the provided GenAI client.
    Args:
        chunks: List of chunk dictionaries with 'chunk' key.
        client: GenAI client initialized with API key.
    Returns:
        List of chunk dictionaries with added 'embedding' key.
    """
    embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c['chunk'] for c in batch]
        response = client.models.embed_content(
            model='text-embedding-004',
            contents=texts
        )
        batch_embeddings = response.embeddings
        for chunk_dict, emb in zip(batch, batch_embeddings):
            out = dict(chunk_dict)
            out['embedding'] = emb.values if hasattr(emb, 'values') else emb
            embeddings.append(out)
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Build RAG chunks from a corpus directory.")
    parser.add_argument('--corpus_dir', type=str, required=True, help='Path to corpus directory')
    parser.add_argument('--chunk_size', type=int, default=256, help='Chunk size (tokens)')
    parser.add_argument('--chunk_overlap', type=int, default=32, help='Chunk overlap (tokens)')
    parser.add_argument('--api_key_file', type=str, required=True, help='Path to text file containing Google API key')
    parser.add_argument('--db_path', type=str, required=True, help='Path to sqlite database to store chunks and metadata')
    parser.add_argument('--tags', nargs='*', default=[], help='Tags to assign to chunks and documents')
    args = parser.parse_args()

    doc_paths = get_document_paths(args.corpus_dir)
    logger.info(f"Found {len(doc_paths)} documents in source directory.")

    ############################################################################
    # START initialize chunker    
    with open(args.api_key_file, 'r') as f:
        api_key = f.read().strip()
    chunker = DocumentParserChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, google_api_key=api_key)
    # END initialize chunker
    ############################################################################

    ############################################################################
    # START initialize DB
    db = RagSqliteDB(args.db_path)
    if db is None:
        logger.error("Failed to initialize RAG sqlite database.")
        return
    # END initialize DB
    ############################################################################

    ############################################################################
    # START determine files to process (new or modified)
    files_to_process = []
    for doc_path in doc_paths:
        try:
            modified_date = os.path.getmtime(doc_path)
        except Exception as e:
            logger.warning(f"Could not get modified date for {doc_path}: {e}")
            modified_date = None

        existing_ts = db.get_file_last_modified(doc_path)
        # If not in DB or file is newer, schedule for processing
        if existing_ts is None or (modified_date is not None and modified_date > existing_ts):
            logger.info(f"Scheduling for processing (new/changed): {doc_path}")
            files_to_process.append((doc_path, modified_date))
        else:
            logger.debug(f"Skipping (unchanged): {doc_path}")
    # END determine files to process (new or modified)
    ############################################################################
    
    ############################################################################
    # START find deleted files and remove from DB
    all_doc_paths = set(doc_paths)
    db_doc_paths = db.get_all_source_files()
    deleted_docs = db_doc_paths - all_doc_paths
    for doc_path in deleted_docs:
        logger.info(f"Removing deleted document from DB: {doc_path}")
        db.remove_document(doc_path)
    # END find deleted files and remove from DB
    ############################################################################

    ############################################################################
    # START chunk the documents
    all_new_chunks = []
    for doc_path, modified_date in files_to_process:
        logger.info(f"Processing: {doc_path}")
        try:
            chunks = chunker.parse_and_chunk(doc_path)
            for c in chunks:
                c["metadata"]["modified_date"] = modified_date
                c["metadata"]["tags"] = args.tags
            if chunks is None:
                logger.warning(f"Failed to parse/chunk document: {doc_path}")
                continue
            all_new_chunks += chunks
        except Exception as e:
            logger.exception(f"Error processing {doc_path}: {e}")

    logger.info(f"Total new/updated chunks ready for embedding: {len(all_new_chunks)}")    
    # END chunk the documents
    ############################################################################

    ############################################################################
    # START generate embeddings for new chunks
    embedded_new_chunks = []
    if all_new_chunks:
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
        client = genai.Client(api_key=api_key)
        embedded_new_chunks = batch_embed_chunks(all_new_chunks, client)
        logger.info(f"Embedded {len(embedded_new_chunks)} new/updated chunks.")
    else:
        logger.info("No new/updated chunks to embed.")
    # END generate embeddings for new chunks
    ############################################################################
    
    ############################################################################
    # START bulk insert new/updated chunks into DB
    db.upsert_chunks(embedded_new_chunks)
    # END bulk insert new/updated chunks into DB
    ############################################################################
        
    ############################################################################
    # START create the vector search model and save to disk
    logger.info("Saving vector search model to disk...")
    db.build_nn_index()
    # extract directory from the provided DB path, ensure it exists, and use it as model_path
    db_dir = Path(args.db_path).resolve().parent
    model_path = str(db_dir) + os.sep + "model.pkl"
    logger.info(f"Saving model to path based on DB location: {model_path}")
    db.save_index(model_path)
    # END create the vector search model
    ############################################################################

    db.close()

if __name__ == "__main__":
    main()
