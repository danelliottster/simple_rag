"""
Script to batch embed RAG chunks using Google Generative AI SDK.

Usage:
    python embed_chunks.py --chunks_file <path_to_chunks.json> --output_file <path_to_output.json>

Dependencies:
    - google-genai (Google GenAI SDK)
    - Chunks file: pickled dicts with 'chunk' key (output from build_rag.py)
"""
import argparse
import json
import pickle
import time
from google import genai
from google.genai import types
import logging

BATCH_SIZE = 32  # TODO: Tune for speed/memory

def batch_embed_chunks(chunks, client):
    """
    Embed chunks in batches using the provided GenAI client.
    Args:
        chunks: List of chunk dictionaries with 'chunk' key storing the text to embed.
        client: GenAI client
    Returns:
        List of chunk dictionaries with added 'embedding' key.
    """
    embeddings = []
    texts = [c['chunk'] for c in chunks]
    # Google GenAI batch embedding API
    response = client.models.embed_content(
        model='text-embedding-004',
        contents=texts
    )
    batch_embeddings = response.embeddings
    for chunk_dict, emb in zip(chunks, batch_embeddings):
        out = dict(chunk_dict)
        out['embedding'] = emb.values if hasattr(emb, 'values') else emb
        embeddings.append(out)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Batch embed RAG chunks using Google GenAI SDK.")
    parser.add_argument('--chunks_file', type=str, required=True, help='Path to input pickled chunks file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output pickled file with embeddings')
    parser.add_argument('--api_key_file', type=str, required=True, help='Path to text file containing Google API key')
    args = parser.parse_args()

    # Read API key from file
    with open(args.api_key_file, 'r') as f:
        api_key = f.read().strip()

    # Load pickled chunks
    with open(args.chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    logging.getLogger(__name__).info(f"Loaded {len(chunks)} chunks.")

    # Embed in batch
    client = genai.Client(api_key=api_key)
    embedded_chunks = batch_embed_chunks(chunks, client)
    logging.getLogger(__name__).info(f"Embedded {len(embedded_chunks)} chunks.")

    # Save output as pickle
    with open(args.output_file, 'wb') as f:
        pickle.dump(embedded_chunks, f)
    logging.getLogger(__name__).info(f"Saved embeddings to {args.output_file}")

if __name__ == "__main__":
    main()
