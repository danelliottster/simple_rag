"""
Document parsing and chunking module for RAG systems.
Uses docling for parsing and chonkie for chunking.
"""

from typing import List, Dict, Optional
from pathlib import Path
import docling
from docling.document_converter import DocumentConverter
from chonkie import TokenChunker
import argparse
import pickle
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import logging

class DocumentParserChunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 50, tokenizer: str = "gpt2", 
                 google_api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", chunk_overlap: Optional[int] = None):
        """
        Initialize parser and chunker.
        Args:
            chunk_size: Maximum size of each chunk in tokens
            overlap: Number of tokens to overlap between chunks
            tokenizer: Tokenizer to use for chunking (default: gpt2)
        """
        self.converter = DocumentConverter()
        self.chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.google_api_key = google_api_key
        self.model_name = model_name

        # Initialize pydantic-ai Agent for summarization if key is provided
        self.summarizer_agent = None
        try:
            provider = GoogleProvider(api_key=google_api_key)
            model = GoogleModel(model_name, provider=provider)
            # simple string output
            instructions = (
                "Provide a concise, plain-text summary of the following document. "
                "Keep it to a paragraph and do not include any extra commentary."
            )
            self.summarizer_agent = Agent(model=model, output_type=str, instructions=instructions)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Warning: failed to initialize pydantic-ai summarizer: {e}")

    def parse_document(self, file_path: str) -> Optional[Dict]:
        """
        Parse a document using docling.
        Args:
            file_path: Path to the document file
        Returns:
            Dictionary with document content and metadata
        """
        try:
            result = self.converter.convert(str(file_path))
            content = ""
            if hasattr(result.document, 'texts'):
                for text_element in result.document.texts:
                    content += text_element.text + "\n"
            else:
                content = str(result.document)
            document = {
                'content': content.strip(),
                'metadata': {
                    'source': str(file_path),
                    'filename': Path(file_path).name,
                    'file_type': Path(file_path).suffix.lower(),
                }
            }
            # run a summarization on the entire document and attach the summary to the metadata.
            try:
                resp = self.summarizer_agent.run_sync(document['content'])
                # Agent returns an object with .output in examples; fallback
                summary_text = getattr(resp, 'output', resp)
                # Ensure summary is a plain string
                if summary_text is None:
                    summary_text = ""
                document['metadata']['summary'] = str(summary_text).strip()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Warning: summarization failed for {file_path}: {e}")
                document['metadata']['summary'] = None
            return document
        except Exception as e:
            logging.getLogger(__name__).exception(f"Error parsing document {file_path}: {e}")
            return None

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a parsed document using chonkie (no paragraph splitting).
        Args:
            document: Dictionary with 'content' and 'metadata'
        Returns:
            List of chunk dictionaries with content and metadata
        """
        content = document['content']
        chunks = []
        try:
            chunks_objs = self.chunker.chunk(content)
            for chunk_idx, chunk_obj in enumerate(chunks_objs):
                chunk = {
                    'chunk': chunk_obj,
                    'metadata': {
                        **document['metadata'],
                        'chunk_id': f"chunk_{chunk_idx}",
                        'chunk_index': chunk_idx,
                        'chunk_size': len(chunk_obj.text),
                    }
                }
                chunks.append(chunk)
        except Exception as e:
            logging.getLogger(__name__).exception(f"Error chunking document: {e}")
        return chunks

    def parse_and_chunk(self, file_path: str) -> Optional[List[Dict]]:
        """Convenience wrapper: parse the document and return a list of simple chunk dicts.

        Each returned chunk dict will have:
          - 'chunk': plain text of the chunk
          - 'source_file', 'chunk_index', 'chunk_size', and any metadata present on the document

        Also ensures the document-level summary (if produced) is available in the
        document metadata via parse_document. Returns None on failure.
        """
        doc = self.parse_document(file_path)
        if doc is None:
            return None
        raw_chunks = self.chunk_document(doc)
        simplified = []
        for ch in raw_chunks:
            # ch['chunk'] may be an object with .text; normalize to plain string
            chunk_text = getattr(ch['chunk'], 'text', str(ch['chunk']))
            md = dict(ch.get('metadata', {}))
            # include document summary if present
            if 'summary' in doc.get('metadata', {}):
                md.setdefault('summary', doc['metadata'].get('summary'))
            simplified.append({
                'chunk': chunk_text,
                'metadata': md,
            })
        return simplified

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and chunk a document using docling and chonkie.")
    parser.add_argument("--file", type=str, help="Path to the document file", required=True)
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in tokens (default: 200)")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap in tokens (default: 20)")
    parser.add_argument("--output_file", type=str, help="Path to output pickle file", required=False)
    args = parser.parse_args()

    parser_chunker = DocumentParserChunker(chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    doc = parser_chunker.parse_document(args.file)
    if doc:
        chunks = parser_chunker.chunk_document(doc)
        logging.getLogger(__name__).info(f"Parsed {len(chunks)} chunks from {args.file}")
        for i, chunk in enumerate(chunks):
            logging.getLogger(__name__).info(f"Chunk {i + 1}:")
            logging.getLogger(__name__).info(f"Content: {getattr(chunk['chunk'], 'text', str(chunk['chunk']))[:100]}...")
            logging.getLogger(__name__).info(f"Metadata: {chunk['metadata']}")
            logging.getLogger(__name__).info("-" * 50)
    
    # Determine output file path
    if args.output_file:
        output_pickle = Path(args.output_file)
    else:
        output_pickle = Path(args.file).with_suffix('.chunks.pkl')
    with open(output_pickle, 'wb') as f:
        pickle.dump(chunks, f)
    logging.getLogger(__name__).info(f"Chunks saved to {output_pickle}")
