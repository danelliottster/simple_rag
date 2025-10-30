"""
rag_sqlite.py
Module for storing and retrieving RAG chunks and embeddings using sqlite3 and sqlite-vec extension.
"""
import sqlite3, pickle, os, json, logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.spatial import cKDTree

# Module-level logger: libraries should not configure logging. Add a NullHandler
# to avoid "No handler" warnings when the package is used by other apps.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class RagSqliteDB:
    def __init__(self, db_path: str):
        """Initialize a simple sqlite DB for chunks and file metadata.

        This version does not load any vector extensions. Embeddings are stored
        as pickled blobs in the chunks table and can be handled later.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.tree = None  # placeholder for nearest neighbor tree
        self.chunks = []  # placeholder for loaded chunks
        self.modelled_chunk_idxs = []  # placeholder for chunks with valid embeddings

    def upsert_chunks(self, embedded_chunks):
        """
        Upsert embedded chunks into the DB.  Update or insert file metadata as needed.
        Args:
            embedded_chunks: List of dicts with keys: chunk, embedding, and file metadata.
        Returns:
            None
        """
        c = self.conn.cursor()
        
        ############################################################################
        # START get a list of unique files from the embedded chunks
        unique_files = {}
        for item in embedded_chunks:
            md = item.get("metadata", {})
            unique_files[md.get("source")] = md
        # END get a list of unique files from the embedded chunks
        ############################################################################
        
        ############################################################################
        # START insert or update the table of file metadata
        # track files we inserted/updated
        # TODO: this code is inefficient
        logger.info("Gathering file metadata for upsert...")
        updated_files = set()
        new_files = set()
        for md in unique_files.values():
            logger.debug(f"Processing chunk for file: {md['source']}")

            # get file metadata
            tags = md.get("tags", [])
            tags_str = ','.join(tags) if tags else ''

            # check if we already have this file and if the last_modified is newer than stored
            existing_ts = self.get_file_last_modified(md.get("source")) # will be None if not found
            if existing_ts is None or (md.get("modified_date") is not None and md.get("modified_date") > existing_ts):
                if existing_ts is None:
                    new_files.add(md.get("source"))
                else:
                    updated_files.add(md.get("source"))
                c.execute('''INSERT INTO file_metadata (source_file, last_modified, summary, tags)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(source_file) DO UPDATE SET last_modified=excluded.last_modified, summary=excluded.summary, tags=excluded.tags''',
                            (md.get("source"), md.get("modified_date"), md.get("summary"), tags_str))

        self.conn.commit()
        # END insert or update the table of file metadata
        ############################################################################
        
        ############################################################################
        # START clear out existing chunks for updated files
        for file in updated_files:
            if file["last_modified"] is not None:
                db_last = self.get_file_last_modified(file["source_file"])
                if db_last is not None and file["last_modified"] > db_last:
                    logger.info(f"Clearing out existing chunks for updated file: {file['source_file']}")
                    self.delete_chunks_for_file(file["source_file"])
        # END clear out existing chunks for updated files
        ############################################################################
        
        ############################################################################
        # START Iterate over embedded chunks and insert into DB
        for chunk in embedded_chunks:
            #check if chunk's source file is in new_files or updated_files, skip otherwise
            if chunk["metadata"]["source"] not in new_files and chunk["metadata"]["source"] not in updated_files:
                continue
            embedding = pickle.dumps(chunk['embedding'])
            tags = chunk["metadata"]["tags"]
            tags_str = ','.join(tags) if tags else ''
            # Insert chunk
            c.execute('''INSERT INTO chunks (source_file, chunk_index, chunk_text, embedding, tags)
                         VALUES (?, ?, ?, ?, ?)''',
                      (chunk["metadata"]["source"], chunk["metadata"]["chunk_index"], chunk["chunk"], embedding, tags_str))
        self.conn.commit()
        # END Iterate over embedded chunks and insert into DB
        ############################################################################
        
    def close(self):
        self.conn.close()

    def get_file_last_modified(self, source_file: str) -> Optional[float]:
        """Return the last_modified timestamp for a source_file recorded in file_metadata, or None."""
        c = self.conn.cursor()
        row = c.execute('SELECT last_modified FROM file_metadata WHERE source_file = ?', (source_file,)).fetchone()
        if row:
            return row[0]
        return None

    def get_all_source_files(self) -> set:
        """Return a set of unique source_file values stored in file_metadata."""
        c = self.conn.cursor()
        rows = c.execute('SELECT source_file FROM file_metadata').fetchall()
        return set(r[0] for r in rows if r and r[0] is not None)

    def delete_chunks_for_file(self, source_file: str) -> int:
        """Delete all chunk rows for the given source_file. Returns number of rows deleted."""
        c = self.conn.cursor()
        # delete from chunks table
        res = c.execute('DELETE FROM chunks WHERE source_file = ?', (source_file,))
        deleted = res.rowcount if hasattr(res, 'rowcount') else None
        self.conn.commit()
        return deleted

    def _load_chunks(self) -> List[Dict]:
        """
        Load chunk records from the DB and unpickle embeddings.
        Only keep the data necessary for vector search.
        Only keep the chunks with valid embeddings.
        Returns:
            A list of dicts with keys: id, source_file, chunk_index, tags
        """
        c = self.conn.cursor()
        sql = 'SELECT id, source_file, chunk_index, embedding, tags, chunk_text FROM chunks'
        rows = c.execute(sql).fetchall()
        self.chunks = []
        for row in rows:
            _id, source_file, chunk_index, embedding_blob, tags, chunk_text = row
            emb = None
            if embedding_blob is not None:
                try:
                    emb = pickle.loads(embedding_blob)
                    emb = np.array(emb)
                except Exception:
                    # store raw blob if unpickling fails
                    logger.error(f"Failed to unpickle embedding for chunk id {_id}, skipping chunk.")
                    continue
            tags = tags.split(',') if tags else []
            self.chunks.append({
                'id': _id,
                'source_file': source_file,
                'chunk_index': chunk_index,
                'embedding': emb,
                'tags': tags,
                'chunk_text': chunk_text
            })

    def build_nn_index(self) -> None:
        """Build a nearest-neighbor index from embeddings in the DB.
        Fills self.tree with a cKDTree instance.
        Also fills self.modelled_chunk_idxs with indices of chunks that have valid embeddings.
        """
        # Load the chunks from the DB
        self._load_chunks()
        # Collect embeddings into array, skip items with None embeddings
        embeddings = np.vstack([chunk["embedding"] for chunk in self.chunks])
        # Build the cKDTree
        self.tree = cKDTree(embeddings)

    def save_index(self, path: str) -> None:
        """
        Save only the scipy cKDTree, the chunks, and the modelled_chunk_idxs to map between the two.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.tree, f)
            pickle.dump(self.chunks, f)

    def load_index_file(self, path: str) -> None:
        """
        Load an index pickle created by save_index.
        """
        with open(path, 'rb') as f:
            self.tree = pickle.load(f)
            self.chunks = pickle.load(f)

    def search_chunks(self, embedding: np.ndarray, tags: Optional[List[str]] = None, top_k: int = 5) -> List[Dict]:
        """
        Perform a vector search using the pre-created nearest neighbor model to find the most similar chunks.
        Args:
            embedding: The query embedding as a numpy array.
            tags: Optional list of tags to filter the chunks.
            top_k: The number of top results to return.
        Returns:
            A list of chunk dictionaries sorted by similarity.
        """
        if not self.tree:
            logger.error("Nearest neighbor tree not loaded. Call load_index_file() first.")
            return None
        
        # make sure top_k is valid
        top_k = min(top_k, len(self.chunks))
        if top_k <= 0:
            logger.error("Invalid top_k value.")
            return None

        # Query the nearest neighbor model
        distances, indices = self.tree.query(embedding, k=top_k)

        # Collect results and filter by tags if provided
        results = []
        for i in indices:
            item = self.chunks[i]
            if tags:
                item_tags = item.get('tags', [])
                if not any(tag in item_tags for tag in tags):
                    continue
            results.append(item)

        return results
