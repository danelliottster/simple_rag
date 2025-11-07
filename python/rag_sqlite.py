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

    def create_conversation(self, username: str) -> int:
        """
        Create a new conversation for the given username with empty conversation history and summary.
        Returns the new conversation's id.
        """
        import datetime
        c = self.conn.cursor()
        # START create empty conversation JSON and timestamps
        empty_conversation = json.dumps([])
        now = datetime.datetime.now().isoformat()
        # END create empty conversation JSON and timestamps
        ###################################################
        # START insert new conversation row
        c.execute('''
            INSERT INTO conversations (username, conversation, conversation_summary, start_datetime, last_modified_datetime)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, empty_conversation, '', now, now))
        self.conn.commit()
        # END insert new conversation row
        ###################################################
        return c.lastrowid

    def load_conversation(self, conversation_id: int) -> list:
        """
        Load the conversation history for a given conversation ID.
        Returns the conversation history as a list.
        """
        c = self.conn.cursor()
        # START query conversation history by id
        # include deleted flag so we can respect soft-deletes
        row = c.execute('''
            SELECT conversation, deleted FROM conversations WHERE id = ?
        ''', (conversation_id,)).fetchone()
        # END query conversation history by id
        ###################################################
        if row:
            conversation_json, deleted = row
            # respect soft-delete: return None when conversation is marked deleted
            if deleted:
                return None
            return json.loads(conversation_json)
        return []

    def update_summary(self, conversation_id: int, summary: str) -> None:
        """
        Update only the conversation summary for a given conversation ID.
        Args:
            conversation_id: The ID of the conversation to update.
            summary: The new summary string.
        Returns:
            None
        """
        c = self.conn.cursor()
        # START update conversation summary
        c.execute('''
            UPDATE conversations
            SET conversation_summary = ?
            WHERE id = ?
        ''', (summary, conversation_id))
        self.conn.commit()
        # END update conversation summary
        ###################################################

    def update_conversation(self, conversation_id: int, conversation: list) -> None:
        """
        Update the conversation history and last modified datetime for a given conversation ID.
        Args:
            conversation_id: The ID of the conversation to update.
            conversation: The conversation history as a list (will be stored as JSON).
        Returns:
            None
        """
        import datetime
        c = self.conn.cursor()
        # START serialize conversation and get current time
        conversation_json = json.dumps(conversation)
        now = datetime.datetime.now().isoformat()
        # END serialize conversation and get current time
        ###################################################
        # START update conversation row (do not update summary)
        c.execute('''
            UPDATE conversations
            SET conversation = ?, last_modified_datetime = ?
            WHERE id = ?
        ''', (conversation_json, now, conversation_id))
        self.conn.commit()
        # END update conversation row
        ###################################################

    def get_conversation_record(self, conversation_id: int) -> Optional[Dict]:
        """
        Load the conversation record for a given conversation ID.
        Returns a dictionary with all conversation fields, or None if not found.
        """
        c = self.conn.cursor()
        # START query conversation record by id
        row = c.execute('''
            SELECT id, username, conversation, conversation_summary, start_datetime, last_modified_datetime, deleted
            FROM conversations WHERE id = ?
        ''', (conversation_id,)).fetchone()
        # END query conversation record by id
        ###################################################
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'conversation': json.loads(row[2]),
                'conversation_summary': row[3],
                'start_datetime': row[4],
                'last_modified_datetime': row[5],
                'deleted': bool(row[6])
            }
        return None

    def get_conversations_for_user(self, username: str) -> dict:
        """
        Load all conversations for a user.
        Returns a JSON string with id, summary, and last modified date.
        """
        c = self.conn.cursor()
        # START query conversations for user
        # only return non-deleted conversations
        rows = c.execute('''
            SELECT id, conversation_summary, last_modified_datetime
            FROM conversations
            WHERE username = ? AND deleted = 0
        ''', (username,)).fetchall()
        # END query conversations for user
        ###################################################
        # START build result list
        result = []
        for row in rows:
            result.append({
                'id': row[0],
                'conversation_summary': row[1],
                'last_modified_datetime': row[2]
            })
        # END build result list
        ###################################################
        return result

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

    def soft_delete_conversation(self, conversation_id: int) -> None:
        """Mark a conversation row as deleted (soft-delete) by setting deleted=1
        and updating the last_modified_datetime.
        """
        import datetime
        c = self.conn.cursor()
        now = datetime.datetime.now().isoformat()
        c.execute('''
            UPDATE conversations
            SET deleted = 1, last_modified_datetime = ?
            WHERE id = ?
        ''', (now, conversation_id))
        self.conn.commit()

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
        Note: this should be moved to a different module.
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
