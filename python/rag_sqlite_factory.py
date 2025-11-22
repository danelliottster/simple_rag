"""Factory for creating and re-using RagSqliteDB instances.

This module provides a singleton factory that keeps a registry of
RagSqliteDB instances keyed by absolute database path. It ensures only
one factory exists (singleton) and that each DB path maps to a single
RagSqliteDB instance.

Usage:
    from rag_sqlite_factory import get_rag_sqlite_db
    db = get_rag_sqlite_db('/path/to/db.sqlite')
    
TODO: make sure we can handle multiple rag sqlite db instances if needed.
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Dict, Optional, TYPE_CHECKING
import config

if TYPE_CHECKING:
    # avoid importing heavy runtime dependencies at module import time
    # RagSqliteDB is only imported for type checking; the real import is
    # performed lazily inside `get` so importing this module doesn't
    # require numpy or other heavy deps.
    from rag_sqlite import RagSqliteDB


class _RagSqliteFactory:
    """Singleton factory that manages RagSqliteDB instances.

    The factory itself is a singleton (only one global registry). Each
    database path is normalized to an absolute path and mapped to a
    single RagSqliteDB instance.
    """

    _singleton_lock = Lock()
    _singleton_instance: Optional["_RagSqliteFactory"] = None

    def __new__(cls):
        # ensure there is only one factory instance
        with cls._singleton_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = super().__new__(cls)
                # initialize instance attributes
                cls._singleton_instance._registry = {}  # type: Dict[str, RagSqliteDB]
                cls._singleton_instance._lock = Lock()
        return cls._singleton_instance

    def get(self, db_path: str) -> RagSqliteDB:
        """Return an existing RagSqliteDB for db_path or create and register a new one.
        This will also load the vector index from the configured model_pkl_name.

        Args:
            db_path: Path to sqlite database. Will be normalized to absolute path.

        Returns:
            RagSqliteDB instance for the given path.
        """
        # import RagSqliteDB lazily to avoid pulling in heavy deps (numpy)
        from rag_sqlite import RagSqliteDB  # local import
        
        cfg = config.get_config()

        key = os.path.abspath(db_path)
        with self._lock:
            instance = self._registry.get(key)
            if instance is None:
                instance = RagSqliteDB(key)
                instance.load_index_file(cfg.get('model_pkl_name'))
                self._registry[key] = instance
            return instance

    def remove(self, db_path: str) -> bool:
        """Remove a DB instance from the registry if present. Returns True if removed."""
        key = os.path.abspath(db_path)
        with self._lock:
            if key in self._registry:
                del self._registry[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the registry of all DB instances."""
        with self._lock:
            self._registry.clear()

    def list_paths(self) -> Dict[str, RagSqliteDB]:
        """Return a shallow copy of the registry mapping of path -> instance."""
        with self._lock:
            return dict(self._registry)


# Module-level singleton factory instance
factory = _RagSqliteFactory()


def get_rag_sqlite_db(db_path: str) -> RagSqliteDB:
    """Convenience function to get a RagSqliteDB from the global factory."""
    return factory.get(db_path)


def remove_rag_sqlite_db(db_path: str) -> bool:
    """Convenience function to remove an entry from the factory registry."""
    return factory.remove(db_path)


__all__ = ["factory", "get_rag_sqlite_db", "remove_rag_sqlite_db"]
