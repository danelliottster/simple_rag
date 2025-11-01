import os
import threading
from typing import Any, Dict
import yaml

class Config:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, config_path: str = None):
        # private constructor
        if not config_path:
            # default to config.yaml next to this file
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self._config_path = config_path
        self._data = self._load_config()

    @classmethod
    def instance(cls, config_path: str = None):
        # double-checked locking - don't really need this
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path=config_path)
        return cls._instance

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self._config_path):
            # return empty dict if no config file found
            return {}

        with open(self._config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def get_config() -> Config:
    """Convenience: return the singleton instance."""
    return Config.instance()
