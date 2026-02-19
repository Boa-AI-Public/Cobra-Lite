import json
import threading
from pathlib import Path
from typing import Any


class JsonStateStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return {}
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _save(self, state: dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_path.replace(self.file_path)

    def set_gateway_url(self, gateway_url: str) -> None:
        with self._lock:
            state = self._load()
            state["gateway_url"] = gateway_url.strip()
            self._save(state)

    def get_gateway_url(self) -> str | None:
        with self._lock:
            state = self._load()
        stored = state.get("gateway_url")
        if not isinstance(stored, str):
            return None
        return stored.strip() or None

    def set_provider_key(self, provider: str, api_key: str) -> None:
        provider_name = (provider or "").strip().lower()
        if not provider_name:
            raise ValueError("Provider is required.")
        key = (api_key or "").strip()
        if not key:
            raise ValueError("API key is required.")
        with self._lock:
            state = self._load()
            provider_keys = state.get("provider_keys")
            if not isinstance(provider_keys, dict):
                provider_keys = {}
            provider_keys[provider_name] = key
            state["provider_keys"] = provider_keys
            self._save(state)

    def get_provider_key(self, provider: str) -> str | None:
        provider_name = (provider or "").strip().lower()
        if not provider_name:
            return None
        with self._lock:
            state = self._load()
        provider_keys = state.get("provider_keys")
        if not isinstance(provider_keys, dict):
            return None
        stored = provider_keys.get(provider_name)
        if not isinstance(stored, str):
            return None
        return stored.strip() or None
