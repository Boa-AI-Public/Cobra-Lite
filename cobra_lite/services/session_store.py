import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from cobra_lite.config import CHAT_HISTORY_MAX_CHARS, CHAT_HISTORY_MAX_MESSAGES

VALID_ROLES = {"user", "assistant"}


class SessionStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = threading.Lock()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "last_session_id": None,
            "sessions": {},
        }

    def _load(self) -> dict[str, Any]:
        if not self.file_path.exists():
            return self._default_state()
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        if not isinstance(data, dict):
            return self._default_state()
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            data["sessions"] = {}
        if "last_session_id" not in data:
            data["last_session_id"] = None
        return data

    def _save(self, state: dict[str, Any]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.file_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp_path.replace(self.file_path)

    def _normalize_messages(self, raw_messages: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_messages, list):
            return []
        out: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in VALID_ROLES:
                continue
            content = str(msg.get("content") or "").strip()
            if not content:
                continue
            ts = msg.get("ts")
            if not isinstance(ts, (int, float)):
                ts = time.time()
            out.append({"role": role, "content": content, "ts": float(ts)})
        return self._prune_messages(out)

    def _prune_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        msgs = list(messages or [])
        if CHAT_HISTORY_MAX_MESSAGES > 0 and len(msgs) > CHAT_HISTORY_MAX_MESSAGES:
            msgs = msgs[-CHAT_HISTORY_MAX_MESSAGES:]

        if CHAT_HISTORY_MAX_CHARS <= 0:
            return msgs

        kept_rev: list[dict[str, Any]] = []
        total = 0
        for msg in reversed(msgs):
            overhead = 12
            total += overhead + len(msg.get("content", ""))
            if total > CHAT_HISTORY_MAX_CHARS:
                break
            kept_rev.append(msg)
        return list(reversed(kept_rev))

    def _first_user_line(self, messages: list[dict[str, Any]]) -> str:
        for msg in messages:
            if msg.get("role") == "user":
                line = str(msg.get("content") or "").strip().splitlines()[0:1]
                if line:
                    return line[0][:72]
        return ""

    def _normalize_session(self, session: Any, session_id: str) -> dict[str, Any] | None:
        if not isinstance(session, dict):
            return None
        created_at = session.get("created_at")
        updated_at = session.get("updated_at")
        if not isinstance(created_at, (int, float)):
            created_at = time.time()
        if not isinstance(updated_at, (int, float)):
            updated_at = created_at

        messages = self._normalize_messages(session.get("messages"))
        title = str(session.get("title") or "").strip()
        if not title:
            title = self._first_user_line(messages) or "New Chat"

        return {
            "id": session_id,
            "title": title[:120],
            "created_at": float(created_at),
            "updated_at": float(updated_at),
            "messages": messages,
        }

    def _ensure_session(self, state: dict[str, Any], session_id: str) -> dict[str, Any]:
        sessions = state.setdefault("sessions", {})
        raw = sessions.get(session_id)
        normalized = self._normalize_session(raw, session_id) if raw is not None else None
        if normalized is None:
            now = time.time()
            normalized = {
                "id": session_id,
                "title": "New Chat",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
        sessions[session_id] = normalized
        return normalized

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            state = self._load()
            sessions = state.get("sessions")
            rows: list[dict[str, Any]] = []
            if isinstance(sessions, dict):
                for session_id, raw_session in sessions.items():
                    session = self._normalize_session(raw_session, str(session_id))
                    if not session:
                        continue
                    rows.append(
                        {
                            "id": session["id"],
                            "title": session["title"],
                            "created_at": session["created_at"],
                            "updated_at": session["updated_at"],
                            "message_count": len(session["messages"]),
                        }
                    )
            rows.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
            return rows

    def get_last_session_id(self) -> str | None:
        with self._lock:
            state = self._load()
            session_id = state.get("last_session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        return session_id

    def set_last_session_id(self, session_id: str) -> None:
        with self._lock:
            state = self._load()
            state["last_session_id"] = session_id
            self._save(state)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._load()
            sessions = state.get("sessions")
            if not isinstance(sessions, dict):
                return None
            raw = sessions.get(session_id)
            session = self._normalize_session(raw, session_id) if raw is not None else None
            if not session:
                return None
            sessions[session_id] = session
            self._save(state)
            return session

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        with self._lock:
            state = self._load()
            now = time.time()
            session_id = f"cobra-{uuid.uuid4().hex}"
            session_title = str(title or "").strip()[:120] or "New Chat"
            session = {
                "id": session_id,
                "title": session_title,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            sessions = state.setdefault("sessions", {})
            sessions[session_id] = session
            state["last_session_id"] = session_id
            self._save(state)
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            state = self._load()
            sessions = state.get("sessions")
            if not isinstance(sessions, dict) or session_id not in sessions:
                return False
            del sessions[session_id]
            if state.get("last_session_id") == session_id:
                newest_id = None
                newest_ts = -1.0
                for sid, raw in sessions.items():
                    normalized = self._normalize_session(raw, sid)
                    if not normalized:
                        continue
                    ts = float(normalized.get("updated_at") or 0.0)
                    if ts > newest_ts:
                        newest_ts = ts
                        newest_id = sid
                state["last_session_id"] = newest_id
            self._save(state)
            return True

    def append_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        normalized_role = str(role or "").strip().lower()
        if normalized_role not in VALID_ROLES:
            raise ValueError("Invalid role. Expected 'user' or 'assistant'.")

        text = str(content or "").strip()
        if not text:
            raise ValueError("Message content cannot be empty.")

        with self._lock:
            state = self._load()
            session = self._ensure_session(state, session_id)
            now = time.time()
            session["messages"].append(
                {
                    "role": normalized_role,
                    "content": text,
                    "ts": now,
                }
            )
            session["messages"] = self._prune_messages(session.get("messages", []))
            if session.get("title") in {"", "New Chat"} and normalized_role == "user":
                session["title"] = text.splitlines()[0][:72]
            session["updated_at"] = now
            state["last_session_id"] = session_id
            self._save(state)
            return session
