import json
import mimetypes
import os
import queue
import socket
import threading
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from cobra_lite.config import BASE_DIR, OPENCLAW_GATEWAY_URL, OPENCLAW_STATE_DIR, SESSIONS_FILE, STATE_FILE
from cobra_lite.services.gateway_client import (
    effective_gateway_url,
    extract_missing_provider,
    send_to_openclaw,
    verify_openclaw_connection,
)
from cobra_lite.services.session_store import SessionStore
from cobra_lite.services.state_store import JsonStateStore


def _find_available_port(host: str, preferred_port: int, max_attempts: int = 50) -> int:
    for offset in range(max_attempts):
        candidate_port = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate_port))
            except OSError:
                continue
            return candidate_port

    raise RuntimeError(
        f"No available port found in range {preferred_port}-{preferred_port + max_attempts - 1}."
    )


def _sse_pack(event_type: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return f"event: {event_type}\n" f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _enqueue_progress(event_queue: queue.Queue, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        event_queue.put({"type": "message", "data": payload})
        return
    event_queue.put(payload)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
        static_url_path="/static",
    )
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "cobra-lite-dev-secret")

    state_store = JsonStateStore(STATE_FILE)
    session_store = SessionStore(SESSIONS_FILE)
    workspace_dir_override = str(os.getenv("OPENCLAW_WORKSPACE_DIR") or "").strip()
    file_read_limit = 256_000
    list_limit = 1000

    def _resolve_workspace_root() -> Path:
        candidates: list[Path] = []
        if workspace_dir_override:
            candidates.append(Path(workspace_dir_override).expanduser())
        candidates.append(Path(OPENCLAW_STATE_DIR).expanduser() / "workspace")
        candidates.append(Path.home() / ".openclaw" / "workspace")

        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_dir():
                    return candidate.resolve()
            except OSError:
                continue
        primary = candidates[0] if candidates else (Path.home() / ".openclaw" / "workspace")
        return primary.resolve()

    def _ensure_workspace_root_exists() -> Path:
        root = _resolve_workspace_root()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _workspace_path_from_relative(relative_path: str | None) -> Path:
        root = _ensure_workspace_root_exists()
        rel = str(relative_path or "").strip().replace("\\", "/")
        rel = rel[1:] if rel.startswith("/") else rel
        target = (root / rel).resolve()
        if target != root and root not in target.parents:
            raise ValueError("Invalid path.")
        return target

    def _relative_workspace_path(path_obj: Path) -> str:
        root = _resolve_workspace_root()
        if path_obj == root:
            return ""
        return str(path_obj.relative_to(root)).replace("\\", "/")

    def _is_binary(path_obj: Path) -> bool:
        try:
            sample = path_obj.read_bytes()[:8192]
        except OSError:
            return True
        if not sample:
            return False
        return b"\x00" in sample

    @app.get("/api/files/root")
    def files_root():
        root = _ensure_workspace_root_exists()
        return jsonify(
            {
                "ok": True,
                "workspace_root": str(root),
                "exists": True,
            }
        )

    @app.get("/api/files/list")
    def files_list():
        rel_path = request.args.get("path")
        try:
            target = _workspace_path_from_relative(rel_path)
        except ValueError:
            return jsonify({"ok": False, "message": "Invalid path."}), 400

        if not target.exists():
            return jsonify({"ok": False, "message": "Path not found."}), 404
        if not target.is_dir():
            return jsonify({"ok": False, "message": "Path is not a directory."}), 400

        entries: list[dict[str, Any]] = []
        try:
            children = sorted(
                list(target.iterdir()),
                key=lambda item: (not item.is_dir(), item.name.lower()),
            )[:list_limit]
        except OSError as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

        for child in children:
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append(
                {
                    "name": child.name,
                    "path": _relative_workspace_path(child),
                    "type": "dir" if child.is_dir() else "file",
                    "size": stat.st_size if child.is_file() else None,
                    "modified_at": int(stat.st_mtime),
                }
            )

        return jsonify(
            {
                "ok": True,
                "workspace_root": str(_resolve_workspace_root()),
                "path": _relative_workspace_path(target),
                "parent_path": _relative_workspace_path(target.parent) if target != _resolve_workspace_root() else None,
                "entries": entries,
            }
        )

    @app.get("/api/files/read")
    def files_read():
        rel_path = request.args.get("path")
        try:
            target = _workspace_path_from_relative(rel_path)
        except ValueError:
            return jsonify({"ok": False, "message": "Invalid path."}), 400

        if not target.exists():
            return jsonify({"ok": False, "message": "File not found."}), 404
        if not target.is_file():
            return jsonify({"ok": False, "message": "Path is not a file."}), 400

        try:
            stat = target.stat()
        except OSError as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

        mime_type, _ = mimetypes.guess_type(str(target))
        is_binary = _is_binary(target)
        if is_binary:
            return jsonify(
                {
                    "ok": True,
                    "path": _relative_workspace_path(target),
                    "name": target.name,
                    "size": stat.st_size,
                    "mime_type": mime_type or "application/octet-stream",
                    "is_binary": True,
                    "content": "",
                    "truncated": False,
                }
            )

        try:
            raw_bytes = target.read_bytes()
        except OSError as exc:
            return jsonify({"ok": False, "message": str(exc)}), 500

        truncated = len(raw_bytes) > file_read_limit
        preview_bytes = raw_bytes[:file_read_limit]
        try:
            content = preview_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = preview_bytes.decode("utf-8", errors="replace")

        return jsonify(
            {
                "ok": True,
                "path": _relative_workspace_path(target),
                "name": target.name,
                "size": stat.st_size,
                "mime_type": mime_type or "text/plain",
                "is_binary": False,
                "content": content,
                "truncated": truncated,
            }
        )

    def _resolve_anthropic_key() -> str | None:
        env_key = str(os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if env_key:
            return env_key
        return state_store.get_provider_key("anthropic")

    def _missing_provider_payload(provider: str = "anthropic") -> dict[str, Any]:
        normalized_provider = (provider or "anthropic").strip().lower() or "anthropic"
        if normalized_provider == "anthropic":
            message = "Anthropic API key is required. Add your key in settings to continue."
        else:
            message = f'{normalized_provider.title()} API key is required. Add your key in settings to continue.'
        return {
            "ok": False,
            "code": "missing_provider_key",
            "provider": normalized_provider,
            "message": message,
        }

    def _resolve_session_id(payload: dict[str, Any]) -> str:
        requested = str(payload.get("session_id") or "").strip()
        if requested and session_store.get_session(requested):
            session_store.set_last_session_id(requested)
            return requested

        last_id = session_store.get_last_session_id()
        if last_id and session_store.get_session(last_id):
            return last_id

        created = session_store.create_session()
        return str(created.get("id"))

    @app.get("/")
    def index() -> str:
        saved_gateway = state_store.get_gateway_url()
        gateway = saved_gateway or OPENCLAW_GATEWAY_URL
        return render_template(
            "index.html",
            has_gateway=bool(saved_gateway),
            has_anthropic_key=bool(_resolve_anthropic_key()),
            default_gateway_url=OPENCLAW_GATEWAY_URL,
            saved_gateway_url=gateway,
        )

    @app.get("/api/sessions")
    def list_sessions():
        sessions = session_store.list_sessions()
        return jsonify(
            {
                "ok": True,
                "sessions": sessions,
                "last_session_id": session_store.get_last_session_id(),
            }
        )

    @app.post("/api/sessions")
    def create_session():
        payload = request.get_json(silent=True) or {}
        title = str(payload.get("title") or "").strip() or None
        session = session_store.create_session(title=title)
        return jsonify({"ok": True, "session": session})

    @app.get("/api/sessions/<session_id>")
    def get_session(session_id: str):
        session = session_store.get_session(session_id)
        if not session:
            return jsonify({"ok": False, "message": "Session not found."}), 404
        session_store.set_last_session_id(session_id)
        return jsonify({"ok": True, "session": session})

    @app.delete("/api/sessions/<session_id>")
    def delete_session(session_id: str):
        deleted = session_store.delete_session(session_id)
        if not deleted:
            return jsonify({"ok": False, "message": "Session not found."}), 404
        return jsonify({"ok": True, "message": "Session deleted."})

    @app.post("/api/verify-gateway")
    def verify_gateway():
        payload = request.get_json(silent=True) or {}
        gateway_url = (payload.get("gateway_url") or "").strip()

        if not gateway_url:
            gateway_url = OPENCLAW_GATEWAY_URL

        is_valid, message = verify_openclaw_connection(gateway_url)
        if not is_valid:
            return jsonify({"ok": False, "message": message}), 400

        state_store.set_gateway_url(gateway_url)
        return jsonify({"ok": True, "message": message})

    @app.get("/api/auth-status")
    def auth_status():
        return jsonify(
            {
                "ok": True,
                "providers": {
                    "anthropic": {
                        "configured": bool(_resolve_anthropic_key()),
                    }
                },
            }
        )

    @app.post("/api/auth/anthropic")
    def save_anthropic_key():
        payload = request.get_json(silent=True) or {}
        api_key = str(payload.get("api_key") or "").strip()
        if not api_key:
            return jsonify({"ok": False, "message": "API key cannot be empty."}), 400
        state_store.set_provider_key("anthropic", api_key)
        return jsonify({"ok": True, "message": "Anthropic key saved."})

    @app.post("/api/prompt")
    def submit_prompt():
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "message": "Prompt cannot be empty."}), 400

        session_id = _resolve_session_id(payload)
        session_store.append_message(session_id, "user", prompt)
        gateway_url = effective_gateway_url(state_store.get_gateway_url())
        anthropic_api_key = _resolve_anthropic_key()
        if not anthropic_api_key:
            missing_payload = _missing_provider_payload("anthropic")
            session_store.append_message(session_id, "assistant", f"Error: {missing_payload['message']}")
            return jsonify({**missing_payload, "session_id": session_id}), 400

        try:
            result = send_to_openclaw(
                prompt=prompt,
                gateway_url=gateway_url,
                session_id=session_id,
                anthropic_api_key=anthropic_api_key,
            )
            final_text = str(result.get("final_observation") or "Task completed.").strip()
            session_store.append_message(session_id, "assistant", final_text)
            return jsonify(
                {
                    "ok": True,
                    "message": "Prompt accepted.",
                    "session_id": session_id,
                    "result": result,
                }
            )
        except Exception as e:
            error_message = str(e)
            missing_provider = extract_missing_provider(error_message)
            if missing_provider:
                missing_payload = _missing_provider_payload(missing_provider)
                session_store.append_message(session_id, "assistant", f"Error: {missing_payload['message']}")
                return jsonify({**missing_payload, "session_id": session_id}), 400
            session_store.append_message(session_id, "assistant", f"Error: {error_message}")
            return jsonify({"ok": False, "session_id": session_id, "message": error_message}), 500

    @app.post("/api/prompt/stream")
    def submit_prompt_stream():
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "message": "Prompt cannot be empty."}), 400

        session_id = _resolve_session_id(payload)
        session_store.append_message(session_id, "user", prompt)
        gateway_url = effective_gateway_url(state_store.get_gateway_url())
        anthropic_api_key = _resolve_anthropic_key()
        if not anthropic_api_key:
            missing_payload = _missing_provider_payload("anthropic")
            session_store.append_message(session_id, "assistant", f"Error: {missing_payload['message']}")
            return jsonify({**missing_payload, "session_id": session_id}), 400

        events: queue.Queue = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            _enqueue_progress(events, event)

        def execute() -> None:
            try:
                result = send_to_openclaw(
                    prompt=prompt,
                    gateway_url=gateway_url,
                    session_id=session_id,
                    anthropic_api_key=anthropic_api_key,
                    progress_callback=emit,
                )
                final_text = str(result.get("final_observation") or "Task completed.").strip()
                session_store.append_message(session_id, "assistant", final_text)
            except Exception as exc:
                message = str(exc)
                missing_provider = extract_missing_provider(message)
                if missing_provider:
                    missing_payload = _missing_provider_payload(missing_provider)
                    session_store.append_message(session_id, "assistant", f"Error: {missing_payload['message']}")
                    emit(
                        {
                            "type": "error",
                            "data": {
                                "message": missing_payload["message"],
                                "code": missing_payload["code"],
                                "provider": missing_provider,
                                "session_id": session_id,
                            },
                        }
                    )
                else:
                    session_store.append_message(session_id, "assistant", f"Error: {message}")
                    emit({"type": "error", "data": {"message": message, "session_id": session_id}})
                emit({"type": "done", "data": {"ok": False, "session_id": session_id}})
                events.put(None)
                return

            emit({"type": "final_result", "data": {"result": result, "session_id": session_id}})
            emit({"type": "done", "data": {"ok": True, "session_id": session_id}})
            events.put(None)

        thread = threading.Thread(target=execute, daemon=True)
        thread.start()

        def event_stream():
            while True:
                event = events.get()
                if event is None:
                    break
                event_type = str(event.get("type", "message")).strip() or "message"
                yield _sse_pack(event_type, event.get("data"))

        response = Response(stream_with_context(event_stream()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


def run_server(app: Flask) -> None:
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "5001"))
    port = _find_available_port(host=host, preferred_port=requested_port)

    print("\n" + "=" * 70)
    print("🦅 Cobra Lite - Security Testing Interface")
    print("=" * 70)
    print(f"🌐 Web UI: http://{host}:{port}")
    print(f"🔧 Gateway: {OPENCLAW_GATEWAY_URL}")
    print("=" * 70 + "\n")

    if port != requested_port:
        print(f"⚠️  Port {requested_port} is busy. Using port {port} instead.\n")

    app.run(host=host, port=port, debug=debug_mode)
