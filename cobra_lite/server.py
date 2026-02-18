import json
import os
import queue
import socket
import threading
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from cobra_lite.config import BASE_DIR, OPENCLAW_GATEWAY_URL, SESSIONS_FILE, STATE_FILE
from cobra_lite.services.gateway_client import effective_gateway_url, send_to_openclaw, verify_openclaw_connection
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

    @app.post("/api/prompt")
    def submit_prompt():
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"ok": False, "message": "Prompt cannot be empty."}), 400

        session_id = _resolve_session_id(payload)
        session_store.append_message(session_id, "user", prompt)
        gateway_url = effective_gateway_url(state_store.get_gateway_url())

        try:
            result = send_to_openclaw(
                prompt=prompt,
                gateway_url=gateway_url,
                session_id=session_id,
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

        events: queue.Queue = queue.Queue()

        def emit(event: dict[str, Any]) -> None:
            _enqueue_progress(events, event)

        def execute() -> None:
            try:
                result = send_to_openclaw(
                    prompt=prompt,
                    gateway_url=gateway_url,
                    session_id=session_id,
                    progress_callback=emit,
                )
                final_text = str(result.get("final_observation") or "Task completed.").strip()
                session_store.append_message(session_id, "assistant", final_text)
            except Exception as exc:
                message = str(exc)
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
