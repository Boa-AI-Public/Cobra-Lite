import asyncio
import base64
import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "cobra-lite-dev-secret")

REQUEST_TIMEOUT_SECONDS = 12
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "2000"))
CHAT_HISTORY_MAX_CHARS = int(os.getenv("CHAT_HISTORY_MAX_CHARS", "240000"))
COBRA_EXECUTION_MODE = (os.getenv("COBRA_EXECUTION_MODE", "cli_only").strip().lower() or "cli_only")
STATE_FILE = Path(
    os.getenv("STATE_FILE", Path(__file__).resolve().parent / ".claw_state.json")
)

# Gateway Configuration
def _default_openclaw_gateway_url() -> str:
    configured_url = (os.getenv("OPENCLAW_GATEWAY_URL") or "").strip()
    if configured_url:
        return configured_url
    return "http://127.0.0.1:18789"


OPENCLAW_GATEWAY_URL = _default_openclaw_gateway_url()
OPENCLAW_SESSION_KEY = os.getenv("OPENCLAW_SESSION_KEY", "")  # Optional: for specific session targeting
OPENCLAW_SESSION_ID = os.getenv("OPENCLAW_SESSION_ID", OPENCLAW_SESSION_KEY or "cobra-lite")
OPENCLAW_AGENT_TIMEOUT_SECONDS = int(os.getenv("OPENCLAW_AGENT_TIMEOUT_SECONDS", "180"))
OPENCLAW_VERBOSE_LEVEL = os.getenv("OPENCLAW_VERBOSE_LEVEL", "full").strip() or "full"
OPENCLAW_PROTOCOL_VERSION = 3
GATEWAY_SCOPES = ["operator.admin", "operator.approvals", "operator.pairing"]
OPENCLAW_STATE_DIR = Path(os.getenv("OPENCLAW_STATE_DIR", str(Path.home() / ".openclaw")))
OPENCLAW_IDENTITY_PATH = Path(
    os.getenv("OPENCLAW_DEVICE_IDENTITY_PATH", str(OPENCLAW_STATE_DIR / "identity" / "device.json"))
)
OPENCLAW_DEVICE_AUTH_PATH = Path(
    os.getenv("OPENCLAW_DEVICE_AUTH_PATH", str(OPENCLAW_STATE_DIR / "identity" / "device-auth.json"))
)
DIAGNOSTIC_EXEC_LINE_RE = re.compile(r"^\s*[⚠️❌✅]?\s*🛠️\s*Exec:", re.IGNORECASE)

CLI_ONLY_EXTRA_SYSTEM_PROMPT = """Runtime policy for this interface:
- Use terminal/local tools only.
- Allowed style: exec/bash/process and local workspace/file operations as needed.
- Do NOT use browser.
- Do NOT use web_search or web_fetch (or any web_* tool).
- Do NOT rely on external API keys beyond the configured model provider.
- If a command is missing, report it clearly and continue with available CLI commands."""


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _resolve_ws_gateway_url(gateway_url: str) -> str:
    raw = (gateway_url or "").strip()
    if not raw:
        return "ws://127.0.0.1:18789"

    parsed = urlparse(raw)
    if parsed.scheme in {"ws", "wss"}:
        return raw
    if parsed.scheme in {"http", "https"}:
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        host = parsed.netloc
        path = parsed.path or "/"
        return f"{ws_scheme}://{host}{path}"
    if "://" not in raw:
        return f"ws://{raw}"
    raise ValueError(f"Unsupported gateway URL scheme: {parsed.scheme or 'unknown'}")


def _load_device_identity() -> dict[str, str]:
    if not OPENCLAW_IDENTITY_PATH.exists():
        raise FileNotFoundError(f"Missing device identity file: {OPENCLAW_IDENTITY_PATH}")
    data = json.loads(OPENCLAW_IDENTITY_PATH.read_text(encoding="utf-8"))
    for key in ("deviceId", "publicKeyPem", "privateKeyPem"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid device identity; missing field: {key}")
    return {
        "deviceId": data["deviceId"].strip(),
        "publicKeyPem": data["publicKeyPem"],
        "privateKeyPem": data["privateKeyPem"],
    }


def _load_device_token(device_id: str, role: str = "operator") -> str | None:
    if not OPENCLAW_DEVICE_AUTH_PATH.exists():
        return None
    try:
        data = json.loads(OPENCLAW_DEVICE_AUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if str(data.get("deviceId") or "").strip() != device_id:
        return None

    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return None
    role_entry = tokens.get(role)
    if not isinstance(role_entry, dict):
        return None
    token = str(role_entry.get("token") or "").strip()
    return token or None


def _build_device_auth(identity: dict[str, str], nonce: str, role: str = "operator") -> tuple[dict[str, Any], str | None]:
    from cryptography.hazmat.primitives import serialization

    private_key = serialization.load_pem_private_key(identity["privateKeyPem"].encode("utf-8"), password=None)
    public_key = serialization.load_pem_public_key(identity["publicKeyPem"].encode("utf-8"))
    public_key_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    token = _load_device_token(identity["deviceId"], role=role) or os.getenv("OPENCLAW_GATEWAY_TOKEN")
    signed_at_ms = int(time.time() * 1000)

    payload_fields = [
        "v2" if nonce else "v1",
        identity["deviceId"],
        "cli",
        "cli",
        role,
        ",".join(GATEWAY_SCOPES),
        str(signed_at_ms),
        token or "",
    ]
    if nonce:
        payload_fields.append(nonce)
    payload = "|".join(payload_fields)

    signature_raw = private_key.sign(payload.encode("utf-8"))
    signature = _b64url_encode(signature_raw)

    device = {
        "id": identity["deviceId"],
        "publicKey": _b64url_encode(public_key_raw),
        "signature": signature,
        "signedAt": signed_at_ms,
        "nonce": nonce,
    }
    return device, token


def verify_openclaw_connection(gateway_url: str) -> tuple[bool, str]:
    """Verify that the configured gateway is reachable."""
    import urllib.request
    import urllib.error
    
    test_url = f"{gateway_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(test_url, method="GET")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.getcode() == 200:
                return True, "Gateway is reachable."
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"Cannot reach gateway at {gateway_url}: {str(e)}"
    
    return False, "Unknown error connecting to gateway."


def _coerce_chat_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _prune_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    msgs = list(messages or [])
    if CHAT_HISTORY_MAX_MESSAGES > 0 and len(msgs) > CHAT_HISTORY_MAX_MESSAGES:
        msgs = msgs[-CHAT_HISTORY_MAX_MESSAGES:]

    if CHAT_HISTORY_MAX_CHARS <= 0:
        return msgs

    kept_rev: list[dict[str, str]] = []
    total = 0
    for msg in reversed(msgs):
        overhead = 12
        total += overhead + len(msg.get("content", ""))
        if total > CHAT_HISTORY_MAX_CHARS:
            break
        kept_rev.append(msg)
    return list(reversed(kept_rev))


def _render_chat_history(messages: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for msg in messages:
        role = (msg.get("role") or "").strip().lower()
        label = "User" if role == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        blocks.append(f"{label}:\n{content}")
    return "\n\n".join(blocks).strip()


def _format_chat_history(raw: Any, *, current_prompt: str) -> str:
    msgs = _coerce_chat_messages(raw)
    current = (current_prompt or "").strip()
    if msgs and current:
        last = msgs[-1]
        if (last.get("role") or "").strip().lower() == "user" and (last.get("content") or "").strip() == current:
            msgs = msgs[:-1]
    return _render_chat_history(_prune_chat_messages(msgs))


def send_to_openclaw(
    prompt: str,
    gateway_url: str,
    chat_history: str | None = None,
    progress_callback: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Send a security testing prompt to the gateway agent.

    Cobra Lite runs in a CLI-first mode by default and prioritizes
    terminal execution plus local workspace operations.
    """
    import urllib.request
    import urllib.error
    
    class _PolicyViolationError(Exception):
        pass

    def _tool_disallowed_for_mode(tool_name: str) -> bool:
        if COBRA_EXECUTION_MODE not in {"cli_only", "cli", "terminal_only"}:
            return False
        name = (tool_name or "").strip().lower()
        if not name:
            return False
        return name == "browser" or name.startswith("web_")

    # Build the context for the security agent
    security_context = """You are a CLI-first security testing agent with access to terminal tools and local workspace operations.

Available capabilities:
- Terminal: Run security tools (nmap, curl, nikto, nuclei, ffuf, etc.)
- Local file operations: Read/write reports, save findings

When testing:
0. Use terminal commands first and keep execution grounded in real command output.
1. Start with reconnaissance (subdomains, ports, technologies)
2. Test common vulnerabilities (XSS, SQLi, CSRF, auth issues)
3. Document findings clearly
4. Be thorough but responsible
5. Synthesize a clean final report; do not dump raw event fragments or repeated partial notes
6. Do not call browser, web_search, web_fetch, or any web_* tool.
7. If a command is unavailable, say it is missing and continue with available CLI tooling.

Final response format (always follow):
- Objective
- Actions Taken (include notable commands/tools used)
- Findings
- Recommended Next Steps"""

    full_prompt = prompt
    if chat_history:
        full_prompt = f"{chat_history}\n\nUser:\n{prompt}"
    
    full_prompt = f"{security_context}\n\n{full_prompt}"
    
    def _flatten_text(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, dict):
            out: list[str] = []
            for key in ("delta", "text", "message", "result", "content"):
                if key in value:
                    out.extend(_flatten_text(value.get(key)))
            return out
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                out.extend(_flatten_text(item))
            return out
        return []

    def _format_tool_output(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("output", "stdout", "stderr", "error", "message"):
                text = str(value.get(key) or "").strip()
                if text:
                    return text
            flattened = _flatten_text(value)
            if flattened:
                return "\n".join(flattened).strip()
        flattened = _flatten_text(value)
        if flattened:
            return "\n".join(flattened).strip()
        return json.dumps(value, ensure_ascii=False, default=str)

    def _clean_final_observation(text: str) -> str:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return normalized
        normalized = re.sub(r"^(?:-{3,}|\*{3,})\s*\n+", "", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        return normalized

    def _extract_final_observation(payload_obj: dict[str, Any], *, fallback_text: str = "") -> str:
        result = payload_obj.get("result")
        texts: list[str] = []
        if isinstance(result, dict):
            payloads = result.get("payloads")
            if isinstance(payloads, list):
                texts = [
                    str(item.get("text") or "").strip()
                    for item in payloads
                    if isinstance(item, dict) and str(item.get("text") or "").strip()
                ]
        if texts:
            filtered = [text for text in texts if not DIAGNOSTIC_EXEC_LINE_RE.match(text)]
            candidates = filtered or texts
            if len(candidates) == 1:
                return _clean_final_observation(candidates[0])

            def score(text: str) -> int:
                points = len(text)
                if "\n" in text:
                    points += 40
                if "##" in text or "\n- " in text or "\n1." in text:
                    points += 80
                if re.search(r"\b(summary|findings|next steps|recommended)\b", text, re.IGNORECASE):
                    points += 80
                if DIAGNOSTIC_EXEC_LINE_RE.match(text):
                    points -= 250
                return points

            return _clean_final_observation(max(candidates, key=score))

        fallback = (fallback_text or "").strip()
        if fallback:
            return _clean_final_observation(fallback)

        summary = str(payload_obj.get("summary") or "").strip()
        if summary:
            return _clean_final_observation(summary)
        return _clean_final_observation("Task completed.")

    def _send_via_gateway_ws() -> dict[str, Any]:
        import websockets

        async def _run() -> dict[str, Any]:
            ws_url = _resolve_ws_gateway_url(gateway_url)
            identity = _load_device_identity()

            connect_request_id = str(uuid.uuid4())
            patch_request_id = str(uuid.uuid4())
            agent_request_id = str(uuid.uuid4())
            connect_sent = False
            agent_sent = False
            run_id: str | None = None
            tool_counter = 0
            tool_steps: dict[str, int] = {}
            tool_commands: dict[str, str] = {}
            latest_assistant_text = ""
            reasoning_buffer = ""
            last_reasoning_emit_at = 0.0

            def flush_reasoning(*, force: bool = False) -> None:
                nonlocal reasoning_buffer, last_reasoning_emit_at
                text = reasoning_buffer.strip()
                if not text:
                    reasoning_buffer = ""
                    return
                if len(text) < 12:
                    if force and progress_callback:
                        progress_callback({"type": "reasoning", "data": {"text": text}})
                    reasoning_buffer = ""
                    last_reasoning_emit_at = time.time()
                    return
                should_emit = (
                    force
                    or len(text) >= 320
                    or bool(re.search(r"\n\n|[\n.!?]\s*$", text))
                    or ((time.time() - last_reasoning_emit_at) >= 2.0 and len(text) >= 80)
                )
                if not should_emit:
                    return
                if progress_callback:
                    progress_callback({"type": "reasoning", "data": {"text": text}})
                reasoning_buffer = ""
                last_reasoning_emit_at = time.time()

            async with websockets.connect(
                ws_url,
                max_size=8_000_000,
                open_timeout=REQUEST_TIMEOUT_SECONDS,
            ) as ws:
                while True:
                    raw = await asyncio.wait_for(
                        ws.recv(),
                        timeout=OPENCLAW_AGENT_TIMEOUT_SECONDS + 30,
                    )
                    msg = json.loads(raw)
                    frame_type = str(msg.get("type") or "").strip()

                    if frame_type == "event":
                        event_name = str(msg.get("event") or "").strip()
                        payload = msg.get("payload") or {}

                        if event_name == "connect.challenge" and not connect_sent:
                            nonce = str((payload or {}).get("nonce") or "")
                            device, token = _build_device_auth(identity, nonce=nonce, role="operator")
                            params: dict[str, Any] = {
                                "minProtocol": OPENCLAW_PROTOCOL_VERSION,
                                "maxProtocol": OPENCLAW_PROTOCOL_VERSION,
                                "client": {
                                    "id": "cli",
                                    "displayName": "Cobra Lite",
                                    "version": "cobra-lite",
                                    "platform": os.getenv("OPENCLAW_CLIENT_PLATFORM", os.name),
                                    "mode": "cli",
                                    "instanceId": str(uuid.uuid4()),
                                },
                                "caps": ["tool-events"],
                                "role": "operator",
                                "scopes": GATEWAY_SCOPES,
                                "device": device,
                            }
                            auth_password = os.getenv("OPENCLAW_GATEWAY_PASSWORD", "").strip()
                            if token or auth_password:
                                params["auth"] = {
                                    **({"token": token} if token else {}),
                                    **({"password": auth_password} if auth_password else {}),
                                }
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "req",
                                        "id": connect_request_id,
                                        "method": "connect",
                                        "params": params,
                                    }
                                )
                            )
                            connect_sent = True
                            continue

                        if event_name not in {"agent", "chat"}:
                            continue

                        payload = payload if isinstance(payload, dict) else {}
                        evt_run_id = str(payload.get("runId") or "")
                        if run_id and evt_run_id and evt_run_id != run_id:
                            continue

                        if event_name == "agent":
                            stream_name = str(payload.get("stream") or "").strip().lower()
                            data = payload.get("data") if isinstance(payload.get("data"), dict) else {}

                            if stream_name == "tool":
                                phase = str(data.get("phase") or "").strip().lower()
                                tool_name = str(data.get("name") or "tool").strip() or "tool"
                                if _tool_disallowed_for_mode(tool_name):
                                    raise _PolicyViolationError(
                                        f"Policy blocked non-CLI tool '{tool_name}' in {COBRA_EXECUTION_MODE} mode."
                                    )
                                execution_id = str(data.get("toolCallId") or "").strip()
                                args = data.get("args") if isinstance(data.get("args"), dict) else {}
                                command = (
                                    str(args.get("command") or "").strip()
                                    or str(args.get("cmd") or "").strip()
                                    or str(data.get("meta") or "").strip()
                                    or tool_name
                                )
                                rationale = str(args.get("reason") or args.get("rationale") or "").strip()

                                if phase == "start":
                                    if execution_id and command:
                                        tool_commands[execution_id] = command
                                    if execution_id and execution_id not in tool_steps:
                                        tool_counter += 1
                                        tool_steps[execution_id] = tool_counter
                                    step_index = tool_steps.get(execution_id, tool_counter or 1)
                                    if progress_callback:
                                        progress_callback(
                                            {
                                                "type": "tool_start",
                                                "data": {
                                                    "tool_name": tool_name,
                                                    "command": command,
                                                    "execution_id": execution_id,
                                                    "rationale": rationale,
                                                    "action_index_1based": step_index,
                                                },
                                            }
                                        )
                                elif phase == "update":
                                    output = _format_tool_output(data.get("partialResult"))
                                    if output and progress_callback:
                                        step_index = tool_steps.get(execution_id, tool_counter or 1)
                                        progress_callback(
                                            {
                                                "type": "tool_update",
                                                "data": {
                                                    "tool_name": tool_name,
                                                    "command": tool_commands.get(execution_id, command),
                                                    "tool_output": output,
                                                    "execution_id": execution_id,
                                                    "action_index_1based": step_index,
                                                },
                                            }
                                        )
                                elif phase == "result":
                                    output = _format_tool_output(data.get("result"))
                                    if progress_callback:
                                        step_index = tool_steps.get(execution_id, tool_counter or 1)
                                        progress_callback(
                                            {
                                                "type": "tool_execution",
                                                "data": {
                                                    "tool_name": tool_name,
                                                    "command": tool_commands.get(execution_id, command),
                                                    "tool_output": output or "(no output)",
                                                    "execution_id": execution_id,
                                                    "is_error": bool(data.get("isError")),
                                                    "action_index_1based": step_index,
                                                },
                                            }
                                        )
                                continue

                            if stream_name == "assistant":
                                delta_raw = data.get("delta")
                                text_raw = data.get("text")
                                delta = str(delta_raw) if delta_raw is not None else ""
                                text = str(text_raw) if text_raw is not None else ""
                                if text.strip():
                                    latest_assistant_text = text.strip()
                                if delta:
                                    reasoning_buffer += delta
                                    flush_reasoning()
                                if progress_callback and text.strip():
                                    progress_callback(
                                        {
                                            "type": "assistant_delta",
                                            "data": {"text": text},
                                        }
                                    )
                                continue

                            if stream_name == "lifecycle":
                                phase = str(data.get("phase") or "").strip().lower()
                                if phase in {"end", "error"}:
                                    flush_reasoning(force=True)
                                if progress_callback and phase:
                                    progress_callback(
                                        {
                                            "type": "run_status",
                                            "data": {"phase": phase},
                                        }
                                    )
                                continue

                        if event_name == "chat":
                            state = str(payload.get("state") or "").strip().lower()
                            message_obj = payload.get("message")
                            text_fragments = _flatten_text(message_obj) if state == "final" else []
                            if text_fragments and not latest_assistant_text:
                                latest_assistant_text = "\n".join(text_fragments).strip()
                            continue

                        continue

                    if frame_type != "res":
                        continue

                    response_id = str(msg.get("id") or "")
                    is_ok = bool(msg.get("ok"))
                    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
                    error_shape = msg.get("error") if isinstance(msg.get("error"), dict) else {}

                    if response_id == connect_request_id:
                        if not is_ok:
                            error_message = str(error_shape.get("message") or "connect failed").strip()
                            raise Exception(f"gateway connect failed: {error_message}")
                        patch_params: dict[str, Any] = {
                            "key": OPENCLAW_SESSION_ID,
                            "verboseLevel": OPENCLAW_VERBOSE_LEVEL,
                        }
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "req",
                                    "id": patch_request_id,
                                    "method": "sessions.patch",
                                    "params": patch_params,
                                }
                            )
                        )
                        continue

                    if response_id == patch_request_id and not agent_sent:
                        agent_params: dict[str, Any] = {
                            "message": full_prompt,
                            "sessionId": OPENCLAW_SESSION_ID,
                            "sessionKey": OPENCLAW_SESSION_ID,
                            "idempotencyKey": str(uuid.uuid4()),
                            "timeout": OPENCLAW_AGENT_TIMEOUT_SECONDS,
                        }
                        if COBRA_EXECUTION_MODE in {"cli_only", "cli", "terminal_only"}:
                            agent_params["extraSystemPrompt"] = CLI_ONLY_EXTRA_SYSTEM_PROMPT
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "req",
                                    "id": agent_request_id,
                                    "method": "agent",
                                    "params": agent_params,
                                }
                            )
                        )
                        agent_sent = True
                        continue

                    if response_id == agent_request_id:
                        if not is_ok:
                            error_message = str(error_shape.get("message") or "agent request failed").strip()
                            raise Exception(error_message)
                        if str(payload.get("status") or "").strip().lower() == "accepted":
                            accepted_run_id = str(payload.get("runId") or "").strip()
                            if accepted_run_id:
                                run_id = accepted_run_id
                            continue
                        flush_reasoning(force=True)
                        final_observation = _extract_final_observation(payload, fallback_text=latest_assistant_text)
                        return {"final_observation": final_observation}

        return asyncio.run(_run())

    def _send_via_cli() -> dict[str, Any]:
        if progress_callback:
            progress_callback(
                {
                    "type": "tool_start",
                    "data": {
                        "tool_name": "gateway-agent",
                        "command": "openclaw agent",
                        "execution_id": "gateway-agent",
                    },
                }
            )

        command = [
            "openclaw",
            "agent",
            "--session-id",
            OPENCLAW_SESSION_ID,
            "--message",
            full_prompt,
            "--json",
            "--timeout",
            str(OPENCLAW_AGENT_TIMEOUT_SECONDS),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=OPENCLAW_AGENT_TIMEOUT_SECONDS + 15,
        )

        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "").strip()
            raise Exception(f"Gateway CLI error: {error_text or f'exit code {completed.returncode}'}")

        raw = (completed.stdout or "").strip()
        if not raw:
            raise Exception("Gateway CLI returned an empty response.")

        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # CLI may emit logs before JSON output; parse from first JSON object start.
            start = raw.find("{")
            if start >= 0:
                parsed = json.loads(raw[start:])
            else:
                raise

        if not isinstance(parsed, dict):
            raise Exception("Gateway CLI returned non-JSON output.")

        if progress_callback:
            progress_callback(
                {
                    "type": "tool_execution",
                    "data": {
                        "tool_name": "gateway-agent",
                        "tool_output": f"Run status: {parsed.get('status', 'unknown')}",
                        "execution_id": str(parsed.get("runId", "gateway-agent")),
                    },
                }
            )

        return {"final_observation": _extract_final_observation(parsed)}

    def _send_via_http() -> dict[str, Any]:
        endpoint = f"{gateway_url.rstrip('/')}/api/chat"
        payload = {
            "message": full_prompt,
            "stream": True
        }

        if OPENCLAW_SESSION_KEY:
            payload["sessionKey"] = OPENCLAW_SESSION_KEY

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=120) as response:
            result_text = ""

            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    event = json.loads(line_str)
                    event_type = event.get("type", "data")
                    data = event.get("data", {})

                    if progress_callback:
                        progress_callback({
                            "type": event_type,
                            "data": data
                        })

                    if event_type == "content":
                        result_text += data.get("text", "")
                    elif event_type == "tool_call":
                        tool_name = str(data.get("tool") or "").strip()
                        if _tool_disallowed_for_mode(tool_name):
                            raise _PolicyViolationError(
                                f"Policy blocked non-CLI tool '{tool_name}' in {COBRA_EXECUTION_MODE} mode."
                            )
                        if progress_callback:
                            progress_callback({
                                "type": "tool_start",
                                "data": {
                                    "tool_name": data.get("tool", "unknown"),
                                    "command": data.get("description", ""),
                                    "execution_id": data.get("id", "")
                                }
                            })
                    elif event_type == "tool_result":
                        if progress_callback:
                            progress_callback({
                                "type": "tool_execution",
                                "data": {
                                    "tool_name": data.get("tool", "unknown"),
                                    "tool_output": data.get("output", ""),
                                    "execution_id": data.get("id", "")
                                }
                            })

                except json.JSONDecodeError:
                    continue

            return {
                "final_observation": result_text or "Task completed."
            }

    errors: list[str] = []

    try:
        return _send_via_gateway_ws()
    except Exception as ws_error:
        if isinstance(ws_error, _PolicyViolationError):
            raise
        errors.append(f"ws: {str(ws_error)}")

    try:
        return _send_via_http()
    except Exception as http_error:
        if isinstance(http_error, _PolicyViolationError):
            raise
        message = str(http_error)
        errors.append(f"http: {message}")
        if "HTTP Error 405" in message or "Method Not Allowed" in message:
            try:
                return _send_via_cli()
            except Exception as cli_error:
                errors.append(f"cli: {str(cli_error)}")

    try:
        return _send_via_cli()
    except Exception as cli_error:
        errors.append(f"cli: {str(cli_error)}")
        raise Exception(f"Gateway error: {' | '.join(errors)}")


def _load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_FILE.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(STATE_FILE)


def set_gateway_url(gateway_url: str) -> None:
    state = _load_state()
    state["gateway_url"] = gateway_url.strip()
    _save_state(state)


def get_gateway_url() -> str | None:
    state = _load_state()
    stored = state.get("gateway_url")
    if not isinstance(stored, str):
        return None
    return stored.strip() or None


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


@app.get("/")
def index() -> str:
    gateway = get_gateway_url() or OPENCLAW_GATEWAY_URL
    return render_template(
        "index.html",
        has_gateway=bool(gateway),
        default_gateway_url=OPENCLAW_GATEWAY_URL,
        saved_gateway_url=gateway,
    )


@app.post("/api/verify-gateway")
def verify_gateway():
    payload = request.get_json(silent=True) or {}
    gateway_url = (payload.get("gateway_url") or "").strip()

    if not gateway_url:
        gateway_url = OPENCLAW_GATEWAY_URL

    is_valid, message = verify_openclaw_connection(gateway_url)
    if not is_valid:
        return jsonify({"ok": False, "message": message}), 400

    set_gateway_url(gateway_url)
    return jsonify({"ok": True, "message": message})


@app.post("/api/prompt")
def submit_prompt():
    gateway_url = get_gateway_url() or OPENCLAW_GATEWAY_URL
    
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "message": "Prompt cannot be empty."}), 400

    history_text = _format_chat_history(payload.get("history"), current_prompt=prompt)
    
    try:
        result = send_to_openclaw(
            prompt=prompt,
            gateway_url=gateway_url,
            chat_history=history_text or None,
        )
        return jsonify(
            {
                "ok": True,
                "message": "Prompt accepted.",
                "result": result,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.post("/api/prompt/stream")
def submit_prompt_stream():
    gateway_url = get_gateway_url() or OPENCLAW_GATEWAY_URL
    
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "message": "Prompt cannot be empty."}), 400

    history_text = _format_chat_history(payload.get("history"), current_prompt=prompt)

    events: queue.Queue = queue.Queue()

    def emit(event: dict[str, Any]) -> None:
        _enqueue_progress(events, event)

    def execute() -> None:
        try:
            result = send_to_openclaw(
                prompt=prompt,
                gateway_url=gateway_url,
                chat_history=history_text or None,
                progress_callback=emit,
            )
        except Exception as exc:
            emit({"type": "error", "data": {"message": str(exc)}})
            emit({"type": "done", "data": {"ok": False}})
            events.put(None)
            return
        emit({"type": "final_result", "data": {"result": result}})
        emit({"type": "done", "data": {"ok": True}})
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


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "127.0.0.1")
    requested_port = int(os.getenv("PORT", "5001"))
    port = _find_available_port(host=host, preferred_port=requested_port)
    
    print("\n" + "="*70)
    print("🦅 Cobra Lite - Security Testing Interface")
    print("="*70)
    print(f"🌐 Web UI: http://{host}:{port}")
    print(f"🔧 Gateway: {OPENCLAW_GATEWAY_URL}")
    print("="*70 + "\n")
    
    if port != requested_port:
        print(f"⚠️  Port {requested_port} is busy. Using port {port} instead.\n")
    
    app.run(host=host, port=port, debug=debug_mode)
