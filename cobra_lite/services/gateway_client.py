import asyncio
import base64
import json
import os
import re
import subprocess
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlparse

from cobra_lite.config import (
    CLI_ONLY_EXTRA_SYSTEM_PROMPT,
    COBRA_EXECUTION_MODE,
    DIAGNOSTIC_EXEC_LINE_RE,
    GATEWAY_SCOPES,
    OPENCLAW_AGENT_TIMEOUT_SECONDS,
    OPENCLAW_DEVICE_AUTH_PATH,
    OPENCLAW_GATEWAY_URL,
    OPENCLAW_IDENTITY_PATH,
    OPENCLAW_PROTOCOL_VERSION,
    OPENCLAW_SESSION_ID,
    OPENCLAW_SESSION_KEY,
    OPENCLAW_VERBOSE_LEVEL,
    REQUEST_TIMEOUT_SECONDS,
    SECURITY_CONTEXT,
)


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
    import urllib.error
    import urllib.request

    test_url = f"{gateway_url.rstrip('/')}/health"
    try:
        req = urllib.request.Request(test_url, method="GET")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.getcode() == 200:
                return True, "Gateway is reachable."
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"Cannot reach gateway at {gateway_url}: {str(e)}"

    return False, "Unknown error connecting to gateway."


def send_to_openclaw(
    prompt: str,
    gateway_url: str,
    session_id: str | None = None,
    progress_callback: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Send a security testing prompt to the gateway agent.

    Cobra Lite runs in a CLI-first mode by default and prioritizes
    terminal execution plus local workspace operations.
    """
    import urllib.error
    import urllib.request

    class _PolicyViolationError(Exception):
        pass

    def _tool_disallowed_for_mode(tool_name: str) -> bool:
        if COBRA_EXECUTION_MODE not in {"cli_only", "cli", "terminal_only"}:
            return False
        name = (tool_name or "").strip().lower()
        if not name:
            return False
        return name == "browser" or name.startswith("web_")

    active_session_id = (session_id or OPENCLAW_SESSION_KEY or OPENCLAW_SESSION_ID).strip() or OPENCLAW_SESSION_ID
    full_prompt = f"{SECURITY_CONTEXT}\n\n{prompt}"

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
                            "key": active_session_id,
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
                            "sessionId": active_session_id,
                            "sessionKey": active_session_id,
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
            active_session_id,
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
            "stream": True,
            "sessionKey": active_session_id,
        }

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
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
                            "data": data,
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
                            progress_callback(
                                {
                                    "type": "tool_start",
                                    "data": {
                                        "tool_name": data.get("tool", "unknown"),
                                        "command": data.get("description", ""),
                                        "execution_id": data.get("id", ""),
                                    },
                                }
                            )
                    elif event_type == "tool_result":
                        if progress_callback:
                            progress_callback(
                                {
                                    "type": "tool_execution",
                                    "data": {
                                        "tool_name": data.get("tool", "unknown"),
                                        "tool_output": data.get("output", ""),
                                        "execution_id": data.get("id", ""),
                                    },
                                }
                            )

                except json.JSONDecodeError:
                    continue

            return {
                "final_observation": result_text or "Task completed.",
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


def effective_gateway_url(saved_gateway_url: str | None) -> str:
    return (saved_gateway_url or OPENCLAW_GATEWAY_URL).strip() or OPENCLAW_GATEWAY_URL
