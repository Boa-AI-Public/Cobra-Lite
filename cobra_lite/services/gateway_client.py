import asyncio
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from cobra_lite.config import (
    COBRA_ALLOW_NONSTREAM_FALLBACK,
    CLI_ONLY_EXTRA_SYSTEM_PROMPT,
    COBRA_EXECUTION_MODE,
    COBRA_REQUIRE_LIVE_TELEMETRY,
    COBRA_REQUIRE_TERMINAL_ACTIONS,
    DIAGNOSTIC_EXEC_LINE_RE,
    GATEWAY_SCOPES,
    OPENCLAW_AGENT_TIMEOUT_SECONDS,
    OPENCLAW_DEVICE_AUTH_PATH,
    OPENCLAW_GATEWAY_URL,
    OPENCLAW_IDENTITY_PATH,
    OPENCLAW_MAIN_AGENT_DIR,
    OPENCLAW_PROTOCOL_VERSION,
    OPENCLAW_SESSION_ID,
    OPENCLAW_SESSION_KEY,
    OPENCLAW_VERBOSE_LEVEL,
    REQUEST_TIMEOUT_SECONDS,
    SECURITY_CONTEXT,
)

MISSING_PROVIDER_KEY_RE = re.compile(r'No API key found for provider\s+"([^"]+)"', re.IGNORECASE)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
FALLBACK_EXEC_LINE_RE = re.compile(r"^\s*(?:[⚠️❌✅]?\s*)?(?:🛠️\s*)?Exec:", re.IGNORECASE)
AUTH_STORE_VERSION = 1
COBRA_ANTHROPIC_PROFILE_ID = "anthropic:cobra-lite"


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


def extract_missing_provider(message: str) -> str | None:
    text = str(message or "")
    match = MISSING_PROVIDER_KEY_RE.search(text)
    if not match:
        return None
    provider = (match.group(1) or "").strip().lower()
    return provider or None


def _atomic_write_json(pathname: Path, payload: dict[str, Any]) -> None:
    pathname.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = pathname.with_suffix(pathname.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(pathname)
    try:
        os.chmod(pathname, 0o600)
    except OSError:
        pass


def _resolve_auth_store_paths() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("OPENCLAW_AGENT_DIR", "PI_CODING_AGENT_DIR"):
        raw = str(os.getenv(env_name) or "").strip()
        if raw:
            candidates.append(Path(raw))
    candidates.append(OPENCLAW_MAIN_AGENT_DIR)

    deduped: list[Path] = []
    seen: set[str] = set()
    for agent_dir in candidates:
        auth_path = (Path(agent_dir).expanduser().resolve() / "auth-profiles.json")
        key = str(auth_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(auth_path)
    return deduped


def _sync_anthropic_auth_profile(api_key: str) -> bool:
    key = str(api_key or "").strip()
    if not key:
        return False

    desired_profile = {
        "type": "api_key",
        "provider": "anthropic",
        "key": key,
    }
    wrote_any = False

    for auth_path in _resolve_auth_store_paths():
        existing: dict[str, Any] = {}
        if auth_path.exists():
            try:
                loaded = json.loads(auth_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError):
                existing = {}

        profiles_obj = existing.get("profiles")
        profiles = profiles_obj if isinstance(profiles_obj, dict) else {}
        current_profile = profiles.get(COBRA_ANTHROPIC_PROFILE_ID)
        has_same_profile = isinstance(current_profile, dict) and all(
            current_profile.get(k) == v for k, v in desired_profile.items()
        )

        order_obj = existing.get("order")
        order = order_obj if isinstance(order_obj, dict) else {}
        current_order_raw = order.get("anthropic")
        current_order = (
            [str(item).strip() for item in current_order_raw if str(item).strip()]
            if isinstance(current_order_raw, list)
            else []
        )
        desired_order = [COBRA_ANTHROPIC_PROFILE_ID] + [pid for pid in current_order if pid != COBRA_ANTHROPIC_PROFILE_ID]
        has_same_order = current_order == desired_order

        if has_same_profile and has_same_order and auth_path.exists():
            continue

        next_store: dict[str, Any] = existing if isinstance(existing, dict) else {}
        next_store["version"] = int(next_store.get("version") or AUTH_STORE_VERSION)
        if next_store["version"] <= 0:
            next_store["version"] = AUTH_STORE_VERSION

        next_profiles = next_store.get("profiles")
        if not isinstance(next_profiles, dict):
            next_profiles = {}
            next_store["profiles"] = next_profiles
        next_profiles[COBRA_ANTHROPIC_PROFILE_ID] = desired_profile

        next_order = next_store.get("order")
        if not isinstance(next_order, dict):
            next_order = {}
            next_store["order"] = next_order
        next_order["anthropic"] = desired_order

        _atomic_write_json(auth_path, next_store)
        wrote_any = True

    return wrote_any


def send_to_openclaw(
    prompt: str,
    gateway_url: str,
    session_id: str | None = None,
    anthropic_api_key: str | None = None,
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
    enforcement_suffix = ""
    if COBRA_REQUIRE_TERMINAL_ACTIONS:
        enforcement_suffix = (
            "\n\nExecution contract:\n"
            "- Execute at least one terminal command for this request before finalizing.\n"
            "- Never claim a command was run unless it appears in tool output."
        )
    full_prompt = f"{SECURITY_CONTEXT}\n\n{prompt}{enforcement_suffix}"

    configured_key = (anthropic_api_key or "").strip()
    if configured_key:
        try:
            _sync_anthropic_auth_profile(configured_key)
        except Exception:
            # Keep prompt execution resilient even if profile sync fails.
            pass

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

    def _collect_payload_texts(payload_obj: dict[str, Any]) -> list[str]:
        texts: list[str] = []

        def _extend_from_payloads(payloads: Any) -> None:
            if not isinstance(payloads, list):
                return
            for item in payloads:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    texts.append(text)

        # OpenClaw payload shape can be either {result:{payloads:[...]}} or {payloads:[...]}.
        _extend_from_payloads(payload_obj.get("payloads"))
        result = payload_obj.get("result")
        if isinstance(result, dict):
            _extend_from_payloads(result.get("payloads"))

        return texts

    def _extract_final_observation(payload_obj: dict[str, Any], *, fallback_text: str = "") -> str:
        texts = _collect_payload_texts(payload_obj)
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
        if not summary:
            result = payload_obj.get("result")
            if isinstance(result, dict):
                summary = str(result.get("summary") or "").strip()
        if summary:
            return _clean_final_observation(summary)
        return _clean_final_observation("Task completed.")

    def _extract_exec_command(line: str) -> str:
        command = re.sub(r"^\s*(?:[⚠️❌✅]?\s*)?(?:🛠️\s*)?Exec:\s*", "", str(line or ""), flags=re.IGNORECASE).strip()
        return command or "(no command)"

    def _emit_cli_actions_from_logs(log_text: str, parent_execution_id: str, *, start_index: int = 2) -> int:
        if not progress_callback:
            return 0

        lines = [ANSI_ESCAPE_RE.sub("", str(line or "")).replace("\r", "").strip() for line in str(log_text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return 0

        actions: list[dict[str, Any]] = []
        current_action: dict[str, Any] | None = None

        for line in lines:
            if DIAGNOSTIC_EXEC_LINE_RE.match(line) or FALLBACK_EXEC_LINE_RE.match(line):
                current_action = {
                    "command": _extract_exec_command(line),
                    "output_lines": [],
                }
                actions.append(current_action)
                continue

            if current_action is not None:
                current_action["output_lines"].append(line)

        if not actions:
            return 0

        for offset, action in enumerate(actions):
            action_index = start_index + offset
            action_execution_id = f"{parent_execution_id}-action-{action_index}"
            output_lines = action.get("output_lines") if isinstance(action.get("output_lines"), list) else []
            output = "\n".join(str(item) for item in output_lines if str(item).strip()).strip() or "(no output)"
            if len(output) > 5000:
                output = output[:5000].rstrip() + "\n...(truncated)"

            progress_callback(
                {
                    "type": "tool_start",
                    "data": {
                        "tool_name": "terminal-exec",
                        "command": str(action.get("command") or "(no command)"),
                        "execution_id": action_execution_id,
                        "action_index_1based": action_index,
                    },
                }
            )
            progress_callback(
                {
                    "type": "tool_execution",
                    "data": {
                        "tool_name": "terminal-exec",
                        "command": str(action.get("command") or "(no command)"),
                        "tool_output": output,
                        "execution_id": action_execution_id,
                        "action_index_1based": action_index,
                    },
                }
            )

        return len(actions)

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
            resolved_verbose_level = OPENCLAW_VERBOSE_LEVEL
            if COBRA_REQUIRE_TERMINAL_ACTIONS and resolved_verbose_level.strip().lower() in {"off", "none", "0", "false"}:
                resolved_verbose_level = "full"

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
                            "verboseLevel": resolved_verbose_level,
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
                        if COBRA_REQUIRE_TERMINAL_ACTIONS and tool_counter <= 0:
                            raise Exception(
                                "No terminal actions were emitted by the gateway run. "
                                "Cobra Lite requires visible command execution telemetry."
                            )
                        final_observation = _extract_final_observation(payload, fallback_text=latest_assistant_text)
                        return {"final_observation": final_observation}

        return asyncio.run(_run())

    def _send_via_cli() -> dict[str, Any]:
        execution_id = "gateway-agent-cli"
        verbose_switch = "off" if OPENCLAW_VERBOSE_LEVEL.strip().lower() in {"off", "none", "0", "false"} else "on"
        command = [
            "openclaw",
            "agent",
            "--session-id",
            active_session_id,
            "--message",
            full_prompt,
            "--json",
            "--verbose",
            verbose_switch,
            "--timeout",
            str(OPENCLAW_AGENT_TIMEOUT_SECONDS),
        ]
        if shutil.which("stdbuf"):
            command = ["stdbuf", "-oL", "-eL", *command]
        command_text = (
            "openclaw agent "
            f"--session-id {active_session_id} "
            "--message <omitted> "
            "--json "
            f"--verbose {verbose_switch} "
            f"--timeout {OPENCLAW_AGENT_TIMEOUT_SECONDS}"
        )

        if progress_callback:
            progress_callback(
                {
                    "type": "tool_start",
                    "data": {
                        "tool_name": "gateway-agent",
                        "command": command_text,
                        "execution_id": execution_id,
                        "action_index_1based": 1,
                    },
                }
            )

        env = os.environ.copy()
        configured_key = (anthropic_api_key or "").strip()
        if configured_key:
            env["ANTHROPIC_API_KEY"] = configured_key
            env["CLAUDE_API_KEY"] = configured_key

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        line_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        active_streams = 0
        closed_streams = 0
        live_action_count = 0
        current_action_id: str | None = None
        current_action_step: int | None = None
        current_action_command = ""
        current_action_output_lines: list[str] = []

        def _emit_running_action_update(line: str) -> None:
            if not progress_callback or not current_action_id or current_action_step is None:
                return
            progress_callback(
                {
                    "type": "tool_update",
                    "data": {
                        "tool_name": "terminal-exec",
                        "command": current_action_command or "(no command)",
                        "tool_output": line,
                        "execution_id": current_action_id,
                        "action_index_1based": current_action_step,
                    },
                }
            )

        def _finish_current_action() -> None:
            nonlocal current_action_id, current_action_step, current_action_command, current_action_output_lines
            if not progress_callback or not current_action_id or current_action_step is None:
                current_action_id = None
                current_action_step = None
                current_action_command = ""
                current_action_output_lines = []
                return
            output = "\n".join(current_action_output_lines).strip() or "(no output)"
            if len(output) > 5000:
                output = output[:5000].rstrip() + "\n...(truncated)"
            progress_callback(
                {
                    "type": "tool_execution",
                    "data": {
                        "tool_name": "terminal-exec",
                        "command": current_action_command or "(no command)",
                        "tool_output": output,
                        "execution_id": current_action_id,
                        "action_index_1based": current_action_step,
                    },
                }
            )
            current_action_id = None
            current_action_step = None
            current_action_command = ""
            current_action_output_lines = []

        def _stream_reader(stream: Any, source: str) -> None:
            try:
                for raw_line in iter(stream.readline, ""):
                    line_queue.put((source, raw_line))
            finally:
                line_queue.put((source, None))
                try:
                    stream.close()
                except Exception:
                    pass

        for source, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            if stream is None:
                continue
            active_streams += 1
            threading.Thread(target=_stream_reader, args=(stream, source), daemon=True).start()

        try:
            while closed_streams < active_streams:
                try:
                    source, raw_line = line_queue.get(timeout=0.2)
                except queue.Empty:
                    if process.poll() is not None and closed_streams >= active_streams:
                        break
                    continue

                if raw_line is None:
                    closed_streams += 1
                    continue

                if source == "stdout":
                    stdout_chunks.append(raw_line)
                    continue

                stderr_chunks.append(raw_line)
                clean_line = ANSI_ESCAPE_RE.sub("", str(raw_line or "")).replace("\r", "").strip()
                if not clean_line:
                    continue

                if DIAGNOSTIC_EXEC_LINE_RE.match(clean_line) or FALLBACK_EXEC_LINE_RE.match(clean_line):
                    _finish_current_action()
                    live_action_count += 1
                    step_index = live_action_count + 1
                    current_action_id = f"{execution_id}-action-{step_index}"
                    current_action_step = step_index
                    current_action_command = _extract_exec_command(clean_line)
                    current_action_output_lines = []
                    if progress_callback:
                        progress_callback(
                            {
                                "type": "tool_start",
                                "data": {
                                    "tool_name": "terminal-exec",
                                    "command": current_action_command,
                                    "execution_id": current_action_id,
                                    "action_index_1based": step_index,
                                },
                            }
                        )
                    continue

                if current_action_id is not None:
                    current_action_output_lines.append(clean_line)
                    _emit_running_action_update(clean_line)

            return_code = process.wait(timeout=OPENCLAW_AGENT_TIMEOUT_SECONDS + 15)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=5)
        finally:
            _finish_current_action()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        if return_code != 0:
            error_text = (stderr_text or stdout_text or "").strip()
            if progress_callback and error_text:
                progress_callback(
                    {
                        "type": "tool_execution",
                        "data": {
                            "tool_name": "gateway-agent",
                            "command": command_text,
                            "tool_output": error_text,
                            "execution_id": execution_id,
                            "is_error": True,
                            "action_index_1based": 1,
                        },
                    }
                )
            raise Exception(f"Gateway CLI error: {error_text or f'exit code {return_code}'}")

        raw_stdout = stdout_text
        raw = raw_stdout.strip()
        if not raw:
            raise Exception("Gateway CLI returned an empty response.")

        def _decode_cli_json(raw_text: str) -> tuple[dict[str, Any], str, str]:
            text = str(raw_text or "").strip()
            if not text:
                raise Exception("Gateway CLI returned an empty response.")
            try:
                decoded = json.loads(text)
                if isinstance(decoded, dict):
                    return decoded, "", ""
            except json.JSONDecodeError:
                pass

            decoder = json.JSONDecoder()
            for match in re.finditer(r"\{", text):
                start = match.start()
                try:
                    candidate, consumed = decoder.raw_decode(text[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    return candidate, text[:start].strip(), text[start + consumed :].strip()
            raise Exception("Gateway CLI returned non-JSON output.")

        parsed, stdout_prefix, stdout_suffix = _decode_cli_json(raw)

        if not isinstance(parsed, dict):
            raise Exception("Gateway CLI returned non-JSON output.")

        def _resolve_cli_status(payload_obj: dict[str, Any]) -> str:
            direct_keys = ("status", "state", "phase")
            for key in direct_keys:
                value = str(payload_obj.get(key) or "").strip()
                if value:
                    return value
            result_obj = payload_obj.get("result")
            if isinstance(result_obj, dict):
                for key in direct_keys:
                    value = str(result_obj.get(key) or "").strip()
                    if value:
                        return value
            meta_obj = payload_obj.get("meta")
            if isinstance(meta_obj, dict):
                aborted = meta_obj.get("aborted")
                if aborted is False:
                    return "completed"
                if aborted is True:
                    return "aborted"
            if _collect_payload_texts(payload_obj):
                return "completed"
            return "success" if bool(payload_obj) else "unknown"

        def _build_cli_terminal_output(payload_obj: dict[str, Any], stderr_text: str) -> str:
            lines: list[str] = []
            run_id = str(payload_obj.get("runId") or payload_obj.get("id") or "").strip()
            if run_id:
                lines.append(f"Run id: {run_id}")

            status_value = _resolve_cli_status(payload_obj)
            lines.append(f"Run status: {status_value}")

            meta_obj = payload_obj.get("meta")
            if isinstance(meta_obj, dict):
                duration_ms = meta_obj.get("durationMs")
                if isinstance(duration_ms, (int, float)) and duration_ms >= 0:
                    lines.append(f"Duration: {int(duration_ms)} ms")
                agent_meta = meta_obj.get("agentMeta")
                if isinstance(agent_meta, dict):
                    provider = str(agent_meta.get("provider") or "").strip()
                    model = str(agent_meta.get("model") or "").strip()
                    if provider or model:
                        lines.append(f"Model: {(provider + ' / ' + model).strip(' /')}")
                    usage = agent_meta.get("usage")
                    if isinstance(usage, dict):
                        total = usage.get("total")
                        if isinstance(total, (int, float)):
                            lines.append(f"Tokens: {int(total)}")

            payload_texts = _collect_payload_texts(payload_obj)
            if payload_texts:
                lines.append("")
                lines.append("Result preview:")
                lines.append("\n\n".join(payload_texts[:2]))

            # Suppress noisy failover diagnostics when the CLI call ultimately succeeds.
            stderr_clean = str(stderr_text or "").strip()
            if stderr_clean and "falling back to embedded" not in stderr_clean.lower():
                lines.append("")
                lines.append("Gateway diagnostics:")
                lines.append(stderr_clean[:1200])

            output = "\n".join(lines).strip()
            max_len = 8000
            if len(output) > max_len:
                output = output[:max_len].rstrip() + "\n...(truncated)"
            return output

        run_status = _resolve_cli_status(parsed)
        cli_terminal_output = _build_cli_terminal_output(parsed, stderr_text)
        cli_action_count = live_action_count
        if cli_action_count <= 0:
            cli_action_logs = "\n".join(
                segment
                for segment in (
                    stdout_prefix,
                    stdout_suffix,
                    stderr_text,
                )
                if str(segment).strip()
            )
            cli_action_count = _emit_cli_actions_from_logs(cli_action_logs, execution_id, start_index=2)

        if COBRA_REQUIRE_TERMINAL_ACTIONS and cli_action_count <= 0 and not COBRA_ALLOW_NONSTREAM_FALLBACK:
            raise Exception(
                "No terminal command actions were emitted by the fallback agent run. "
                "Cobra Lite requires live, per-command execution telemetry."
            )

        if progress_callback:
            progress_callback(
                {
                    "type": "tool_update",
                    "data": {
                        "tool_name": "gateway-agent",
                        "command": command_text,
                        "tool_output": cli_terminal_output,
                        "execution_id": execution_id,
                        "action_index_1based": 1,
                    },
                }
            )
            if cli_action_count <= 0:
                progress_callback(
                    {
                        "type": "tool_update",
                        "data": {
                            "tool_name": "gateway-agent",
                            "command": command_text,
                            "tool_output": "No explicit terminal actions were emitted by the gateway for this run.",
                            "execution_id": execution_id,
                            "action_index_1based": 1,
                        },
                    }
                )
            progress_callback(
                {
                    "type": "tool_execution",
                    "data": {
                        "tool_name": "gateway-agent",
                        "command": command_text,
                        "tool_output": f"Run status: {run_status}",
                        "execution_id": execution_id,
                        "action_index_1based": 1,
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
    ws_failure_message = ""

    try:
        return _send_via_gateway_ws()
    except Exception as ws_error:
        if isinstance(ws_error, _PolicyViolationError):
            raise
        ws_failure_message = str(ws_error)
        errors.append(f"ws: {ws_failure_message}")

        missing_provider = extract_missing_provider(ws_failure_message)
        if missing_provider == "anthropic" and configured_key:
            try:
                _sync_anthropic_auth_profile(configured_key)
            except Exception:
                pass
            try:
                return _send_via_gateway_ws()
            except Exception as ws_retry_error:
                if isinstance(ws_retry_error, _PolicyViolationError):
                    raise
                ws_failure_message = str(ws_retry_error)
                errors.append(f"ws-retry: {ws_failure_message}")

    if COBRA_REQUIRE_LIVE_TELEMETRY and not COBRA_ALLOW_NONSTREAM_FALLBACK:
        reason = ws_failure_message or "unknown websocket transport failure"
        raise Exception(
            "Live gateway telemetry is required, but WebSocket transport is unavailable. "
            f"{reason}. Configure gateway auth/connectivity and retry."
        )

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
