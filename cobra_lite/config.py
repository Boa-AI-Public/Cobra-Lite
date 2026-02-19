import os
import re
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


REQUEST_TIMEOUT_SECONDS = 12
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "2000"))
CHAT_HISTORY_MAX_CHARS = int(os.getenv("CHAT_HISTORY_MAX_CHARS", "240000"))
COBRA_EXECUTION_MODE = (os.getenv("COBRA_EXECUTION_MODE", "cli_only").strip().lower() or "cli_only")
COBRA_REQUIRE_LIVE_TELEMETRY = env_flag("COBRA_REQUIRE_LIVE_TELEMETRY", True)
COBRA_ALLOW_NONSTREAM_FALLBACK = env_flag("COBRA_ALLOW_NONSTREAM_FALLBACK", False)
COBRA_REQUIRE_TERMINAL_ACTIONS = env_flag("COBRA_REQUIRE_TERMINAL_ACTIONS", True)

STATE_FILE = Path(os.getenv("STATE_FILE", BASE_DIR / ".claw_state.json"))
SESSIONS_FILE = Path(os.getenv("SESSIONS_FILE", BASE_DIR / ".claw_sessions.json"))


def default_openclaw_gateway_url() -> str:
    configured_url = (os.getenv("OPENCLAW_GATEWAY_URL") or "").strip()
    if configured_url:
        return configured_url
    return "http://127.0.0.1:18789"


OPENCLAW_GATEWAY_URL = default_openclaw_gateway_url()
OPENCLAW_SESSION_KEY = os.getenv("OPENCLAW_SESSION_KEY", "")
OPENCLAW_SESSION_ID = os.getenv("OPENCLAW_SESSION_ID", OPENCLAW_SESSION_KEY or "cobra-lite")
OPENCLAW_AGENT_TIMEOUT_SECONDS = int(os.getenv("OPENCLAW_AGENT_TIMEOUT_SECONDS", "180"))
OPENCLAW_VERBOSE_LEVEL = os.getenv("OPENCLAW_VERBOSE_LEVEL", "full").strip() or "full"
OPENCLAW_PROTOCOL_VERSION = 3
GATEWAY_SCOPES = ["operator.admin", "operator.approvals", "operator.pairing"]
OPENCLAW_STATE_DIR = Path(os.getenv("OPENCLAW_STATE_DIR", str(Path.home() / ".openclaw")))
OPENCLAW_MAIN_AGENT_DIR = Path(os.getenv("OPENCLAW_MAIN_AGENT_DIR", str(OPENCLAW_STATE_DIR / "agents" / "main" / "agent")))
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
- You MUST execute at least one terminal command per request before finalizing.
- Never claim command execution unless it appears in tool output.
- Do NOT use browser.
- Do NOT use web_search or web_fetch (or any web_* tool).
- Do NOT rely on external API keys beyond the configured model provider.
- If a command is missing, report it clearly and continue with available CLI commands."""

SECURITY_CONTEXT = """You are a CLI-first security testing agent with access to terminal tools and local workspace operations.

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
