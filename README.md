# ClawBot Pentester 🦅🔐

A web-based security testing interface powered by **OpenClaw** - your AI-driven penetration testing assistant.

## What It Does

ClawBot Pentester provides a chat-style interface for security testing tasks, leveraging OpenClaw's powerful agent system which includes:

- **Browser Automation** - Test web applications, forms, authentication flows, session handling
- **Terminal Execution** - Run security tools like nmap, nuclei, nikto, ffuf, curl, and more
- **Web Research** - Search for CVEs, exploits, and security documentation
- **Multi-Agent Coordination** - Spawn sub-agents for different testing phases
- **File Operations** - Generate reports, save findings, organize results

## Quick Start

### Prerequisites

1. **OpenClaw Gateway** must be running. Install from:
   ```bash
   npm install -g openclaw
   openclaw gateway start
   ```
   
   Default gateway URL: `http://127.0.0.1:18789`

### Installation

```bash
# Clone or navigate to the project
cd claw-bot-pentester

# Install dependencies
python -m pip install -r requirements.txt

# Run the app
python app.py
```

The app starts on `127.0.0.1:5001` by default. If that port is taken, it automatically uses the next free port.

### Configuration

Optional environment variables:

```bash
# Custom port
PORT=7000 python app.py

# Expose on network
HOST=0.0.0.0 python app.py

# Custom OpenClaw Gateway URL
OPENCLAW_GATEWAY_URL=http://your-gateway:18789 python app.py

# Flask secret key (recommended for production)
FLASK_SECRET_KEY="your-secret-key" python app.py
```

Or create a `.env` file:

```env
FLASK_SECRET_KEY=your-secret-key
OPENCLAW_GATEWAY_URL=http://127.0.0.1:18789
PORT=5001
HOST=127.0.0.1
```

## Usage

1. **Connect to OpenClaw Gateway**
   - On first load, enter your OpenClaw Gateway URL (default: `http://127.0.0.1:18789`)
   - The app verifies connectivity and saves your settings

2. **Start Testing**
   - Enter security testing prompts in natural language
   - Examples:
     ```
     Scan example.com for open ports and services
     
     Test the login form at https://example.com/login for common vulnerabilities
     
     Find subdomains for example.com and check which are live
     
     Run a nuclei scan against example.com with default templates
     ```

3. **View Execution Details**
   - Expand "Execution details" to see:
     - Tools being executed
     - Command outputs
     - Real-time progress
   - Chat history is preserved across sessions

4. **Settings**
   - Click the ⚙️ icon to update your Gateway URL anytime

## Features

### Security Testing Capabilities

Through OpenClaw, you get:

- **Reconnaissance**: subdomain enumeration, port scanning, tech detection
- **Vulnerability Testing**: XSS, SQLi, CSRF, authentication bypasses
- **API Testing**: endpoint discovery, parameter fuzzing, auth testing
- **Web App Testing**: form validation, session handling, cookie analysis
- **Network Scanning**: service detection, banner grabbing
- **Report Generation**: automated documentation of findings

### Multi-Agent Architecture

ClawBot can:
- Spawn sub-agents for parallel testing
- Coordinate different phases (recon → scanning → exploitation)
- Aggregate results from multiple tools
- Maintain conversation context across complex workflows

## Architecture

```
┌─────────────────────────┐
│   ClawBot Web UI        │
│   (Flask + JavaScript)  │
└───────────┬─────────────┘
            │ HTTP
            ↓
┌─────────────────────────┐
│   OpenClaw Gateway      │
│   (Agent Orchestration) │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ↓               ↓
┌─────────┐   ┌─────────┐
│ Browser │   │ Terminal│
│  Tools  │   │  Tools  │
└─────────┘   └─────────┘
```
## Troubleshooting

### No Tools Available

If security tools aren't working, ensure they're installed on the system running OpenClaw Gateway:
```bash
# Example: Install common tools
sudo apt update
sudo apt install nmap curl nikto
go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
```


### Extending

To add custom security workflows:

1. Modify `SECURITY_CONTEXT` in `cobra_lite/config.py` to guide the agent
2. Add custom prompts or templates
3. Integrate additional tools via OpenClaw skills

### API Integration

The app communicates with OpenClaw Gateway via:
- `POST /api/chat` - Send messages and receive streaming responses
- Tool calls and results are handled by OpenClaw automatically

## Security & Responsibility

⚠️ **Important**: This tool is for **authorized security testing only**.

- Only test systems you own or have explicit permission to test
- Respect bug bounty program rules and scope
- Be aware of rate limits and avoid causing service disruption
- Document all findings responsibly
- Never use for malicious purposes


## Contributing

Built on top of OpenClaw. For issues or contributions:

- OpenClaw: https://github.com/openclaw/openclaw
- ClawBot: Create issues in this repo

**Made with 🦅 by the OpenClaw community**
