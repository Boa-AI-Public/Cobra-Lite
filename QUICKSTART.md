# Cobra Lite - Quick Start

## 3-Step Setup

### 1. Start Gateway

```bash
# Install gateway CLI (if not already installed)
npm install -g openclaw

# Start the gateway
openclaw gateway start
```

✅ Gateway should be running at `http://127.0.0.1:18789`

### 2. Install Cobra Lite

```bash
cd /home/willi/Boa-Ai/claw-bot-pentester
pip install -r requirements.txt
```

### 3. Launch Cobra Lite

```bash
# Optional (default): enforce CLI-only tool policy
export COBRA_EXECUTION_MODE=cli_only

python app.py
```

🎉 **That's it!** Open your browser to the URL shown (usually `http://127.0.0.1:5001`)

## First Test

Try this prompt in the Cobra Lite interface:

```
Scan scanme.nmap.org for open ports and services
```

Or for web application testing:

```
Test the demo site at https://demo.testfire.net for common vulnerabilities
```

## Common Commands

| Task | Command |
|------|---------|
| Start gateway | `openclaw gateway start` |
| Check gateway status | `openclaw gateway status` |
| Stop gateway | `openclaw gateway stop` |
| Run Cobra Lite | `python app.py` |
| Custom port | `PORT=8000 python app.py` |

## Troubleshooting

### "Cannot reach gateway"
```bash
openclaw gateway status
openclaw gateway start
```

### "Port already in use"
App auto-selects next available port. Check startup message.

### No security tools available
Install tools on the same system running the gateway:
```bash
# Ubuntu/Debian
sudo apt install nmap curl nikto

# Install nuclei
go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
```

## Example Tests

### 1. Basic Recon
```
Find subdomains for example.com and check which are live
```

### 2. Port Scan
```
Scan example.com ports 1-10000 and identify services
```

### 3. Web App Test
```
Test https://example.com/login for:
- XSS
- SQL injection
- CSRF
- Weak passwords
```

### 4. API Security
```
Discover and test all endpoints at https://api.example.com/v1
```

## Pro Tips

1. **Context Matters** - Cobra Lite maintains chat history. Build on previous results.
2. **Be Specific** - More detail = better results
3. **Tool Selection** - Mention specific tools if you prefer them (nmap, nuclei, etc.)
4. **Sub-Agents** - Ask Cobra Lite to "spawn an agent for X" for parallel work

---

Start chatting with Cobra Lite.
