# SC4020-Project-2

## Setup (Python venv)

Requires Python 3.11+. From the project root:

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Notebooks:** in VS Code, click **Select Kernel** → **Python Environments** → `.venv`.
