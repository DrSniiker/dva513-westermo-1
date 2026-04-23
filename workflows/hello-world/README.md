# uv version
`uv 0.11.7 (9d177269e 2026-04-15 x86_64-unknown-linux-gnu)`

# Virtual environment
the virtual environment is located in .venv/

# Activate the environment
* Linux:
```bash
source .venv/bin/activate
```

* Windows (powershell):
```ps1
.venv/bin/activate.ps1
```

* Windows (cmd):
```cmd
.venv/bin/activate.bat
```

# Workflow
Once activated, `uv` will automatically manage dependencies.
* Sync dependencies:
```bash
uv sync
```
* Run a script from within the environment without activation
```bash
uv run main.py
```

