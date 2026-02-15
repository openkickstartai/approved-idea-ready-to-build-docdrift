# DocDrift 🌊

> Detect cognitive drift between your documentation and code using semantic analysis.

DocDrift parses Python source via AST, extracts semantic tokens from identifiers,
and compares them against docstrings using cosine similarity — surfacing functions
where docs no longer match what the code actually does.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Analyze one or more Python files
python main.py mymodule.py

# JSON output for CI pipelines
python main.py mymodule.py --json

# Fail CI if drift detected (exit code 1)
python main.py mymodule.py --fail-on-drift --threshold 0.6
```

### As a Library

```python
from docdrift import analyze

source = open("mymodule.py").read()
for report in analyze(source, threshold=0.7):
    print(report["name"], report["status"], f"{report['drift']:.0%}")
```

### Output Example

```
📄 mymodule.py
──────────────────────────────────────────────────
  🟢 function 'calculate_sum': drift=48.5%
  🔴 function 'send_email': drift=100.0%
  🟡 No docstring for function 'helper'

Summary: 2/3 elements with drift issues
```

## How It Works

1. **Parse** — AST extracts functions/classes, their docstrings, arg names, and body identifiers
2. **Tokenize** — CamelCase/snake_case identifiers are split into semantic word tokens
3. **Compare** — Cosine similarity between docstring tokens and code tokens
4. **Report** — Drift score 0%=aligned, 100%=completely drifted

## Run Tests

```bash
pytest test_docdrift.py -v
```

## License

MIT
