# CodeOptimizer

An agentic code-review system powered by **LangGraph** and a **local Ollama** model.  
Two specialised agents run in sequence:

1. **Peer Reviewer** – audits code quality, readability, edge cases and best practices.
2. **Manager Reviewer** – reviews architecture, technical debt, risk and priorities (informed by the peer review).

```
START ──► peer_reviewer ──► manager_reviewer ──► END
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| [Ollama](https://ollama.com) | any recent release |

> **Mac M1 Pro**: Ollama runs natively on Apple Silicon – download the macOS package from <https://ollama.com/download>.

---

## Quick-start

### 1 – Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 – Start Ollama and pull a model

```bash
# pull a capable general-purpose model (≈2 GB)
ollama pull llama3.2

# or a code-focused model
ollama pull codellama
```

### 3 – Run a review

```bash
# review a file
python main.py --file path/to/your_script.py --language Python

# review an inline snippet
python main.py --code "def add(a, b): return a + b" --language Python

# use a different model
python main.py --file my_code.ts --language TypeScript --model codellama

# point at a non-default Ollama instance
python main.py --file my_code.py --ollama-url http://192.168.1.10:11434
```

### 4 – Read piped input (stdin)

```bash
cat my_code.py | python main.py --language Python
```

---

## Project layout

```
CodeOptimizer/
├── main.py              # CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── state.py         # ReviewState TypedDict
│   ├── agents.py        # Peer & Manager reviewer agent factories
│   └── workflow.py      # LangGraph StateGraph definition
└── tests/
    └── test_workflow.py # Unit tests (Ollama mocked)
```

---

## Running the tests

```bash
pytest tests/ -v
```

---

## Configuration reference

| CLI flag | Default | Description |
|---|---|---|
| `--file PATH` | – | Source file to review |
| `--code CODE` | – | Inline code snippet |
| `--language LANG` | *(empty)* | Language hint passed to both agents |
| `--model TAG` | `llama3.2` | Ollama model tag |
| `--ollama-url URL` | `http://localhost:11434` | Ollama server base URL |
