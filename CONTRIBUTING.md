# Contributing to NazoratAI

Thank you for your interest in contributing to **NazoratAI**!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zizx4uM/NazoratAI.git
   cd NazoratAI
   ```

2. **Create and activate a virtual environment (Python 3.10+):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies in editable mode with development tools:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Running Tests

Before submitting any code changes or pull requests, verify that all unit and integration tests pass:

```bash
pytest -v tests/
```

## Guidelines

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions for Python code formatting.
- Add type annotations (`typing`) to all new function signatures and data contracts.
- Ensure any added features include unit tests under `tests/`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
