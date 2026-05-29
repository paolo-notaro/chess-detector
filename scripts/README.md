# scripts/

One-off, runnable scripts that are *not* part of the importable
`chess_detector` package. This directory is intentionally **not** a Python
package: there is no `__init__.py` and nothing in `pyproject.toml` ships
these files as console entry points.

Files here are expected to be invoked directly with `poetry run python`, e.g.

```bash
poetry run python scripts/calibration_demo.py
```

Anything that needs to be installed and exposed as a CLI lives in
[`src/chess_detector/`](../src/chess_detector/) and is wired up through
`[tool.poetry.scripts]` in `pyproject.toml`.
