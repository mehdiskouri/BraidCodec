

---

## Plan: Phase 0 — Project Scaffold & Algebra Migration

**TL;DR**: Initialize git, create PEP 517 src-layout package, MOVE algebra into `src/braidcodec/algebra/` (deleting the old algebra dir), fix all typing/lint issues for `mypy --strict` + strict ruff, get 459 tests green, push with CI.

---

### Step 0.1 — Git Init & Root Scaffold

1. `git init` in BraidCodec
2. Create `.gitignore` — Python template (`__pycache__/`, `*.pyc`, .venv, `dist/`, `build/`, `*.egg-info/`, `.mypy_cache/`, .pytest_cache, `.ruff_cache/`, `htmlcov/`, `coverage.xml`, `.coverage`)
3. Create `LICENSE` (MIT), skeleton `README.md`, `CHANGELOG.md`
4. First commit: `chore: initial scaffold`

### Step 0.2 — pyproject.toml

Full config per PRD §11:
- **Build**: setuptools>=75.0, src-layout with `[tool.setuptools.packages.find] where=["src"]`
- **Deps**: `numpy>=1.26,<3`, `msgpack>=1.0,<2`, `blake3>=1.0,<2`
- **Extras**: `[quantum]` (qiskit), `[cli]` (click), `[dev]` (pytest, pytest-cov, pytest-benchmark, hypothesis, mypy, ruff, pre-commit)
- **Ruff**: `target-version="py313"`, `line-length=99`, `select=["E","F","W","I","N","UP","B","A","SIM","TCH","RUF","ARG","PTH"]`
- **Mypy**: `strict=true`, `python_version="3.13"`, overrides ignore `qiskit.*`, `msgpack.*`
- **Coverage**: `fail_under=95`, `branch=true`

### Step 0.3 — Package Directory Structure

```
src/braidcodec/
├── __init__.py          (imports __version__ only for now)
├── _version.py          (__version__ = "0.1.0")
├── _types.py            (GeneratorSeq, SectorName, JonesValue, ByteChunk, tolerance Final consts)
├── py.typed             (PEP 561 marker, empty)
├── algebra/
│   ├── __init__.py      (re-exports all 96 public symbols with `as` for PEP 484)
│   ├── _pauli_compat.py (~15-line Pauli stub)
│   ├── tsr_constants.py
│   ├── braid_equations.py
│   ├── yang_baxter.py
│   └── fermion_bounds.py
├── codec/__init__.py    (empty placeholder)
├── crypto/__init__.py   (empty placeholder)
└── cli/__init__.py      (empty placeholder)

tests/
├── __init__.py
├── conftest.py          (empty)
└── algebra/
    ├── __init__.py
    ├── test_tsr_constants.py
    ├── test_braid_equations.py
    ├── test_yang_baxter.py
    └── test_fermion_bounds.py
```

### Step 0.4 — Vendored Pauli Fallback

Create `src/braidcodec/algebra/_pauli_compat.py` — a ~15-line class. The entire qiskit Pauli surface used by algebra is: `Pauli(label_string)` constructor returning an object. No methods/properties/operators are ever called on it. Tests use `isinstance(obj, Pauli)`. This stub is trivial.

### Step 0.5 — MOVE Algebra (delete old tree)

**Move** (not copy) all 4 source files from src → `src/braidcodec/algebra/` and all 4 test files from tests → `tests/algebra/`. Then **delete** the entire algebra directory (including the dead requirements.txt with unused scipy/matplotlib/sympy/cupy deps).

### Step 0.6 — Fix Internal Cross-Imports

| File | Current import | New import |
|---|---|---|
| `braid_equations.py` | `from .tsr_constants import C_CONSTANT` (already relative) | Keep as-is ✓ |
| `yang_baxter.py` | `from .braid_equations import ...` (already relative) | Keep as-is ✓ |
| `fermion_bounds.py` | `from qiskit.quantum_info import Pauli` | `try: from qiskit... except: from ._pauli_compat import Pauli` |

### Step 0.7 — Type Annotation Fixes (mypy --strict)

**tsr_constants.py** (~50 edits):
- Add `from typing import Final` + `import numpy.typing as npt`
- All 47 constants → `Final[float]` / `Final[int]` annotations
- Remove redundant string quotes from union type hints (4 functions)
- All `np.ndarray` in signatures → `npt.NDArray[np.floating[Any]]`

**braid_equations.py** (~15 edits):
- Replace `Optional` import → `Final`; add `npt`
- `self.braid_matrix: Optional[np.ndarray]` → `npt.NDArray[np.complex128] | None`
- `def get(...) -> Optional[BraidEquation]` → `BraidEquation | None`
- `VALID_SECTORS` → `Final[frozenset[str]]`
- **Replace all `assert` with `if/raise ValueError`** — `__init__` (3 asserts), `get_braid_generator_matrix` (1 assert), any others
- All bare `np.ndarray` → `npt.NDArray[np.complex128]`

**yang_baxter.py** (~3 edits):
- `assert b1.n_strands == b2.n_strands` → `if ... raise ValueError`
- Keep `bool()` wrappers (needed: numpy `np.bool_` is not `bool`, ruff/mypy strict requires explicit conversion for return type `-> bool`)
- Add `npt` types if any bare `np.ndarray`

**fermion_bounds.py** (~20 edits, most work):
- Conditional qiskit import (Step 0.6)
- Replace `Optional` → `X | None` throughout
- `FB_C_CONSTANT`, `FB_D_PRIME` → `Final[float]`
- **Replace ~15+ `assert` statements with `if/raise ValueError`** — this is the highest-volume edit in Phase 0
  - Lines: `create_fermion_bounds` (2 asserts), `occupy`/`vacate`/`is_occupied` (site range asserts), `check_pauli_exclusion`, `get_total_parity`, `measure_occupation`, `apply_hopping` (from/to range + distinct), plus others

### Step 0.8 — Ruff Lint Fixes

Run `ruff check src/ tests/ --fix`, then manually address:
- **I** (isort): import ordering stdlib → third-party → first-party
- **UP** (pyupgrade): any remaining old-style type hints
- **TCH**: typing-only imports behind `if TYPE_CHECKING:` (e.g., numpy.typing may need conditional guard)
- **ARG**: unused function arguments
- **SIM**: simplifiable if/else
- Run `ruff format src/ tests/` to auto-format

### Step 0.9 — Fix Test Imports

| Test file | Old import | New import |
|---|---|---|
| test_tsr_constants.py | `from algebra.src.tsr_constants import ...` | `from braidcodec.algebra.tsr_constants import ...` |
| test_braid_equations.py | `from algebra.src.braid_equations import ...` | `from braidcodec.algebra.braid_equations import ...` |
| test_braid_equations.py | `from algebra.src.tsr_constants import C_CONSTANT` | `from braidcodec.algebra.tsr_constants import C_CONSTANT` |
| test_yang_baxter.py | `from algebra.src.braid_equations import ...` | `from braidcodec.algebra.braid_equations import ...` |
| test_yang_baxter.py | `from algebra.src.yang_baxter import ...` | `from braidcodec.algebra.yang_baxter import ...` |
| test_fermion_bounds.py | `from qiskit.quantum_info import Pauli` | `from braidcodec.algebra.fermion_bounds import Pauli` (re-export from source) |
| test_fermion_bounds.py | `from algebra.src.fermion_bounds import ...` | `from braidcodec.algebra.fermion_bounds import ...` |

### Step 0.10 — CI/CD

**`.github/workflows/ci.yml`** — 4 jobs:
- `lint`: ruff check + ruff format --check
- `typecheck`: mypy src/braidcodec --strict
- `test`: matrix (ubuntu/macos/windows × py3.13), `pip install -e ".[dev]"`, pytest with coverage, codecov upload
- `build`: python -m build, upload artifact

**`.pre-commit-config.yaml`**: ruff (check --fix + format), trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files

### Step 0.11 — Validate & Push

1. `pip install -e ".[dev]"`
2. `ruff check src/ tests/` — 0 violations
3. `ruff format --check src/ tests/` — 0 diffs
4. `mypy src/braidcodec --strict` — 0 errors
5. `pytest tests/algebra/ -v` — 459 tests pass
6. `pytest --cov=braidcodec.algebra --cov-report=term-missing` — ≥98%
7. `python -m build` — dist/*.whl produced
8. `git add -A && git commit -m "feat: project scaffold + algebra migration with strict typing"`
9. Create repo on GitHub, push, verify CI green on all 3 OS

---

### Relevant files

- `pyproject.toml` — entire project config (new)
- `src/braidcodec/algebra/tsr_constants.py` — moved + ~50 typing edits (all `Final` annotations + npt types)
- `src/braidcodec/algebra/braid_equations.py` — moved + ~15 edits (asserts→raises, Optional→union, Final, npt)
- `src/braidcodec/algebra/yang_baxter.py` — moved + ~3 edits (assert→raise, npt)
- `src/braidcodec/algebra/fermion_bounds.py` — moved + ~20 edits (asserts→raises, Optional→union, Final, qiskit guard)
- `src/braidcodec/algebra/_pauli_compat.py` — vendored Pauli stub (~15 lines, new)
- `src/braidcodec/algebra/__init__.py` — re-exports 96 symbols (new)
- `src/braidcodec/_types.py` — shared types + tolerances (new)
- `src/braidcodec/_version.py` — version (new)
- `tests/algebra/test_*.py` — 4 files moved, imports patched
- `.github/workflows/ci.yml` — CI pipeline (new)
- `.pre-commit-config.yaml` — hooks (new)

### Verification

1. `ruff check src/ tests/` — 0 violations
2. `ruff format --check src/ tests/` — 0 diffs
3. `mypy src/braidcodec --strict` — 0 errors
4. `pytest tests/algebra/ -v` — 459 pass
5. `pytest --cov=braidcodec.algebra` — ≥98%
6. `python -m build` — works
7. CI green on GitHub (ubuntu + macos + windows)

### Decisions

- **Algebra moved, not copied** — old algebra dir deleted entirely. Single source of truth.
- **`assert` → `if/raise` everywhere** — tests may assert on exception types changing from `AssertionError` to `ValueError`. Check test expectations and update if needed.
- **Pauli stub is ~15 lines** — constructor stores `label: str`, nothing else. Full qiskit available via `[quantum]` extra for users who want it.
- **`from __future__ import annotations` kept** — harmless on py313, already in all modules, avoids unnecessary churn.
- **Dead deps dropped** — scipy, matplotlib, sympy, cupy, plotly from old requirements.txt are unused.