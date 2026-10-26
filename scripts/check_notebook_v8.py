"""Check V8 source/module/lock parity without changing V7."""
import ast
import hashlib
from pathlib import Path
import re

import nbformat
from sync_notebook_v8 import MODULES

ROOT=Path(__file__).resolve().parents[1]
nb=nbformat.read(ROOT/"notebooks/banking_churn_v8.ipynb",as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=="code"]
assert len(code)==8
assert all(c.execution_count is None and not c.outputs for c in code)
for cell in code:ast.parse(cell.source)
embedded=ast.parse(next(c.source for c in code if c.id=="shared_package"))
sources=ast.literal_eval(embedded.body[0].value)
hashes=ast.literal_eval(embedded.body[1].value)
assert set(sources)==set(MODULES)
actual={name:(ROOT/"src/banking_churn"/name).read_text(encoding="utf-8") for name in MODULES}
assert sources==actual,"Rebuild V8: python scripts/sync_notebook_v8.py"
assert hashes=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in actual.items()}
dependencies=ast.parse(next(c.source for c in code if c.id=="dependencies"))
assert ast.literal_eval(dependencies.body[0].value)==(ROOT/"requirements-colab-lock.txt").read_text(encoding="utf-8")
raw=(ROOT/"notebooks/banking_churn_v8.ipynb").read_text(encoding="utf-8")
assert not re.search(r"[A-Z]:[\\/]Users[\\/]|[A-Z]:[\\/]File Alan|@gmail\.com|github_pat_|ghp_[A-Za-z0-9]{20,}",raw)
assert "widgets" not in nb.metadata
assert not re.search(r"\.fit\s*\(","\n".join(c.source for c in code)),"V8 audits a frozen model"
print(f"V8 valid: {len(code)} code cells, {len(actual)} matching modules, locked requirements, no training or saved outputs.")
