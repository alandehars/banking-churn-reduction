"""Verify the public V7 notebook matches the tested package and dependency lock."""
import ast
import hashlib
from pathlib import Path
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]
nb=nbformat.read(ROOT/"notebooks/banking_churn_v7.ipynb",as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=="code"]
assert len(code)==14
assert all(c.execution_count is None and not c.outputs for c in code)
for cell in code:ast.parse(cell.source)
assert len([c for c in code if c.id.startswith("form_") and c.id!="form_state"])==7
embedded=ast.parse(next(c.source for c in code if c.id=="shared_package"))
sources=ast.literal_eval(embedded.body[0].value)
hashes=ast.literal_eval(embedded.body[1].value)
actual={p.name:p.read_text(encoding="utf-8") for p in (ROOT/"src/banking_churn").glob("*.py")}
assert sources==actual,"Rebuild notebook after module edits: python scripts/sync_notebook_v7.py"
assert hashes=={k:hashlib.sha256(v.encode()).hexdigest() for k,v in actual.items()}
dependencies=ast.parse(next(c.source for c in code if c.id=="dependencies"))
assert ast.literal_eval(dependencies.body[0].value)==(ROOT/"requirements-lock.txt").read_text(encoding="utf-8")
raw=(ROOT/"notebooks/banking_churn_v7.ipynb").read_text(encoding="utf-8")
assert not re.search(r"[A-Z]:[\\/]Users[\\/]|[A-Z]:[\\/]File Alan|@gmail\.com|github_pat_|ghp_[A-Za-z0-9]{20,}",raw)
assert "widgets" not in nb.metadata
print(f"V7 valid: {len(code)} code cells, seven simulation forms, {len(actual)} matching modules and locked requirements; clean outputs.")
