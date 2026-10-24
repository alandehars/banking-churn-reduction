"""Check public notebook structure, clean outputs and source-module parity without executing training."""
import ast
import json
from pathlib import Path
import re

import nbformat

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"notebooks/banking_churn_v6.ipynb"
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=="code"]
assert len(code)==38
assert all(c.execution_count is None and not c.outputs for c in code)
assert "widgets" not in nb.metadata
for c in code:
    ast.parse(c.source)
    assert not (set(c.metadata)&{"executionInfo","execution","outputId"})
helpers=next(c.source for c in code if c.id=="v6_helpers")
tree=ast.parse(helpers)
module=(ROOT/"src/banking_churn/decision_tools.py").read_text(encoding="utf-8")
assert ast.literal_eval(tree.body[0].value)==module
assert ast.dump(ast.Module(body=tree.body[1:],type_ignores=[]))==ast.dump(ast.parse(module))
raw=path.read_text(encoding="utf-8")
assert not re.search(r"[A-Z]:[\\/]Users[\\/]|[A-Z]:[\\/]File Alan|@gmail\.com|github_pat_|ghp_[A-Za-z0-9]{20,}",raw)
print("Notebook valid: 38 code cells; outputs/identity metadata removed; module and embedded source match.")
