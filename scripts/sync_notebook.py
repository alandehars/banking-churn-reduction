"""Update embedded decision helpers from src, then clear execution metadata and outputs."""
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"notebooks/banking_churn_v6.ipynb"
nb=nbformat.read(path,as_version=4)
module=(ROOT/"src/banking_churn/decision_tools.py").read_text(encoding="utf-8")
for cell in nb.cells:
    if cell.id=="v6_helpers":
        cell.source="#@title Fungsi V6 — buka untuk code review\nV6_TOOLS_SOURCE = "+repr(module)+"\n"+module
    if cell.cell_type=="code":
        cell.outputs=[]
        cell.execution_count=None
    for key in ["executionInfo","execution","outputId","colab"]:
        cell.metadata.pop(key,None)
nb.metadata.pop("widgets",None)
nbformat.validate(nb)
nbformat.write(nb,path)
print("Embedded helpers updated; notebook outputs cleared.")
