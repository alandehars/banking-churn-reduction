"""Execute a local notebook using the current Python environment, saving genuine outputs separately."""
import argparse
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",default="outputs/banking_churn_v6.executed.ipynb")
parser.add_argument("--data-file",type=Path,help="Optional private source CSV kept outside the repository")
args=parser.parse_args()
target=(ROOT/args.output).resolve()
if target==(ROOT/"notebooks/banking_churn_v6.ipynb").resolve():
    raise ValueError("Choose a separate output file to preserve the clean public notebook.")
target.parent.mkdir(parents=True,exist_ok=True)
nb=nbformat.read(ROOT/"notebooks/banking_churn_v6.ipynb",as_version=4)
if args.data_file:
    if not args.data_file.is_file():
        raise ValueError("The requested source CSV does not exist.")
    configuration=next(c for c in nb.cells if c.id=="configuration")
    configuration.source=configuration.source.replace('DATA_FILE = Path("churn.csv")',
        'DATA_FILE = Path('+repr(str(args.data_file.resolve()))+')')
kernel_root=ROOT/".jupyter"
kernel_dir=kernel_root/"kernels/bank-churn-public"
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
    "display_name":"Banking churn project Python","language":"python"}),encoding="utf-8")
os.environ["JUPYTER_PATH"]=str(kernel_root)
os.environ["MPLCONFIGDIR"]=str(ROOT/".tmp/matplotlib")
os.environ["IPYTHONDIR"]=str(ROOT/".tmp/ipython")
for sub in [".tmp/matplotlib",".tmp/ipython"]:
    (ROOT/sub).mkdir(parents=True,exist_ok=True)
client=NotebookClient(nb,kernel_name="bank-churn-public",timeout=900,allow_errors=False,
    resources={"metadata":{"path":str(ROOT)}},store_widget_state=False)
try:
    client.execute()
finally:
    nbformat.write(nb,target)
errors=sum(o.output_type=="error" for c in nb.cells for o in c.get("outputs",[]))
print(f"Saved {target.name}: {len([c for c in nb.cells if c.cell_type=='code'])} code cells; {errors} errors.")
