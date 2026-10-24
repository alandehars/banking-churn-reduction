"""Execute all V7 cells with a real kernel; synthetic smoke or explicit reference CSV."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke",action="store_true",help="Use labelled synthetic data, small nested CV and all seven forms")
parser.add_argument("--data-file",type=Path)
parser.add_argument("--output",default="outputs/banking_churn_v7.executed.ipynb")
parser.add_argument("--output-base",default="bank_churn_artifacts_v7")
args=parser.parse_args()
if not args.smoke and not args.data_file:
    parser.error("Supply --smoke or --data-file; source download is an explicit notebook action")
source=ROOT/"notebooks/banking_churn_v7.ipynb"
target=(ROOT/args.output).resolve()
if target==source.resolve():raise ValueError("Preserve the clean source notebook; choose a separate output")
nb=nbformat.read(source,as_version=4)
configuration=next(c for c in nb.cells if c.id=="configuration")
if args.smoke:configuration.source=configuration.source.replace('Mode_data = "reference_source"','Mode_data = "synthetic_smoke"')
elif args.data_file:
    if not args.data_file.is_file():raise FileNotFoundError(args.data_file)
    configuration.source=configuration.source.replace('Path_CSV = "churn.csv"','Path_CSV = '+repr(str(args.data_file.resolve())))
configuration.source=configuration.source.replace('OUTPUT_BASE = Path("bank_churn_artifacts_v7")',
    'OUTPUT_BASE = Path('+repr(str((ROOT/args.output_base).resolve()))+')')
kernel_root=ROOT/".jupyter";kernel_dir=kernel_root/"kernels/bank-churn-v7"
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
    "display_name":"Banking churn V7 locked Python","language":"python"}),encoding="utf-8")
os.environ["JUPYTER_PATH"]=str(kernel_root)
os.environ["BANK_CHURN_SKIP_INSTALL"]="1"
os.environ["MPLCONFIGDIR"]=str(ROOT/".tmp/matplotlib")
os.environ["IPYTHONDIR"]=str(ROOT/".tmp/ipython")
for path in [target.parent,ROOT/".tmp/matplotlib",ROOT/".tmp/ipython"]:path.mkdir(parents=True,exist_ok=True)
client=NotebookClient(nb,kernel_name="bank-churn-v7",timeout=900,allow_errors=False,
    resources={"metadata":{"path":str(ROOT)}},store_widget_state=False)
try:client.execute()
finally:nbformat.write(nb,target)
code=[c for c in nb.cells if c.cell_type=="code"]
errors=[o for c in code for o in c.outputs if o.output_type=="error"]
assert not errors and all(c.execution_count for c in code)
manifest=json.loads(((ROOT/args.output_base)/"latest_run.json").read_text(encoding="utf-8"))
assert manifest["synthetic"]==args.smoke
assert manifest["status"]=="complete" and manifest["checks_passed"]==13
print(json.dumps({"output":str(target),"code_cells_executed":len(code),"errors":len(errors),
    "run_id":manifest["run_id"],"synthetic":manifest["synthetic"],"checks_passed":manifest["checks_passed"]},indent=2))
