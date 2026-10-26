"""Execute V8 on synthetic data, optionally using a trusted frozen V7 bundle without retraining."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
mode=parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--smoke",action="store_true")
mode.add_argument("--bundle",type=Path,help="Load only a trusted local V7 bundle")
parser.add_argument("--reference",type=Path)
parser.add_argument("--batch",type=Path)
parser.add_argument("--output",type=Path,default=Path("outputs/banking_churn_v8.executed.ipynb"))
parser.add_argument("--output-base",type=Path,default=Path("bank_churn_artifacts_v8"))
args=parser.parse_args()
source=ROOT/"notebooks/banking_churn_v8.ipynb"
target=(ROOT/args.output).resolve()
if target.exists():raise FileExistsError("Choose a new output filename; saved notebooks are never overwritten: "+str(target))
for path in [args.bundle,args.reference,args.batch]:
    if path is not None and not path.is_file():raise FileNotFoundError(path)
if args.smoke and (args.reference or args.batch):parser.error("Synthetic smoke does not accept private reference/batch files")
nb=nbformat.read(source,as_version=4)
configuration=next(c for c in nb.cells if c.id=="configuration")
if args.bundle:
    configuration.source=configuration.source.replace('Mode = "demo_sintetis"','Mode = "bundle_tepercaya"')
    for name,old,path in [("Path_Bundle","bank_churn_bundle.joblib",args.bundle),
        ("Path_Referensi","Churn_Modelling.csv",args.reference),("Path_Batch_Baru","",args.batch)]:
        value=str(path.resolve()) if path else ""
        configuration.source=configuration.source.replace(name+" = "+repr(old).replace("'",'"'),name+" = "+repr(value))
configuration.source=configuration.source.replace('OUTPUT_BASE = Path("bank_churn_artifacts_v8")',
    'OUTPUT_BASE = Path('+repr(str((ROOT/args.output_base).resolve()))+')')
kernel_root=ROOT/".jupyter";kernel_dir=kernel_root/"kernels/bank-churn-v8"
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
    "display_name":"Banking churn V8 locked Python","language":"python"}),encoding="utf-8")
os.environ["JUPYTER_PATH"]=str(kernel_root)
os.environ["BANK_CHURN_SKIP_INSTALL"]="1"
os.environ["MPLCONFIGDIR"]=str(ROOT/".tmp/matplotlib")
os.environ["IPYTHONDIR"]=str(ROOT/".tmp/ipython")
for path in [target.parent,ROOT/".tmp/matplotlib",ROOT/".tmp/ipython"]:path.mkdir(parents=True,exist_ok=True)
client=NotebookClient(nb,kernel_name="bank-churn-v8",timeout=300,allow_errors=False,
    resources={"metadata":{"path":str(ROOT)}},store_widget_state=False)
try:client.execute()
finally:nbformat.write(nb,target)
code=[c for c in nb.cells if c.cell_type=="code"]
errors=[o for c in code for o in c.outputs if o.output_type=="error"]
assert not errors and all(c.execution_count for c in code)
output_text="\n".join(o.get("text","") for c in code for o in c.outputs if o.output_type=="stream")
assert '"status": "complete"' in output_text and '"model_refit": false' in output_text
print(json.dumps({"output":str(target),"code_cells_executed":len(code),"errors":len(errors),
    "execution_environment":"local_kernel","model_refit":False,"private_bundle_loaded":bool(args.bundle)},indent=2))
