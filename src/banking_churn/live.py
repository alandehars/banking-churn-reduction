"""Keep V7 report exports and the current run archive in sync after form changes."""
import json
from pathlib import Path
import shutil

from .decision_tools import v6_save_section
from .reporting import generate_executive_reports


def refresh_reports(directory):
    directory=Path(directory)
    outputs=generate_executive_reports(directory)
    meta=json.loads((directory/"model_metadata.json").read_text(encoding="utf-8"))
    base=directory.parent.parent
    latest=base/"latest_run.json"
    if latest.is_file():
        manifest=json.loads(latest.read_text(encoding="utf-8"))
        if manifest.get("run_id")==meta["run_id"] and manifest.get("status")=="complete":
            shutil.make_archive(str(base/meta["run_id"]),"zip",root_dir=directory)
    return outputs


def save_live_section(kind,inputs,summary,directory,context,tables=None):
    path,report=v6_save_section(kind,inputs,summary,directory,context,tables)
    refresh_reports(directory)
    return path,report
