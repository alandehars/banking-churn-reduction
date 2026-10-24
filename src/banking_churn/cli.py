"""Command line entrypoints for the same V7 pipeline used in Colab and CI."""
import argparse
import json
from pathlib import Path
import uuid


def main(argv=None):
    parser=argparse.ArgumentParser(prog="banking-churn")
    sub=parser.add_subparsers(dest="command",required=True)
    train=sub.add_parser("train",help="Train on an explicitly supplied CSV")
    train.add_argument("--data",required=True,type=Path)
    train.add_argument("--output-base",default="bank_churn_artifacts_v7")
    train.add_argument("--allow-reviewed-source",action="store_true",help="Explicitly use a reviewed non-reference CSV")
    smoke=sub.add_parser("smoke",help="Run every stage on labelled synthetic data")
    smoke.add_argument("--output-base",default="bank_churn_artifacts_v7")
    smoke.add_argument("--rows",type=int,default=1000)
    smoke.add_argument("--seed",type=int,default=42)
    score=sub.add_parser("score",help="Score a CSV with a trusted local bundle")
    score.add_argument("--bundle",required=True,type=Path)
    score.add_argument("--input",required=True,type=Path)
    score.add_argument("--output",required=True,type=Path)
    report=sub.add_parser("report",help="Refresh aggregate HTML/PDF for an existing run")
    report.add_argument("--run-dir",required=True,type=Path)
    args=parser.parse_args(argv)
    if args.command in {"train","smoke"}:
        from .pipeline import run_pipeline
        from .config import TrainingConfig
        if args.command=="smoke":
            from .synthetic import write_synthetic_csv
            path=Path(args.output_base)/"inputs"/("synthetic-"+uuid.uuid4().hex[:8]+".csv")
            write_synthetic_csv(path,args.rows,args.seed)
            run=run_pipeline(path,args.output_base,TrainingConfig.quick(args.seed),"synthetic_smoke",False)
        else:
            run=run_pipeline(args.data,args.output_base,TrainingConfig(),
                "external_reviewed" if args.allow_reviewed_source else "reference_source",not args.allow_reviewed_source)
        print(json.dumps({"run_id":run.run_id,"data_context":run.metadata["data_context"],
            "synthetic":run.metadata["synthetic"],"checks_passed":len(run.checks),"run_dir":str(run.artifact_dir.resolve()),
            "model":run.metadata["model_name"]},indent=2))
    elif args.command=="score":
        import joblib
        from .inference import score_customer_csv
        bundle=joblib.load(args.bundle)
        result,path=score_customer_csv(args.input,bundle,args.output)
        print(json.dumps({"rows":len(result),"output":str(path),"policy":bundle["metadata"]["default_policy"]},indent=2))
    else:
        from .live import refresh_reports
        outputs=refresh_reports(args.run_dir)
        print(json.dumps({key:str(path) for key,path in outputs.items()},indent=2))


if __name__=="__main__":
    main()
