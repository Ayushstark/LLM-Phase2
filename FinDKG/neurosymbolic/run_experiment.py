"""Prepare a graph-only neural-symbolic FinDKG run without editing legacy code."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from graph_symbolic import write_audit, write_filtered_dataset


HERE = Path(__file__).resolve().parent
FINDKG_ROOT = HERE.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_snapshot() -> dict:
    result = {"python": sys.version, "platform": platform.platform(), "created_utc": datetime.now(timezone.utc).isoformat()}
    for package in ("torch", "dgl", "torch_scatter", "numpy", "pandas"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
            if package == "torch":
                result["cuda_available"] = module.cuda.is_available()
                result["cuda_version"] = module.version.cuda
        except Exception as exc:
            result[package] = {"available": False, "error": str(exc)}
    return result


def copy_baseline_graph(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for filename in ("train.txt", "valid.txt", "test.txt", "stat.txt", "entity2id.txt", "relation2id.txt", "time2id.txt"):
        src = source / filename
        if src.exists():
            shutil.copy2(src, target / filename)


def write_run_record(run_dir: Path, dataset_name: str, source: Path, variants: list[str], notes: list[str]) -> dict:
    config_path = HERE / "configs" / "experiment.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [source / f for f in ("train.txt", "valid.txt", "test.txt", "stat.txt", "entity2id.txt", "relation2id.txt")]
    record = {
        "run_id": run_dir.name,
        "status": "prepared_waiting_for_training_runtime",
        "dataset": dataset_name,
        "dataset_dir": str(source.resolve()),
        "variants": variants,
        "config": config,
        "input_sha256": {p.name: sha256(p) for p in files if p.exists()},
        "environment": environment_snapshot(),
        "notes": notes,
        "metrics": None,
        "checkpoints": [],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def parse_test_metrics(log_text: str) -> dict | None:
    pattern = re.compile(
        r"Test loss total=([0-9.eE+-]+).*?MRR=([0-9.eE+-]+),\s*Rec@1=([0-9.eE+-]+),"
        r"\s*Rec@3=([0-9.eE+-]+),\s*Rec@10=([0-9.eE+-]+),\s*Rec@100=([0-9.eE+-]+)"
    )
    matches = pattern.findall(log_text)
    if not matches:
        return None
    values = [float(x) for x in matches[-1]]
    return dict(zip(("test_loss", "mrr", "rec1", "rec3", "rec10", "rec100"), values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("FinDKG", "FinDKG-full"), default="FinDKG")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--train", action="store_true", help="Run legacy KGTransformer on the baseline and symbolic datasets")
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU training; GPU is strongly recommended")
    args = parser.parse_args()

    source = FINDKG_ROOT / "FinDKG_dataset" / args.dataset
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = HERE / "results" / run_id
    input_copy = run_dir / "data" / "FinDKG"
    symbolic_copy = run_dir / "data_symbolic" / "FinDKG"
    copy_baseline_graph(source, input_copy)
    baseline_audit = write_audit(source, run_dir / "baseline_audit")
    filtered_report = write_filtered_dataset(source, symbolic_copy)
    notes = [
        "The symbolic constraints use graph schema and temporal metadata only.",
        "The neural component is the unchanged KGTransformer architecture from train_DKG_run.py.",
        "A relation/entity-type signature rule is fitted on the earliest 80% of training time points and applied only to the final 20% of train.",
        "Valid recurrence events are retained; validation and test files are copied unchanged.",
        "This rule may reject valid novel signatures; rejection counts and coverage are reported, and the rule is a graph-derived hypothesis rather than source evidence.",
    ]
    manifest = write_run_record(run_dir, args.dataset, source, ["baseline", "symbolic_filter"], notes)
    manifest["audits"] = {
        "baseline_rejected_rows": sum(v["rejected"] for v in baseline_audit["splits"].values()),
        "symbolic_rejected_rows": sum(v["rejected"] for v in filtered_report["splits"].values()),
        "symbolic_rule_rejected_training_rows": filtered_report["export"]["filtered_by_symbolic_rule"],
        "symbolic_fit_signature_count": filtered_report["export"]["fit_signature_count"],
        "symbolic_filter_dataset": str(symbolic_copy.resolve()),
        "baseline_dataset": str(input_copy.resolve()),
    }

    if args.train:
        missing = []
        for package in ("torch", "dgl", "torch_scatter", "pandas", "numpy"):
            try:
                __import__(package)
            except Exception:
                missing.append(package)
        if missing:
            manifest["status"] = "blocked_missing_runtime_dependencies"
            manifest["missing_dependencies"] = missing
        else:
            import torch
            manifest["environment"]["cuda_available"] = torch.cuda.is_available()
            if not torch.cuda.is_available() and not args.allow_cpu:
                manifest["status"] = "blocked_gpu_unavailable"
                manifest["message"] = "Enable a Kaggle GPU accelerator or explicitly pass --allow-cpu."
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(json.dumps(manifest, indent=2))
                return
            # The legacy script is copied into the run area and edited only there.
            for variant, data_dir in (("baseline", input_copy), ("symbolic", symbolic_copy)):
                variant_dir = run_dir / variant
                variant_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(FINDKG_ROOT / "train_DKG_run.py", variant_dir / "train_DKG_run.py")
                # All result/checkpoint outputs for the run are redirected into the isolated run folder.
                script = (variant_dir / "train_DKG_run.py").read_text(encoding="utf-8")
                script = script.replace('data_root_path = \'./data\'', f"data_root_path = r'{data_dir.parent}'")
                script = script.replace("args.patience = 20", f"args.patience = {args.patience}")
                script = script.replace("epoch_times = 150", f"epoch_times = {args.max_epochs}")
                script = script.replace(
                    "import DKG",
                    "import DKG\n"
                    "from DKG import settings as _run_settings\n"
                    "from DKG.utils import log_utils as _run_log_utils\n"
                    "_run_settings.RESULT_ROOT = r'" + str((variant_dir / "result").resolve()) + "'\n"
                    "_run_log_utils.get_log_root_path = lambda graph_name, log_root_dir, result_root=None: os.path.join(_run_settings.RESULT_ROOT, graph_name, log_root_dir)"
                )
                (variant_dir / "train_DKG_run.py").write_text(script, encoding="utf-8")
                env = os.environ.copy()
                env["PYTHONPATH"] = str(FINDKG_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
                log_path = variant_dir / "console.log"
                with log_path.open("w", encoding="utf-8") as log_stream:
                    proc = subprocess.Popen([sys.executable, "train_DKG_run.py"], cwd=variant_dir, env=env,
                                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(f"[{variant}] {line}", end="", flush=True)
                        log_stream.write(line)
                        log_stream.flush()
                    returncode = proc.wait()
                log_text = log_path.read_text(encoding="utf-8")
                checkpoints = [str(p) for p in variant_dir.rglob("*.pt")]
                manifest.setdefault("training", {})[variant] = {
                    "returncode": returncode,
                    "console_log": str(log_path),
                    "test_metrics": parse_test_metrics(log_text),
                    "checkpoints": checkpoints,
                }
                if returncode:
                    manifest["status"] = "training_failed"
                    break
            else:
                manifest["status"] = "training_complete"
                manifest["metrics"] = {variant: details["test_metrics"] for variant, details in manifest["training"].items()}
                manifest["checkpoints"] = [p for details in manifest["training"].values() for p in details["checkpoints"]]
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
