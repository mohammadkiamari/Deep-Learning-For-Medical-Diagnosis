"""check_submission.py

DLMD exercise 2, Deep Learning for Medical Diagnostics, RWTH Aachen University, Winter Semester 2026/27.
"""

import argparse
import importlib
import subprocess
import sys
import zipfile
from pathlib import Path

REQUIRED = ["report.pdf", "run_analysis.py", "dlmd_e2/__init__.py", "dlmd_e2/cam.py",
            "dlmd_e2/data.py", "dlmd_e2/encoder.py", "dlmd_e2/metrics.py", "dlmd_e2/model.py",
            "dlmd_e2/report.py", "dlmd_e2/resample.py"]
MODULES = ["dlmd_e2.cam", "dlmd_e2.data", "dlmd_e2.encoder", "dlmd_e2.metrics", "dlmd_e2.model",
           "dlmd_e2.report", "dlmd_e2.resample"]
ZIP_NAME = "DLMD_E2_submission.zip"
SKIP_DIRS = {"data", "__pycache__", ".ipynb_checkpoints", ".git", ".venv", "venv"}


def files_to_pack(root):
    """Every file under root except the data folder, caches, hidden files, and zip files."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or path.suffix == ".zip":
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        yield path, rel


def main():
    """Check that the submission is complete and runs, then pack it into one zip file.

    The check says nothing about whether your results are correct.
    """
    ap = argparse.ArgumentParser(description="Check and pack an E2 submission.")
    ap.add_argument("--root", default=".", type=Path, help="the code folder")
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-run", action="store_true",
                    help="skip running run_analysis.py")
    args = ap.parse_args()
    root = args.root.resolve()

    problems = []
    for rel in REQUIRED:
        if not (root / rel).exists():
            problems.append("missing file: {}".format(rel))

    sys.path.insert(0, str(root))
    for m in MODULES:
        try:
            importlib.import_module(m)
        except Exception as exc:   # report the error and keep checking
            problems.append("cannot import {}: {}".format(m, exc))

    if not args.no_run:
        print("running run_analysis.py ...")
        proc = subprocess.run([sys.executable, "run_analysis.py", "--data", args.data],
                              cwd=str(root), capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join(proc.stdout.splitlines()[-5:] + proc.stderr.splitlines()[-5:])
            problems.append("run_analysis.py stopped with exit code {}:\n{}".format(
                proc.returncode, tail))
        elif not (root / "results.csv").exists():
            problems.append("run_analysis.py did not write results.csv")
        if proc.returncode == 0 and not (root / "model.npz").exists():
            problems.append("run_analysis.py did not save your final model to model.npz")

    if problems:
        print("Not ready. {} problem(s):".format(len(problems)))
        for p in problems:
            print("  - {}".format(p))
        sys.exit(1)

    out = root / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path, rel in files_to_pack(root):
            z.write(path, Path("DLMD_E2_submission") / rel)
    size_mb = out.stat().st_size / 1e6
    print("Complete. Upload {} ({:.1f} MB) to Moodle.".format(out.name, size_mb))
    if size_mb > 20:
        print("The zip is larger than 20 MB. Remove large files that are not code or "
              "the report, then run this check again.")


if __name__ == "__main__":
    main()
