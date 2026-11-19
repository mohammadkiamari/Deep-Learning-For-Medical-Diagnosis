"""run_analysis.py

DLMD exercise 2, Deep Learning for Medical Diagnostics, RWTH Aachen University, Winter Semester 2026/27.
"""

import argparse
from pathlib import Path

from dlmd_e2.data import ExerciseData
from dlmd_e2.encoder import Encoder
from dlmd_e2.report import ResultsTable


def main():
    """Your analysis.

    Running `python run_analysis.py` has to reproduce every number in your report from the files in data/, with no manual step. It writes the numbers to results.csv and saves your final model to model.npz with LinearHead.save().
    """
    ap = argparse.ArgumentParser(description="Analysis of the reference model.")
    ap.add_argument("--data", default="data", help="folder with index.csv, features.npz, and images/")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--model", default="model.npz")
    args = ap.parse_args()

    data = ExerciseData(args.data)
    encoder = Encoder(Path(args.data) / "encoder" / "resnet18.onnx")
    results = ResultsTable()

    raise NotImplementedError("Write your analysis here.")

    print(results)
    results.write(args.out)


if __name__ == "__main__":
    main()
