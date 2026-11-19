"""run_baseline.py

DLMD exercise 2, Deep Learning for Medical Diagnostics, RWTH Aachen University, Winter Semester 2026/27.
"""

import argparse

from dlmd_e2.data import ExerciseData
from dlmd_e2.metrics import auroc
from dlmd_e2.model import LinearHead
from dlmd_e2.resample import Bootstrap, format_ci


def main():
    ap = argparse.ArgumentParser(
        description="Train the reference model at hospital A and evaluate it at both hospitals.")
    ap.add_argument("--data", default="data", help="folder with index.csv, features.npz, and images/")
    args = ap.parse_args()

    data = ExerciseData(args.data)
    print(data.describe())
    print()

    train, test, external = data.split == "train", data.split == "test", data.split == "external"
    head = LinearHead().fit(data.features[train], data.y[train])
    print("The reference model, trained on the training images of hospital A")
    for name, rows in (("hospital A, test", test), ("hospital B", external)):
        y, p = data.y[rows], head.predict_proba(data.features[rows])
        stats = Bootstrap().run(lambda i: auroc(y[i], p[i]), len(y))
        print("AUROC, {:<18} {:.3f}   bootstrap {}".format(name, auroc(y, p), format_ci(*stats)))


if __name__ == "__main__":
    main()
