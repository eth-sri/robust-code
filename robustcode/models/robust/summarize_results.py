import glob
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def parse_args():
    parser = ArgumentParser("Summarize Results")
    parser.add_argument(
        "--results_dir", type=str, default="results", help="directory with the results"
    )
    args = parser.parse_args()
    return args


def summarize_results(csv_path):
    df = pd.read_csv(csv_path)
    if "test_acc" not in df:
        return None
    data = {
        "accuracy (median)": np.median(df["test_acc"]),
        "accuracy (std)": np.std(df["test_acc"]),
        "robustness (median)": np.median(df["T:SOUND"]),
        "robustness (std)": np.std(df["T:SOUND"]),
        "N": len(df),
        "name": "/".join(os.path.dirname(csv_path).split("/")[1:]),
    }
    return data


def main():
    args = parse_args()
    results = []
    for file in glob.glob("{}/**/*.csv".format(args.results_dir), recursive=True):
        data = summarize_results(file)
        if data is not None:
            results.append(data)
    df = pd.DataFrame(results)
    print(df)


if __name__ == "__main__":
    main()
