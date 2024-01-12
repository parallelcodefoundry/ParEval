""" Compute the metrics over the data.
"""
# std imports
import argparse

# tpl imports
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
    parser.add_argument("-m", "--metric", default="pass", choices=["build", "pass", "speedup"], help="Metric to compute.")
    parser.add_argument("-k", "--k", type=int, default=1, help="K value for pass@k, build@k, and speedup@k.")
    parser.add_argument("-n", "--n", type=int, default=1, help="N value for speedup@k.")
    parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
    return parser.parse_args()

def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Group by name, parallelism_model, and output_idx, and set is_valid to true only if all rows in the group have is_valid = true.
        Set it to false otherwise.
    """
    agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["count", "sum"]
    agg["is_valid"] = agg["count"] == agg["sum"]
    agg = agg.reset_index()
    agg = agg.drop(columns=["count", "sum"])
    return agg

def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the pass@k metric """
    agg = df.groupby(["name", "parallelism_model"]).agg({"is_valid": "sum"})
    agg = agg.reset_index()
    agg["pass@k"] = agg.apply(lambda x: _passk(20, x["is_valid"], k), axis=1)
    return agg.groupby(["parallelism_model"]).agg({"pass@k": "mean"})


def main():
    args = get_args()

    # read in input
    df = pd.read_csv(args.input_csv)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # filter/aggregate
    df["did_run"] = df["did_run"].fillna(False)
    df["is_valid"] = df["is_valid"].fillna(False)

    # compute metric
    if args.metric == "build":
        pass
    elif args.metric == "pass":
        df = get_correctness_df(df)
        result = passk(df, args.k)
        print(result)
    elif args.metric == "speedup":
        pass


if __name__ == "__main__":
    main()