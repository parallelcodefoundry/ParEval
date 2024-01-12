""" Compute the metrics over the data.
"""
# std imports
import argparse
from math import comb
from typing import Union

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
    # group all the runs for this LLM output
    agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["count", "sum"]

    # mark as valid only if all runs are valid
    agg["is_valid"] = agg["count"] == agg["sum"]
    agg = agg.reset_index()
    agg = agg.drop(columns=["count", "sum"])

    return agg

def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)

def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the pass@k metric """
    agg = df.groupby(["name", "parallelism_model"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "valid_count"]
    agg = agg.reset_index()
    agg["pass@k"] = agg.apply(lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1)
    return agg.groupby(["parallelism_model"]).agg({"pass@k": "mean"})

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int) -> float:
    """ Compute the speedup@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    # sort the runtimes
    runtimes.sort()

    # compute expected value
    sum = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * runtimes[j-1]
        sum += num / den
    return pd.Series({f"speedup@{k}": sum})

def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the speedup@k metric """
    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[(df["parallelism_model"] == "serial") |
            (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 64)) |
            ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
            ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]
    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
        ).reset_index()

    # compute the mean speedup@k
    df = df.groupby(["parallelism_model"]).agg({f"speedup@{k}": "mean"})

    return df

def main():
    args = get_args()

    # read in input
    df = pd.read_csv(args.input_csv)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # filter/aggregate
    df["did_run"] = df["did_run"].fillna(False)     # if it didn't build, then this will be nan; overwrite
    df["is_valid"] = df["is_valid"].fillna(False)   # if it didn't build, then this will be nan; overwrite

    # compute metric
    if args.metric == "build":
        pass
    elif args.metric == "pass":
        df = get_correctness_df(df)
        result = passk(df, args.k)
        print(result)
    elif args.metric == "speedup":
        result = speedupk(df, args.k, args.n)
        print(result)


if __name__ == "__main__":
    main()