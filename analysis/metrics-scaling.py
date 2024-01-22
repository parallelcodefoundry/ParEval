""" Compute the metrics over the data for various resource counts.
"""
# std imports
import argparse
import json
from math import comb
from typing import Union

# tpl imports
import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=str, help="Input CSV file containing the test cases.")
    parser.add_argument("-k", "--k", type=int, default=1, help="K value for speedup@k and efficiency@k")
    parser.add_argument("-n", "--n", type=int, nargs='+', default=[1,2,4,8,16,32,64,128,256,512], help="Number of resources for speedup@k and efficiency@k")
    parser.add_argument("--execution-model", choices=['mpi', 'mpi+omp', 'omp', 'kokkos'], default='mpi', help="Execution model to use for speedup@k and efficiency@k")
    parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
    parser.add_argument("--problem-sizes", type=str, default='../drivers/problem-sizes.json', help="Json with problem sizes. Used for calculating GPU efficiency.")
    parser.add_argument("--model-name", type=str, help="Add model name column with this value")
    return parser.parse_args()

def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n: int) -> float:
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
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8)
        sum += num / den
    return pd.Series({f"speedup_{n}@{k}": sum})

def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the speedup@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[df["n"] == n]
    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, n)
        ).reset_index()

    # compute the mean speedup@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_{n}@{k}": "mean"})

    return df

def _efficiencyk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n_resources: Union[pd.Series, np.ndarray]) -> float:
    """ Compute the efficiency@k metric """
    # create a copy of the runtimes
    if isinstance(runtimes, pd.Series):
        runtimes = runtimes.values.copy()
    else:
        runtimes = runtimes.copy()

    if isinstance(n_resources, pd.Series):
        n_resources = n_resources.values.copy()
    else:
        n_resources = n_resources.copy()

    # sort the runtimes
    runtimes.sort()

    # make sure n_resources is all the same value and get that value
    assert np.all(n_resources == n_resources[0])
    n = int(n_resources[0])

    # compute expected value
    sum = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8) * n_resources[j-1]
        sum += num / den
    return pd.Series({f"efficiency_{n}@{k}": sum})

def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the efficiency@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[df["n"] == n]
    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n"])
        ).reset_index()
    
    # compute the mean efficiency@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_{n}@{k}": "mean"})

    return df

def parse_problem_size(problem_size: str) -> int:
    """ problem size is of format '(1<<n)' """
    num = problem_size.split("<<")[1][:-1]
    return 2 ** int(num)

def main():
    args = get_args()

    # read in input
    df = pd.read_csv(args.input_csv)

    # read in problem sizes
    with open(args.problem_sizes, "r") as f:
        problem_sizes = json.load(f)
        for problem in problem_sizes:
            for parallelism_model, problem_size in problem_sizes[problem].items():
                df.loc[(df["name"] == problem) & (df["parallelism_model"] == parallelism_model), "problem_size"] = parse_problem_size(problem_size)

    # remove rows where parallelism_model is kokkos and num_threads is 64
    #df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # filter/aggregate
    df["did_run"] = df["did_run"].fillna(False)     # if it didn't build, then this will be nan; overwrite
    df["is_valid"] = df["is_valid"].fillna(False)   # if it didn't build, then this will be nan; overwrite

    if args.execution_model == "mpi":
        df = df[df["parallelism_model"] == "mpi"]
        df["n"] = df["num_procs"]
    elif args.execution_model == "mpi+omp":
        df = df[df["parallelism_model"] == "mpi+omp"]
        df["n"] = df["num_procs"] * df["num_threads"]
    elif args.execution_model == "omp":
        df = df[df["parallelism_model"] == "omp"]
        df["n"] = df["num_threads"]
    elif args.execution_model == "kokkos":
        df = df[df["parallelism_model"] == "kokkos"]
        df["n"] = df["num_threads"]
    else:
        raise NotImplementedError(f"Unsupported execution model {args.execution_model}")

    # get values for each k
    all_results = []
    for n in args.n:
        speedup_values = speedupk(df, args.k, n)
        efficiency_values = efficiencyk(df, args.k, n)
        all_results.extend([speedup_values, efficiency_values])
    
    # merge all_results; each df has one column and the same index
    # build a new df with all the columns and the same index
    merged_df = pd.concat(all_results, axis=1).reset_index()

    # if there were no successfull builds or runs, then speedup@k will be nan after merging
    # replace NaN speedup@k values with 0.0
    for n in args.n:
        merged_df[f"speedup_{n}@{args.k}"] = merged_df[f"speedup_{n}@{args.k}"].fillna(0.0)
        merged_df[f"efficiency_{n}@{args.k}"] = merged_df[f"efficiency_{n}@{args.k}"].fillna(0.0)

    # add model name column
    if args.model_name:
        merged_df.insert(0, "model_name", args.model_name)

    # clean up column names
    column_name_map = {
        "model_name": "model",
        "parallelism_model": "execution model",
        "problem_type": "problem type",
    }
    merged_df = merged_df.rename(columns=column_name_map)

    # write to csv
    if args.output:
        merged_df.to_csv(args.output, index=False)
    else:
        pd.set_option('display.max_columns', merged_df.shape[1]+1)
        pd.set_option('display.max_rows', merged_df.shape[0]+1)
        print(merged_df)
        


if __name__ == "__main__":
    main()