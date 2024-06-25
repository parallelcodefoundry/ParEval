""" Compute the metrics over the data.
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
    parser.add_argument("-k", "--k", type=int, nargs='+', default=[1,5,10,20], help="K value for pass@k, build@k, and speedup@k.")
    parser.add_argument("-n", "--n", type=int, default=1, help="N value for speedup@k.")
    parser.add_argument("-o", "--output", type=str, help="Output csv file containing the results.")
    parser.add_argument("--problem-sizes", type=str, default='../drivers/problem-sizes.json', help="Json with problem sizes. Used for calculating GPU efficiency.")
    parser.add_argument("--model-name", type=str, help="Add model name column with this value")
    return parser.parse_args()

def get_correctness_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Group by name, parallelism_model, and output_idx, and set is_valid to true only if all rows in the group have is_valid = true.
        Set it to false otherwise.
    """
    # group all the runs for this LLM output
    df = df.copy()
    agg = df.groupby(["name", "parallelism_model", "output_idx"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["count", "sum"]

    # mark as valid only if all runs are valid
    agg["is_valid"] = agg["count"] == agg["sum"]
    agg = agg.reset_index()
    agg = agg.drop(columns=["count", "sum"])
    
    # add problem_type column from df
    agg = agg.merge(df[["name", "problem_type"]].drop_duplicates(), on="name", how="left")

    return agg

def nCr(n: int, r: int) -> int:
    if n < r:
        return 1
    return comb(n, r)

def buildk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the build@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"did_build": ["count", "sum"]})
    agg.columns = ["total_build_attempts", "successful_builds"]
    agg = agg.reset_index()
    agg[f"build@{k}"] = agg.apply(lambda x: _passk(x["total_build_attempts"], x["successful_builds"], k), axis=1)
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"build@{k}": "mean"})

def _passk(num_samples: int, num_correct: int, k: int) -> float:
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def passk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the pass@k metric """
    agg = df.groupby(["name", "parallelism_model", "problem_type"]).agg({"is_valid": ["count", "sum"]})
    agg.columns = ["total_runs", "valid_count"]
    agg = agg.reset_index()
    agg[f"pass@{k}"] = agg.apply(lambda x: _passk(x["total_runs"], x["valid_count"], k), axis=1)
    return agg.groupby(["parallelism_model", "problem_type"]).agg({f"pass@{k}": "mean"})

def _speedupk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, col_name: str = 'speedup@{}') -> float:
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
    return pd.Series({col_name.format(k): sum})

def speedupk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the speedup@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[(df["parallelism_model"] == "serial") |
            (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
            ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]
    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k)
        ).reset_index()

    # compute the mean speedup@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup@{k}": "mean"})

    return df



def speedupk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the speedup_max@k. Same as speedup_n@k, but instead of a fixed n
        we use the n that gives the max speedup
    """
    df = df.copy()
    df.drop(columns=['prompt'], inplace=True)

    # get all the runs where the submission is valid
    df = df[df["is_valid"] == True]

    # choose the min across processor counts
    df["runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["runtime"].transform("min")

    # use the min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # select only run_idx 0
    df["run_idx"] = df["run_idx"].astype(int)
    df = df[df["run_idx"] == 0]

    # group by name, parallelism_model, and output_idx and call _speedupk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _speedupk(row["runtime"], np.min(row["best_sequential_runtime"]), k, col_name="speedup_max@{}")
        ).reset_index()

    # compute the mean speedup_max@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"speedup_max@{k}": "mean"})

    return df



def _efficiencyk(runtimes: Union[pd.Series, np.ndarray], baseline_runtime: float, k: int, n_resources: Union[pd.Series, np.ndarray], col_name: str = 'efficiency@{}') -> float:
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

    # compute expected value
    sum = 0.0
    num_samples = runtimes.shape[0]
    for j in range(1, num_samples+1):
        num = nCr(j-1, k-1) * baseline_runtime
        den = nCr(num_samples, k) * max(runtimes[j-1], 1e-8) * n_resources[j-1]
        sum += num / den
    return pd.Series({col_name.format(k): sum})

def efficiencyk(df: pd.DataFrame, k: int, n: int) -> pd.DataFrame:
    """ Compute the efficiency@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # choose processor count; hardcoded right now
    df = df[(df["parallelism_model"] == "serial") |
           (df["parallelism_model"] == "cuda") |
            (df["parallelism_model"] == "hip") |
            ((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "omp") & (df["num_threads"] == 32)) |
            ((df["parallelism_model"] == "mpi") & (df["num_procs"] == 512)) |
            ((df["parallelism_model"] == "mpi+omp") & (df["num_procs"] == 4) & (df["num_threads"] == 64))]

    # set n_resources column to 1 for serial; 32 for kokkos; 32 for omp; 512 for mpi; 4*64 for mpi+omp;
    # set it to problem_size for cuda and hip
    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = 32
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = 8
    df.loc[df["parallelism_model"] == "mpi", "n_resources"] = 512
    df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = 4*64

    df = df.copy()

    # use min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"])
        ).reset_index()
    
    # compute the mean efficiency@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency@{k}": "mean"})

    return df


def efficiencyk_max(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """ Compute the efficiency_max@k metric """
    df = df.copy()

    # get all runs where is_valid is true
    df = df[df["is_valid"] == True]

    # set n_resources column
    df["n_resources"] = 1
    df.loc[df["parallelism_model"] == "cuda", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "hip", "n_resources"] = df["problem_size"]
    df.loc[df["parallelism_model"] == "kokkos", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "omp", "n_resources"] = df["num_threads"]
    df.loc[df["parallelism_model"] == "mpi", "n_resources"] = df["num_procs"]
    df.loc[df["parallelism_model"] == "mpi+omp", "n_resources"] = df["num_procs"] * df["num_threads"]

    # choose the row with min num_resources * runtime
    df = df.groupby(["name", "parallelism_model", "output_idx"]).apply(
            lambda row: row.iloc[np.argmin(row["runtime"] * row["n_resources"])]
        ).reset_index(drop=True)

    # use the min best_sequential_runtime
    df["best_sequential_runtime"] = df.groupby(["name", "parallelism_model", "output_idx"])["best_sequential_runtime"].transform("min")

    # group by name, parallelism_model, and output_idx and call _efficiencyk
    df = df.groupby(["name", "parallelism_model", "problem_type"]).apply(
            lambda row: _efficiencyk(row["runtime"], np.min(row["best_sequential_runtime"]), k, row["n_resources"], col_name='efficiency_max@{}')
        ).reset_index()

    # compute the mean efficiency_max@k
    df = df.groupby(["parallelism_model", "problem_type"]).agg({f"efficiency_max@{k}": "mean"})

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
    df = df[~((df["parallelism_model"] == "kokkos") & (df["num_threads"] == 64))]

    # filter/aggregate
    df["did_run"] = df["did_run"].fillna(False)     # if it didn't build, then this will be nan; overwrite
    df["is_valid"] = df["is_valid"].fillna(False)   # if it didn't build, then this will be nan; overwrite

    # get only valid runs
    valid_runs = get_correctness_df(df)
    
    # get values for each k
    all_results = []
    for k in args.k:
        build_values = buildk(df, k)
        pass_values = passk(valid_runs, k)
        speedup_values = speedupk(df, k, args.n)
        speedup_max_values = speedupk_max(df, k)
        efficiency_values = efficiencyk(df, k, args.n)
        efficiency_max_values = efficiencyk_max(df, k)
        all_results.extend([build_values, pass_values, speedup_values, speedup_max_values, efficiency_values, efficiency_max_values])
    
    # merge all_results; each df has one column and the same index
    # build a new df with all the columns and the same index
    merged_df = pd.concat(all_results, axis=1).reset_index()

    # if there were no successfull builds or runs, then speedup@k will be nan after merging
    # replace NaN speedup@k values with 0.0
    for k in args.k:
        merged_df[f"speedup@{k}"] = merged_df[f"speedup@{k}"].fillna(0.0)
        merged_df[f"speedup_max@{k}"] = merged_df[f"speedup_max@{k}"].fillna(0.0)
        merged_df[f"efficiency@{k}"] = merged_df[f"efficiency@{k}"].fillna(0.0)
        merged_df[f"efficiency_max@{k}"] = merged_df[f"efficiency_max@{k}"].fillna(0.0)

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