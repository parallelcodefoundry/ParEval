# Analysis

This subdirectory contains scripts for analyzing the LLM outputs and driver
results.

`create-dataframe.py` -- convert results json files into CSV format

`metrics.py` -- compute pass@k, efficiency@k, speedup@k, and build@k for a 
particular results csv file

`metrics-scaling.py` -- compute the metrics at different resource counts; used
to get scaling results

`bin-the-stack.py` -- a utility script for analyzing The Stack dataset.

The arguments to each of these scripts can be found with `--help`. In general,
the workflow is to use `create-dataframe.py` to get a CSV results file for the
driver outputs and then feed this into `metrics.py` to get the relevant metrics.
This will in turn output another CSV file with the metrics divided by problem
type and execution model.