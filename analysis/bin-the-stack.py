""" Count the number of times different parallel programming models are used
    in The Stack data set.
    See https://huggingface.co/datasets/bigcode/the-stack for more info on
    The Stack data set.
    author: Daniel Nichols
    date: December 2023
"""
# std imports
from argparse import ArgumentParser
from collections import Counter
import json
from multiprocessing import Pool
from typing import List

# tpl imports
from alive_progress import alive_it
from datasets import load_dataset
from numba import njit, jit
from tqdm import tqdm


""" io helper funtions """
def write_json(obj: str, fpath: str):
    with open(fpath, 'w') as fp:
        json.dump(obj, fp)
    
def read_json(fpath: str):
    with open(fpath, 'r') as fp:
        return json.load(fp)


""" Helper functions for checking model type """
@njit
def any_in(s: str, substrs: List[str]) -> bool:
    for substr in substrs:
        if substr in s:
            return True
    return False

@njit
def uses_mpi(contents: str, language: str) -> bool:
    if language in ['C', 'C++']:
        return any_in(contents, ['MPI_', '#include <mpi.h>', '#include "mpi.h"'])
    elif language == 'FORTRAN':
        return any_in(contents, ['MPI_', 'include \'mpif.h\'', 'include "mpif.h"'])
    elif language == 'Python':
        return any_in(contents, ['mpi4py'])
    return False

@njit
def uses_omp(contents: str, language: str) -> bool:
    if language in ['C', 'C++']:
        return any_in(contents, ['#pragma omp', '#include <omp.h>'])
    elif language == 'FORTRAN':
        return any_in(contents, ['!$OMP PARALLEL ', 'USE OMP_LIB'])
    return False

@njit
def uses_kokkos(contents: str, language: str) -> bool:
    if language in ['C++']:
        return any_in(contents, ['#include <Kokkos_Core.hpp>', 'Kokkos::'])
    return False

@njit
def uses_cuda(contents: str, language: str) -> bool:
    return language == 'Cuda'

def count_row(row):
    contents, language = row
    if language in ['C', 'C++', 'FORTRAN', 'Python', 'Cuda']:
        mpi = uses_mpi(contents, language)
        omp = uses_omp(contents, language)
        kokkos = uses_kokkos(contents, language)
        cuda = uses_cuda(contents, language)
    else:
        mpi, omp, kokkos, cuda = False, False, False, False
    return mpi, omp, kokkos, cuda, language


#def count_models_parallel(batch, pool, chunksize=1000):
#    models = Counter()
#    languages = Counter()
#
#    results = pool.imap_unordered(count_row, zip(batch['content'], batch['lang']), chunksize=chunksize)
#    for mpi, omp, kokkos, cuda, language in results:
#        models["mpi"] += 1 if mpi else 0
#        models["omp"] += 1 if omp else 0
#        models["kokkos"] += 1 if kokkos else 0
#        models["cuda"] += 1 if cuda else 0
#        languages[language] += 1
#        models["total"] += 1
#        languages["total"] += 1
#    
#    return models, languages

def count_models(batch):
    models = Counter()
    languages = Counter()

    for contents, language in zip(batch['content'], batch['lang']):
        if language in ['C', 'C++', 'FORTRAN', 'Python', 'Cuda']:
            models["mpi"] += 1 if uses_mpi(contents, language) else 0
            models["omp"] += 1 if uses_omp(contents, language) else 0
            models["kokkos"] += 1 if uses_kokkos(contents, language) else 0
            models["cuda"] += 1 if uses_cuda(contents, language) else 0
        languages[language] += 1
        models["total"] += 1
        languages["total"] += 1
    
    return models, languages


""" Parse Args """
parser = ArgumentParser(description=__doc__)
parser.add_argument("-p", "--num_processes", type=int, help="number of processes")
parser.add_argument("-c", "--chunk_size", type=int, help="chunk size for multiprocessing")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--skip", type=int, help="skip first n batches")
args = parser.parse_args()

""" Create the data set """
print("Getting streaming data set...", flush=True)
dataset = load_dataset("bigcode/the-stack", streaming=True, split="train")
if args.skip:
    dataset = dataset.skip(args.skip * args.batch_size)
    counts = read_json(f'model-counts-{args.skip}.json')
    language_counts_json = read_json(f'language-counts-{args.skip}.json')
    dict.update(language_counts, language_counts_json)
print("Got data set.", flush=True)

""" Compute new columns """
print("Computing map...", flush=True)
THE_STACK_ROWS = 545_547_422
total_iter = THE_STACK_ROWS // args.batch_size + 1
if args.skip:
    total_iter -= args.skip

counts, language_counts = Counter(), Counter()
with Pool(processes=args.num_processes) as pool:
    chunksize = args.chunk_size if args.chunk_size else 1
    results = pool.imap(count_models, dataset.iter(batch_size=args.batch_size), chunksize=chunksize)

    #for idx, batch in alive_it(enumerate(batches), total=total_iter):
    for idx, (c, lc) in tqdm(enumerate(results), total=total_iter):
        #c, lc = count_models(batch)
        counts.update(c)
        language_counts.update(lc)

        if idx != 0 and idx % 150_000 == 0:
            offset = args.skip if args.skip else 0
            write_json(counts, f'model-counts-{idx + offset}.json')
            write_json(language_counts, f'language-counts-{idx + offset}.json')

print("Final counts:")
print(counts)
print(language_counts, flush=True)

write_json(counts, 'model-counts.json')
write_json(language_counts, 'language-counts.json')