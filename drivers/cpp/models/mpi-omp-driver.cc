/* Main driver for generated MPI+OMP C++ code. This relies on five externally available
* functions:
*
*   - init() -- returns a pointer to a context object
*   - compute(Context *ctx) -- runs the benchmark
*   - best(Context *ctx) -- runs the best sequential code
*   - validate(Context *ctx) -- returns true if the benchmark is valid
*   - reset(Context *ctx) -- resets the benchmark
*   - destroy(Context *ctx) -- frees the context object
*
* These functions are defined in the driver for the given benchmark and handle
* the data and calling the generated code.
*/
#include <cstdio>
#include <string>

#include <mpi.h>
#include <omp.h>


class Context;
extern "C++" {
    /* todo -- these could all be in a class, but I'm not sure if virtual 
       overloading would incur a noticable overhead here. */
    Context *init();                // initialize data in context
    void compute(Context *ctx);     // benchmark the generated code
    void best(Context *ctx);        // benchmark the best code
    bool validate(Context *ctx);    // validate the generated code
    void reset(Context *ctx);       // reset data in context
    void destroy(Context *ctx);     // free context
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    /* initialize settings from arguments */
    if (argc > 2) {
        printf("Usage: %s <?num_threads>\n", argv[0]);
        exit(1);
    }

    const int NITER = 50;
    int num_threads = 1;
    if (argc > 1) {
        num_threads = std::stoi(std::string(argv[1]));
    }
    omp_set_num_threads(num_threads);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* initialize */
    Context *ctx = init();

    /* benchmark */
    double totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = MPI_Wtime();
        compute(ctx);
        totalTime += MPI_Wtime() - start;
    
        MPI_Barrier(MPI_COMM_WORLD);
        reset(ctx);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &totalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        printf("Time: %f\n", totalTime / size / NITER);
    } else {
        MPI_Reduce(&totalTime, nullptr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* best case */
    if (rank == 0) {
        totalTime = 0.0;
        for (int i = 0; i < NITER; i += 1) {
            double start = MPI_Wtime();
            best(ctx);
            totalTime += MPI_Wtime() - start;

            reset(ctx);
        }
        printf("BestSequential: %f\n", totalTime / NITER);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* validate */
    const bool isValid = validate(ctx);
    if (rank == 0) {
        printf("Validation: %s\n", isValid ? "PASS" : "FAIL");
    }

    /* cleanup */
    destroy(ctx);

    MPI_Finalize();
    return 0;
}