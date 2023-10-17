/** Main driver for generated MPI C++ code. This relies on five externally available
* functions:
*
*   - init() -- returns a pointer to a context object
*   - benchmark(Context *ctx) -- runs the benchmark
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


class Context;
extern "C++" {
    /* todo -- these could all be in a class, but I'm not sure if virtual 
       overloading would incur a noticable overhead here. */
    Context *init();
    void benchmark(Context *ctx);
    bool validate(Context *ctx);
    void reset(Context *ctx);
    void destroy(Context *ctx);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    /* initialize settings from arguments */
    if (argc > 2) {
        printf("Usage: %s <?num_iter>\n", argv[0]);
        exit(1);
    }

    int NITER = 50;
    if (argc > 1) {
        NITER = std::stoi(std::string(argv[1]));
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* initialize */
    Context *ctx = init();

    /* benchmark */
    double totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = MPI_Wtime();
        benchmark(ctx);
        totalTime += MPI_Wtime() - start;
    
        MPI_Barrier(MPI_COMM_WORLD);
        reset(ctx);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Reduce(MPI_IN_PLACE, &totalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Time: %f\n", totalTime / NITER);
    }

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
