/* Main driver for generated Kokkos C++ code. This relies on five externally available
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
#include <chrono>
#include <cstdio>
#include <string>

#include <Kokkos_Core.hpp>

class Context;
extern "C++" {
    /* todo -- these could all be in a class, but I'm not sure if virtual 
       overloading would incur a noticable overhead here. */
    Context *init();
    void compute(Context *ctx);
    void best(Context *ctx);
    bool validate(Context *ctx);
    void reset(Context *ctx);
    void destroy(Context *ctx);
}

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);

    /* initialize settings from arguments */
    if (argc > 1) {
        printf("Usage: %s\n", argv[0]);
        exit(1);
    }

    const int NITER = 50;

    /* initialize */
    Context *ctx = init();

    /* benchmark */
    double totalTime = 0.0;
    Kokkos::Timer timer;
    for (int i = 0; i < NITER; i += 1) {
        double start = timer.seconds();
        compute(ctx);
        totalTime += timer.seconds() - start;
    
        reset(ctx);
    }
    printf("Time: %f\n", totalTime / NITER);

    /* benchmark best */
    totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = timer.seconds();
        best(ctx);
        totalTime += timer.seconds() - start;

        reset(ctx);
    }
    printf("BestSequential: %f\n", totalTime / NITER);

    /* validate */
    const bool isValid = validate(ctx);
    printf("Validation: %s\n", isValid ? "PASS" : "FAIL");

    /* cleanup */
    destroy(ctx);

    Kokkos::finalize();

    return 0;
}