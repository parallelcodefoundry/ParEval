/* Main driver for generated openmp C++ code. This relies on five externally available
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
#include <cfloat>

#include <omp.h>

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

    /* initialize settings from arguments */
    if (argc > 2) {
        printf("Usage: %s <?num_threads>\n", argv[0]);
        exit(1);
    }

    const int NITER = 5;
    int num_threads = 1;
    if (argc > 1) {
        num_threads = std::stoi(std::string(argv[1]));
    }
    omp_set_num_threads(num_threads);

    /* initialize */
    Context *ctx = init();

    /* validate */
    const bool isValid = validate(ctx);
    printf("Validation: %s\n", isValid ? "PASS" : "FAIL");
    if (!isValid) {
        destroy(ctx);
        return 0;
    }

    /* benchmark */
    double totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = omp_get_wtime();
        compute(ctx);
        totalTime += omp_get_wtime() - start;
    
        reset(ctx);
    }
    printf("Time: %.*f\n", DBL_DIG-1, totalTime / NITER);

    /* benchmark best */
    totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = omp_get_wtime();
        best(ctx);
        totalTime += omp_get_wtime() - start;

        reset(ctx);
    }
    printf("BestSequential: %.*f\n", DBL_DIG-1, totalTime / NITER);

    /* cleanup */
    destroy(ctx);

    return 0;
}