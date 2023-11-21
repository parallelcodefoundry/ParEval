/* Main driver for generated openmp C++ code. This relies on five externally available
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

#include <omp.h>

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

    /* initialize settings from arguments */
    if (argc > 2) {
        printf("Usage: %s <?num_threads>\n", argv[0]);
        exit(1);
    }

    int NITER = 50;
    int num_threads = 1;
    if (argc > 1) {
        num_threads = std::stoi(std::string(argv[1]));
    }
    omp_set_num_threads(num_threads);

    /* initialize */
    Context *ctx = init();

    /* benchmark */
    double totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        double start = omp_get_wtime();
        benchmark(ctx);
        totalTime += omp_get_wtime() - start;
    
        reset(ctx);
    }
    printf("Time: %f\n", totalTime / NITER);

    /* validate */
    const bool isValid = validate(ctx);
    printf("Validation: %s\n", isValid ? "PASS" : "FAIL");

    /* cleanup */
    destroy(ctx);

    return 0;
}