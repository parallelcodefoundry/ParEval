/* Main driver for generated openmp C++ code. This relies on two externally available
* functions:
*   - Context init() which initializes the problem, data etc.
*   - void loop() which is called repeatedly
* These functions are defined in the generated code.
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
    if (argc > 3) {
        printf("Usage: %s <?num_iter> <?num_threads>\n", argv[0]);
        exit(1);
    }

    int NITER = 50;
    if (argc > 1) {
        NITER = std::stoi(std::string(argv[1]));
    }

    int numThreads = 1;
    if (argc > 2) {
        numThreads = std::stoi(std::string(argv[2]));
    }
    omp_set_num_threads(numThreads);

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