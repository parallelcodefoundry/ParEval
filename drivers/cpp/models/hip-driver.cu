/* Main driver for generated HIP C++ code. This relies on five externally available
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
#include <cfloat>
#include <string>

#include <hip/hip_runtime.h>

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


class HipTimer {
  public:
    HipTimer() {
        hipEventCreate(&startEvent);
        hipEventCreate(&stopEvent);
    }

    ~HipTimer() {
        hipEventDestroy(startEvent);
        hipEventDestroy(stopEvent);
    }

    void start() {
        hipEventRecord(startEvent, 0);
    }

    void stop() {
        hipEventRecord(stopEvent, 0);
        hipEventSynchronize(stopEvent);
    }

    double elapsed() {
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, startEvent, stopEvent);
        return static_cast<double>(milliseconds);
    }

    void reset() {
        hipEventRecord(startEvent, 0);
        hipEventRecord(stopEvent, 0);
        hipEventSynchronize(stopEvent);
    }

  private:
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
};


int main(int argc, char **argv) {

    /* initialize settings from arguments */
    if (argc > 1) {
        printf("Usage: %s\n", argv[0]);
        exit(1);
    }

    const int NITER = 5;

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
    HipTimer timer;
    for (int i = 0; i < NITER; i += 1) {
        timer.start();
        compute(ctx);
        timer.stop();
        totalTime += timer.elapsed();
        timer.reset();
    
        reset(ctx);
        hipDeviceSynchronize();
    }
    printf("Time: %.*f\n", DBL_DIG-1, totalTime / NITER / 1000.0);

    /* benchmark best */
    totalTime = 0.0;
    for (int i = 0; i < NITER; i += 1) {
        auto start = std::chrono::high_resolution_clock::now();
        best(ctx);
        auto end = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        reset(ctx);
    }
    printf("BestSequential: %.*f\n", DBL_DIG-1, totalTime / NITER);

    /* cleanup */
    destroy(ctx);

    return 0;
}