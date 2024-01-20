// Driver for 11_geometry_convex_hull_perimeter for Kokkos
// #include <Kokkos_Core.hpp>
//
// struct Point {
//     double x, y;
// };
//
// double distance(Point const& p1, Point const& p2) {
//     return std::sqrt(std::pow(p2.x-p1.x, 2) + std::pow(p2.y-p1.y, 2));
// }
//
// /* Return the perimeter of the smallest convex polygon that contains all the points in the vector points.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
//
//    input: [{0, 3}, {1, 1}, {2, 2}, {4, 4}, {0, 0}, {1, 2}, {3, 1}, {3, 3}]
//    output: 13.4477
// */
// double convexHullPerimeter(Kokkos::View<const Point*> &points) {

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "kokkos-includes.hpp"
#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<Point*> points;
    Kokkos::View<const Point*> pointsConst;

    std::vector<Point> h_points;
    std::vector<double> h_x, h_y;
};

void reset(Context *ctx) {
    fillRand(ctx->h_x, -1000.0, 1000.0);
    fillRand(ctx->h_y, -1000.0, 1000.0);
    for (size_t i = 0; i < ctx->points.size(); i++) {
        ctx->h_points[i].x = ctx->h_x[i];
        ctx->h_points[i].y = ctx->h_y[i];
    }

    copyVectorToView(ctx->h_points, ctx->points);
    ctx->pointsConst = ctx->points;
}

Context *init() {
    Context *ctx = new Context();

    ctx->h_points.resize(DRIVER_PROBLEM_SIZE);
    ctx->h_x.resize(DRIVER_PROBLEM_SIZE);
    ctx->h_y.resize(DRIVER_PROBLEM_SIZE);

    ctx->points = Kokkos::View<Point*>("points", DRIVER_PROBLEM_SIZE);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    double perimeter = convexHullPerimeter(ctx->pointsConst);
    (void)perimeter;
}

void NO_OPTIMIZE best(Context *ctx) {
    double perimeter = correctConvexHullPerimeter(ctx->h_points);
    (void)perimeter;
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<Point> h_points(TEST_SIZE);
    std::vector<double> x(TEST_SIZE), y(TEST_SIZE);
    double correct = 0.0, test = 0.0;

    Kokkos::View<Point*> points("points", TEST_SIZE);
    Kokkos::View<const Point*> pointsConst = points;

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int trialIter = 0; trialIter < numTries; trialIter += 1) {
        // set up input
        fillRand(x, -1000.0, 1000.0);
        fillRand(y, -1000.0, 1000.0);
        test = 0.0;
        correct = 0.0;

        for (size_t i = 0; i < h_points.size(); i++) {
            h_points[i].x = x[i];
            h_points[i].y = y[i];
        }

        copyVectorToView(h_points, points);
        pointsConst = points;

        // compute correct result
        correct = correctConvexHullPerimeter(h_points);

        // compute test result
        test = convexHullPerimeter(pointsConst);

        if (std::abs(correct - test) > 1e-4) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
