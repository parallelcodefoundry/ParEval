// Driver for 23_histogram_first_letter_counts for Kokkos
// #include <Kokkos_Core.hpp>
// 
// /* For each letter in the alphabet, count the number of strings in the vector s that start with that letter.
//    Assume all strings are in lower case. Store the output in `bins` array.
//    Use Kokkos to compute in parallel. Assume Kokkos has already been initialized.
//    Example:
// 
//    input: ["dog", "cat", "xray", "cow", "code", "type", "flower"]
//    output: [0, 0, 3, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
// */
// void firstLetterCounts(Kokkos::View<const char**> const& s, Kokkos::View<size_t[26]> &bins) {

#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

struct Context {
    Kokkos::View<const char**> s;
    Kokkos::View<char**> sNonConst;
    Kokkos::View<size_t[26]> bins;

    std::vector<std::string> s_host;
    std::array<size_t, 26> bins_host;
};

void reset(Context *ctx) {
    fillRandString(ctx->s_host, 10, 11);
    ctx->bins_host.fill(0);

    for (int i = 0; i < ctx->s_host.size(); i += 1) {
        for (int j = 0; j < ctx->s_host[i].size(); j += 1) {
            ctx->sNonConst(i, j) = ctx->s_host[i][j];
        }
    }
    ctx->s = ctx->sNonConst;
    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), ctx->bins, 0);
}

Context *init() {
    Context *ctx = new Context();

    ctx->s_host.resize(1 << 18);

    ctx->sNonConst = Kokkos::View<char**>("sNonConst", 1 << 18, 10);
    ctx->bins = Kokkos::View<size_t[26]>("bins");

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    firstLetterCounts(ctx->s, ctx->bins);
}

void NO_OPTIMIZE best(Context *ctx) {
    correctFirstLetterCounts(ctx->s_host, ctx->bins_host);
}

bool validate(Context *ctx) {
    const size_t TEST_SIZE = 1024;

    std::vector<std::string> s_host(TEST_SIZE);
    std::array<size_t, 26> correct;

    Kokkos::View<const char**> s;
    Kokkos::View<char**> sNonConst = Kokkos::View<char**>("sNonConst", TEST_SIZE, 10);
    Kokkos::View<size_t[26]> test = Kokkos::View<size_t[26]>("test");

    const size_t numTries = MAX_VALIDATION_ATTEMPTS;
    for (int i = 0; i < numTries; i += 1) {
        // set up input
        fillRandString(s_host, 10, 11);
        correct.fill(0);

        for (int j = 0; j < s_host.size(); j += 1) {
            for (int k = 0; k < s_host[j].size(); k += 1) {
                sNonConst(j, k) = s_host[j][k];
            }
        }
        s = sNonConst;
        Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), test, 0);

        // compute correct result
        correctFirstLetterCounts(s_host, correct);

        // compute test result
        firstLetterCounts(s, test);
        
        for (int j = 0; j < 26; j += 1) {
            if (test(j) != correct[j]) {
                return false;
            }
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}
