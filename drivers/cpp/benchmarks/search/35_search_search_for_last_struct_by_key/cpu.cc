// Driver for 35_search_search_for_last_struct_by_key
// /* Return the index of the last Book item in the vector books where Book.pages is less than 100.
//    Example:
//    
//    input: [{title=\"Green Eggs and Ham\", pages=72}, {title=\"gulliver's travels\", pages=362}, {title=\"Stories of Your Life\", pages=54}, {title=\"Hamilton\", pages=818}]
//    output: 2
// */
// size_t findLastShortBook(std::vector<Book> const& books) {

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM
#include "baseline.hpp"

struct Context {
    std::vector<Book> books;
    std::vector<int> pages;
    std::vector<std::string> titles;
};

void reset(Context *ctx) {
    fillRandString(ctx->titles, 5, 15);
    fillRand(ctx->pages, 101, 1000);
    size_t min = 0;
    size_t max = ctx->pages.size() / 4;
    ctx->pages[rand() % (max - min) + min] = 72;  // make sure there is at least one book with < 100 pages
    BCAST(ctx->pages, INT);

    for (int i = 0; i < ctx->books.size(); i += 1) {
        ctx->books[i].title = ctx->titles[i];
        ctx->books[i].pages = ctx->pages[i];
    }
}

Context *init() {
    Context *ctx = new Context();
    ctx->books.resize(1 << 18);
    ctx->pages.resize(1 << 18);
    ctx->titles.resize(1 << 18);
    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    size_t idx = findLastShortBook(ctx->books);
    (void)idx;
}

void NO_OPTIMIZE best(Context *ctx) {
    size_t idx = correctFindLastShortBook(ctx->books);
    (void)idx;
}

bool validate(Context *ctx) {

    int rank;
    GET_RANK(rank);

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> pages(1024);
        std::vector<Book> input(1024);
        fillRand(pages, 1, 1000);
        pages[rand() % pages.size()] = 72;  // make sure there is at least one book with < 100 pages
        BCAST(pages, INT);
        for (int j = 0; j < input.size(); j += 1) {
            input[j].title = "title";
            input[j].pages = pages[j];
        }

        // compute correct result
        size_t correctIdx = correctFindLastShortBook(input);

        // compute test result
        size_t testIdx = findLastShortBook(input);
        SYNC();
        
        bool isCorrect = true;
        if (IS_ROOT(rank) && correctIdx != testIdx) {
            isCorrect = false;
        }
        BCAST_PTR(&isCorrect, 1, CXX_BOOL);
        if (!isCorrect) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}


