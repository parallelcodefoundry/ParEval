// Driver for 35_search_search_for_last_struct_by_key
// #include <Kokkos_Core.hpp>
// 
// struct Book {
//    const char *title;
//    int pages;
// };
// 
// /* Return the index of the last Book item in the vector books where Book.pages is less than 100.
//    Use Kokkos to search in parallel. Assume Kokkos is already initialized.
//    Example:
//    
//    input: [{title=\"Green Eggs and Ham\", pages=72}, {title=\"gulliver's travels\", pages=362}, {title=\"Stories of Your Life\", pages=54}, {title=\"Hamilton\", pages=818}]
//    output: 2
// */
// size_t findLastShortBook(Kokkos::View<const Book*> const& books) {

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "utilities.hpp"
#include "generated-code.hpp"   // code generated by LLM
#include "baseline.hpp"

struct Context {
    Kokkos::View<Book*> books;
    std::vector<Book> booksHost;
    std::vector<std::string> titles;
    std::vector<int> pages;
};

void reset(Context *ctx) {
    fillRand(ctx->pages, 101, 1000);
    fillRandString(ctx->titles, 5, 15);
    size_t min = 0;
    size_t max = ctx->pages.size() / 4;
    ctx->pages[rand() % (max - min) + min] = 72;  // make sure there is at least one book with < 100 pages

    for (int i = 0; i < ctx->pages.size(); i += 1) {
        ctx->books(i).title = ctx->titles[i].c_str();
        ctx->books(i).pages = ctx->pages[i];

        ctx->booksHost[i].title = ctx->titles[i].c_str();
        ctx->booksHost[i].pages = ctx->pages[i];
    }
}

Context *init() {
    Context *ctx = new Context();

    ctx->books = Kokkos::View<Book*>("books", 1 << 20);
    ctx->booksHost.resize(1 << 20);
    ctx->titles.resize(1 << 20);
    ctx->pages.resize(1 << 20);

    reset(ctx);
    return ctx;
}

void NO_OPTIMIZE compute(Context *ctx) {
    size_t idx = findLastShortBook(ctx->books);
    (void)idx;
}

void NO_OPTIMIZE best(Context *ctx) {
    size_t idx = correctFindLastShortBook(ctx->booksHost);
    (void)idx;
}

bool validate(Context *ctx) {

    const size_t numTries = 5;
    for (int i = 0; i < numTries; i += 1) {
        std::vector<int> pages(1024);
        std::vector<Book> input(1024);
        fillRand(pages, 1, 1000);
        pages[rand() % pages.size()] = 72;  // make sure there is at least one book with < 100 pages
        for (int j = 0; j < pages.size(); j += 1) {
            input[j].pages = pages[j];
        }

        // compute correct result
        size_t correctIdx = correctFindLastShortBook(input);

        // compute test result
        Kokkos::View<Book*> testView("testResult", input.size());
        copyVectorToView(input, testView);

        size_t testIdx = findLastShortBook(testView);
        
        if (testIdx != correctIdx) {
            return false;
        }
    }

    return true;
}

void destroy(Context *ctx) {
    delete ctx;
}


