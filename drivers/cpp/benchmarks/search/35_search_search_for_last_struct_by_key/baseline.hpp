#pragma once
#include <vector>

// defined in generated code
//struct Book {
//    std::string title;
//    int pages;
//};

/* Return the index of the last Book item in the vector books where Book.pages is less than 100.
   Example:
   
   input: [{title=\"Green Eggs and Ham\", pages=72}, {title=\"gulliver's travels\", pages=362}, {title=\"Stories of Your Life\", pages=54}, {title=\"Hamilton\", pages=818}]
   output: 2
*/
size_t NO_INLINE correctFindLastShortBook(std::vector<Book> const& books) {
    for (int i = books.size() - 1; i >= 0; i--) {
        if (books[i].pages < 100) {
            return i;
        }
    }
    return books.size();
}