#pragma once
#include <vector>

/* Solve the linear system Ax=b for x.
   A is an NxN matrix in row-major. x and b have N elements.
   Example:
   
   input: A=[[1,4,2], [1,2,3], [2,1,3]] b=[11, 11, 13]
   output: x=[3, 1, 2]
*/
void solveLinearSystem(std::vector<double> const& A, std::vector<double> const& b, std::vector<double> &x, size_t N) {
	
}