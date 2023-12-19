#pragma once
#include <vector>
#include <cmath>
#include <complex>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

/* Compute the fourier transform of x. Store the result in output.
   Example:

   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: [{4,0}, {1,-2.42421}, {0,0}, {1,-0.414214}, {0,0}, {1,0.414214}, {0,0}, {1,2.41421}]
*/
void correctFft(std::vector<std::complex<double>> const& x, std::vector<std::complex<double>> &output) {
   output = x;

   // DFT
	unsigned int N = output.size(), k = N, n;
	double thetaT = 3.14159265358979323846264338328L / N;
	std::complex<double> phiT = std::complex<double>(std::cos(thetaT), -std::sin(thetaT)), T;
	while (k > 1) {
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				std::complex<double> t = output[a] - output[b];
				output[a] += output[b];
				output[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)std::log2(N);
	for (unsigned int a = 0; a < N; a++) {
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a) {
			std::complex<double> t = output[a];
			output[a] = output[b];
			output[b] = t;
		}
	}
}