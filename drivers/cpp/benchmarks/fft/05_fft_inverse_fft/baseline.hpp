#pragma once
#include <cmath>
#include <vector>
#include <complex>

#if !defined(M_PI)
#define M_PI 3.14159265358979323846
#endif

/* fft. computes fourier transform in-place
   from https://rosettacode.org/wiki/Fast_Fourier_transform#C++
*/
void fft(std::vector<std::complex<double>> &x) {
   // DFT
	unsigned int N = x.size(), k = N, n;
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
				std::complex<double> t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)std::log2(N);
	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			std::complex<double> t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
}

/* Compute the inverse fourier transform of x in-place.
   Example:
   
   input: [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
   output: [{0.5,0}, {0.125,0.301777}, {0,-0}, {0.125,0.0517767}, {0,-0}, {0.125,-0.0517767}, {0,-0}, {0.125,-0.301777}]
*/
void NO_INLINE correctIfft(std::vector<std::complex<double>> &x) {
   // conjugate the complex numbers
   std::transform(x.begin(), x.end(), x.begin(), [](auto const& val) { return std::conj(val); });

   // forward fft
   fft( x );

   // conjugate the complex numbers again
   std::transform(x.begin(), x.end(), x.begin(), [](auto const& val) { return std::conj(val); });

   // scale the numbers
   std::transform(x.begin(), x.end(), x.begin(), [&](std::complex<double> c) { return c / static_cast<double>(x.size()); });
}

#if defined(USE_CUDA)
// a lot of model outputs assume this is defined for some reason, so just define it
__device__ DOUBLE_COMPLEX_T cexp(DOUBLE_COMPLEX_T arg) {
   DOUBLE_COMPLEX_T res;
   float s, c;
   float e = expf(arg.x);
   sincosf(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}

__device__ DOUBLE_COMPLEX_T cuCexp(DOUBLE_COMPLEX_T arg) {
   return cexp(arg);
}

__device__ DOUBLE_COMPLEX_T hipCexp(DOUBLE_COMPLEX_T arg) {
   return cexp(arg);
}
#endif