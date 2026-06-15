package fft

import "math"

// RFFT returns the forward discrete Fourier transform of a real-valued signal,
// keeping only the non-redundant spectrum. For input of length N the result has
// length N/2+1 (integer division): bins 0..N/2. The discarded upper half is the
// conjugate mirror of the lower half (X[N-k] = conj(X[k])), so it carries no new
// information. This matches numpy.fft.rfft.
//
// The transform is unnormalized, consistent with FFT:
//
//	X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi·k·n/N),  k = 0 .. N/2.
//
// The input is not modified. Empty input returns an empty (non-nil) slice.
func RFFT(x []float64) []complex128 {
	n := len(x)
	if n == 0 {
		return []complex128{}
	}
	if n == 1 {
		return []complex128{complex(x[0], 0)}
	}
	// Odd lengths cannot use the half-length packing trick (which needs N=2m);
	// fall back to the full complex transform and slice the kept bins. This path
	// is still correct and O(N log N) for arbitrary N via Bluestein.
	if n%2 != 0 {
		full := FFTReal(x)
		return full[:n/2+1]
	}

	// Even length N = 2m: pack the real signal into a length-m complex array
	// z[j] = x[2j] + i·x[2j+1], take one complex FFT of size m, then untangle.
	m := n / 2
	z := make([]complex128, m)
	for j := 0; j < m; j++ {
		z[j] = complex(x[2*j], x[2*j+1])
	}
	Z := FFT(z)

	out := make([]complex128, m+1)
	// The DFT of the even-indexed samples (Xe) and odd-indexed samples (Xo) are
	// recovered from Z via Z[k] = Xe[k] + i·Xo[k] and the conjugate symmetry of
	// real-input transforms: Z[m-k] = conj(Xe[k]) + i·conj(Xo[k]).
	for k := 0; k <= m; k++ {
		zk := Z[k%m]
		zmk := Z[(m-k)%m]
		// Xe[k] = (Z[k] + conj(Z[m-k])) / 2
		// Xo[k] = (Z[k] - conj(Z[m-k])) / (2i)
		xe := (zk + complexConj(zmk)) * 0.5
		xo := (zk - complexConj(zmk)) * complex(0, -0.5)
		// X[k] = Xe[k] + exp(-2πi·k/N)·Xo[k].
		ang := -2 * math.Pi * float64(k) / float64(n)
		tw := complex(math.Cos(ang), math.Sin(ang))
		out[k] = xe + tw*xo
	}
	return out
}

// IRFFT inverts RFFT, reconstructing a real-valued signal of length n from the
// n/2+1 non-redundant spectral bins produced by RFFT. The caller passes the
// desired output length n explicitly, because n and n-1 yield the same number
// of kept bins so the half spectrum alone is ambiguous; this mirrors
// numpy.fft.irfft(spectrum, n).
//
// spectrum is read up to min(len(spectrum), n/2+1) bins; any bins beyond that
// (or beyond what the Hermitian mirror needs) are treated as zero. The result
// is normalized by n so that IRFFT(RFFT(x), len(x)) ≈ x.
//
// The input is not modified. n <= 0 returns an empty (non-nil) slice.
func IRFFT(spectrum []complex128, n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	// Rebuild the full length-n Hermitian spectrum from the half spectrum, then
	// run the standard inverse transform and take the real parts. X[n-k] is the
	// conjugate of X[k]; bins not supplied by the caller stay zero.
	full := make([]complex128, n)
	half := n/2 + 1
	for k := 0; k < half && k < len(spectrum); k++ {
		full[k] = spectrum[k]
		if k > 0 && k < n-k {
			full[n-k] = complexConj(spectrum[k])
		}
	}
	inv := IFFT(full)
	out := make([]float64, n)
	for i := range out {
		out[i] = real(inv[i])
	}
	return out
}
