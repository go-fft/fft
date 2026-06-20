package fft

import "sync"

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
// For even N the real signal is packed into an N/2-point complex transform and
// untangled — about twice as fast as a full complex FFT — using twiddle tables
// cached per length. The input is not modified. Empty input returns an empty
// (non-nil) slice.
func RFFT(x []float64) []complex128 {
	n := len(x)
	if n == 0 {
		return []complex128{}
	}
	dst := make([]complex128, n/2+1)
	return cachedRealPlan(n).RFFT(dst, x)
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
	dst := make([]float64, n)
	return cachedRealPlan(n).IRFFT(dst, spectrum)
}

// realPlanCache memoizes RealPlans by length so RFFT/IRFFT reuse twiddle tables.
var (
	realPlanMu    sync.Mutex
	realPlanCache = map[int]*RealPlan{}
)

// cachedRealPlan returns a shared real-input plan for length n, building and
// memoizing it on first use.
func cachedRealPlan(n int) *RealPlan {
	realPlanMu.Lock()
	p, ok := realPlanCache[n]
	if !ok {
		p = NewRealPlan(n)
		realPlanCache[n] = p
	}
	realPlanMu.Unlock()
	return p
}
