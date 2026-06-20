package fft

import "math"

// A RealPlan is a reusable, precomputed real-input transform of a fixed length
// N. Like Plan it amortizes all twiddle computation across calls; it is the
// real-signal counterpart used by RFFT/IRFFT. RealPlans are immutable after
// construction and safe for concurrent use.
//
// For even N the plan packs the N real samples into an N/2-point complex
// transform and untangles the result with a precomputed table — about twice as
// fast as a full complex FFT. For odd N (which cannot be packed) it wraps a
// full complex plan of length N.
type RealPlan struct {
	n int

	// Even path: half-length complex plan + untangle table.
	half *Plan        // length n/2, nil for the odd path
	tw   []complex128 // exp(-2πi·k/n) for k = 0 .. n/2, untangle twiddles

	// Odd path: full complex plan of length n.
	full *Plan
}

// NewRealPlan returns a real-input transform plan for length n, precomputing
// all twiddle factors. n must be non-negative.
func NewRealPlan(n int) *RealPlan {
	p := &RealPlan{n: n}
	if n <= 1 || n%2 != 0 {
		// Trivial or odd: defer to a full complex plan of length n.
		if n >= 1 {
			p.full = cachedPlan(n)
		}
		return p
	}
	m := n / 2
	p.half = cachedPlan(m)
	p.tw = make([]complex128, m+1)
	for k := 0; k <= m; k++ {
		ang := -2 * math.Pi * float64(k) / float64(n)
		p.tw[k] = complex(math.Cos(ang), math.Sin(ang))
	}
	return p
}

// Len reports the real-input length the plan was built for.
func (p *RealPlan) Len() int { return p.n }

// RFFT writes the non-redundant N/2+1 spectral bins of the real signal src into
// dst and returns dst. src must have length Len(); dst must have length
// Len()/2+1. src is not modified.
func (p *RealPlan) RFFT(dst []complex128, src []float64) []complex128 {
	n := p.n
	if n == 0 {
		return dst[:0]
	}
	if n == 1 {
		dst[0] = complex(src[0], 0)
		return dst[:1]
	}
	if p.half == nil {
		// Odd length: full complex transform, keep the lower bins.
		c := make([]complex128, n)
		for i, v := range src {
			c[i] = complex(v, 0)
		}
		full := make([]complex128, n)
		p.full.FFT(full, c)
		copy(dst, full[:n/2+1])
		return dst[:n/2+1]
	}

	// Even length N = 2m: pack z[j] = src[2j] + i·src[2j+1], one m-point FFT.
	m := n / 2
	z := make([]complex128, m)
	for j := 0; j < m; j++ {
		z[j] = complex(src[2*j], src[2*j+1])
	}
	Z := make([]complex128, m)
	if p.half.sr != nil {
		// m is a power of two: feed the freshly packed buffer straight to the
		// split-radix kernel as its scratch, skipping the engine's internal copy
		// (and its allocation). z is local, so consuming it here is safe.
		p.half.sr.transformScratch(Z, z, false)
	} else {
		p.half.FFT(Z, z)
	}

	// Untangle. The k=0 and k=m bins wrap on Z (indices 0 and 0) and are both
	// purely real, so they are handled outside the loop; the interior k in
	// [1,m-1] reads Z[k] and Z[m-k] with no modulo. The even/odd split and the
	// twiddle recombination are expanded into real arithmetic to avoid the
	// per-bin complex-multiply helper calls.
	z0r, z0i := real(Z[0]), imag(Z[0])
	dst[0] = complex(z0r+z0i, 0)
	dst[m] = complex(z0r-z0i, 0)
	for k := 1; k < m; k++ {
		zk := Z[k]
		zmk := Z[m-k]
		// xe = (zk + conj(zmk))/2, xo = (zk - conj(zmk))·(-i/2).
		xer := (real(zk) + real(zmk)) * 0.5
		xei := (imag(zk) - imag(zmk)) * 0.5
		// (zk - conj(zmk)) = (real(zk)-real(zmk)) + i(imag(zk)+imag(zmk));
		// times -i/2 swaps and scales: xo = (imag-sum)/2 - i(real-diff)/2.
		sumI := (imag(zk) + imag(zmk)) * 0.5
		difR := (real(zk) - real(zmk)) * 0.5
		xor := sumI
		xoi := -difR
		// dst[k] = xe + tw[k]·xo.
		wr, wi := real(p.tw[k]), imag(p.tw[k])
		dst[k] = complex(xer+wr*xor-wi*xoi, xei+wr*xoi+wi*xor)
	}
	return dst[:m+1]
}

// IRFFT writes the real signal of length Len() reconstructed from the half
// spectrum src into dst and returns dst. dst must have length Len(); src is
// read up to min(len(src), Len()/2+1) bins. The result is normalized by N. src
// is not modified.
func (p *RealPlan) IRFFT(dst []float64, src []complex128) []float64 {
	n := p.n
	if n <= 0 {
		return dst[:0]
	}
	full := make([]complex128, n)
	half := n/2 + 1
	for k := 0; k < half && k < len(src); k++ {
		full[k] = src[k]
		if k > 0 && k < n-k {
			full[n-k] = complexConj(src[k])
		}
	}
	inv := make([]complex128, n)
	cachedPlan(n).IFFT(inv, full)
	for i := 0; i < n; i++ {
		dst[i] = real(inv[i])
	}
	return dst
}
