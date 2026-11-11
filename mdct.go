package fft

import "sync"

// The modified discrete cosine transform (MDCT) and its inverse, the lapped
// transform of the audio codecs (AAC, Vorbis, Opus CELT, MP3). With N the number
// of coefficients, an MDCT maps 2N samples to N coefficients and an IMDCT maps N
// coefficients to 2N samples:
//
//	MDCT   X[k] = Σ_{n=0}^{2N-1} x[n] cos(π/N (n + 1/2 + N/2)(k + 1/2)),  k < N
//	IMDCT  y[n] = Σ_{k=0}^{N-1}  X[k] cos(π/N (n + 1/2 + N/2)(k + 1/2)),  n < 2N
//
// both unscaled; the scale argument of the methods multiplies the result. This is
// the definition of the Wikipedia article "Modified discrete cosine transform"
// (Princen and Bradley, 1986 for the transform, TDAC for the aliasing
// cancellation): with a window w satisfying w[n]² + w[n+N]² = 1 applied before the
// MDCT and after the IMDCT, overlap-adding blocks that advance by N samples
// reconstructs the signal when the IMDCT is scaled by 2/N.
//
// The algorithm is the one the article gives: fold the 2N inputs, as
// (a, b, c, d) → (−c_r − d, a − b_r) for the four quarters of the block and r the
// reversal, into N values, then one DCT-IV of length N (this package's DCT-IV is one
// N/2-point complex FFT). The IMDCT is the same DCT-IV followed by the unfolding
// (u, v) → (v, −v_r, −u_r, −u). N must be even and at least 2.

// MDCTPlan is a reusable MDCT of N coefficients (2N samples), in float64. A plan
// is immutable and safe for concurrent use; the methods allocate nothing in
// steady state.
type MDCTPlan struct {
	n    int
	dct  *DCTPlan
	pool sync.Pool // *[]float64 of length n
}

// NewMDCTPlan returns a plan for N coefficients. It panics if n is odd or below 2.
func NewMDCTPlan(n int) *MDCTPlan {
	checkMDCT(n)
	p := &MDCTPlan{n: n, dct: NewDCTPlan(n, 4)}
	p.pool.New = func() any { s := make([]float64, n); return &s }
	return p
}

func checkMDCT(n int) {
	if n < 2 || n%2 != 0 {
		panic("fft: MDCT needs an even number of coefficients, at least 2")
	}
}

// Len reports N, the number of coefficients.
func (p *MDCTPlan) Len() int { return p.n }

// MDCT writes scale times the MDCT of the 2N samples of src into dst[:N] and
// returns it. dst must hold at least N values and src at least 2N; they must not
// overlap. src is not modified.
func (p *MDCTPlan) MDCT(dst, src []float64, scale float64) []float64 {
	n, h := p.n, p.n/2
	if len(dst) < n || len(src) < 2*n {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	sp := p.pool.Get().(*[]float64)
	u := *sp
	for i := 0; i < h; i++ {
		u[i] = -src[n+h-1-i] - src[n+h+i]
		u[h+i] = src[i] - src[n-1-i]
	}
	p.dct.DCT(u, u, NormBackward)
	f := scale / 2 // the package's DCT-IV is twice the sum
	for i := 0; i < n; i++ {
		dst[i] = u[i] * f
	}
	p.pool.Put(sp)
	return dst[:n]
}

// IMDCT writes scale times the inverse MDCT of the N coefficients of src into
// dst[:2N] and returns it. dst must hold at least 2N values and src at least N;
// they must not overlap. src is not modified. A scale of 2/N makes the
// overlap-add of windowed blocks reconstruct the signal (see above).
func (p *MDCTPlan) IMDCT(dst, src []float64, scale float64) []float64 {
	n, h := p.n, p.n/2
	if len(dst) < 2*n || len(src) < n {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	sp := p.pool.Get().(*[]float64)
	u := *sp
	p.dct.DCT(u, src, NormBackward)
	f := scale / 2
	for i := 0; i < h; i++ {
		s, t := u[i]*f, u[h+i]*f
		dst[i] = t
		dst[n-1-i] = -t
		dst[n+h-1-i] = -s
		dst[n+h+i] = -s
	}
	p.pool.Put(sp)
	return dst[:2*n]
}

// MDCTPlan32 is the single-precision counterpart of MDCTPlan.
type MDCTPlan32 struct {
	n    int
	dct  *DCTPlan32
	pool sync.Pool // *[]float32 of length n
}

// NewMDCTPlan32 returns a single-precision plan for N coefficients. It panics if
// n is odd or below 2.
func NewMDCTPlan32(n int) *MDCTPlan32 {
	checkMDCT(n)
	p := &MDCTPlan32{n: n, dct: NewDCTPlan32(n, 4)}
	p.pool.New = func() any { s := make([]float32, n); return &s }
	return p
}

// Len reports N, the number of coefficients.
func (p *MDCTPlan32) Len() int { return p.n }

// MDCT is MDCTPlan.MDCT on float32 values.
func (p *MDCTPlan32) MDCT(dst, src []float32, scale float32) []float32 {
	n, h := p.n, p.n/2
	if len(dst) < n || len(src) < 2*n {
		panic("fft: MDCTPlan32 slice shorter than the plan's length")
	}
	sp := p.pool.Get().(*[]float32)
	u := *sp
	for i := 0; i < h; i++ {
		u[i] = -src[n+h-1-i] - src[n+h+i]
		u[h+i] = src[i] - src[n-1-i]
	}
	p.dct.DCT(u, u, NormBackward)
	f := scale / 2
	for i := 0; i < n; i++ {
		dst[i] = u[i] * f
	}
	p.pool.Put(sp)
	return dst[:n]
}

// IMDCT is MDCTPlan.IMDCT on float32 values.
func (p *MDCTPlan32) IMDCT(dst, src []float32, scale float32) []float32 {
	n, h := p.n, p.n/2
	if len(dst) < 2*n || len(src) < n {
		panic("fft: MDCTPlan32 slice shorter than the plan's length")
	}
	sp := p.pool.Get().(*[]float32)
	u := *sp
	p.dct.DCT(u, src, NormBackward)
	f := scale / 2
	for i := 0; i < h; i++ {
		s, t := u[i]*f, u[h+i]*f
		dst[i] = t
		dst[n-1-i] = -t
		dst[n+h-1-i] = -s
		dst[n+h+i] = -s
	}
	p.pool.Put(sp)
	return dst[:2*n]
}
