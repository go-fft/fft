package fft

import "math"

// The modified discrete cosine transform (MDCT) and its inverse, the lapped
// transform of the audio codecs (AAC, Vorbis, Opus CELT, MP3). N is the number of
// samples of a block, always even; an MDCT maps N samples to M = N/2 coefficients
// and an IMDCT maps M coefficients to N samples:
//
//	MDCT   X[k] = Σ_{n=0}^{N-1} x[n] cos(2π/N (n + 1/2 + M/2)(k + 1/2)),  k < M
//	IMDCT  y[n] = Σ_{k=0}^{M-1} X[k] cos(2π/N (n + 1/2 + M/2)(k + 1/2)),  n < N
//
// Both are unscaled and the scale argument of the methods multiplies the result.
// This is the definition of the Wikipedia article "Modified discrete cosine
// transform" (read 2026-11-11), whose N is M here; Princen and Bradley (1986)
// introduced the transform and the time-domain aliasing cancellation (TDAC). The
// ISO/IEC 14496-3 text of the AAC IMDCT was not opened: AAC's 2/N is a scale, which
// the caller passes, and the cosine kernel is the one above.
//
// Perfect reconstruction. With a window w of length N, symmetric and satisfying
// the Princen-Bradley condition w[n]² + w[n+M]² = 1 (SineWindow, KBDWindow),
// blocks that advance by M samples are windowed, transformed, inverse
// transformed, windowed again and overlap-added; the sum equals the input
// wherever two blocks overlap when the product of the two scales is 2/M = 4/N
// (for instance MDCT scale 1 and IMDCT scale 2/M; the article's "2/N" with its N
// the number of coefficients). With the product 1/M the sum is half the input.
//
// Algorithm. For even M the fold of the article, (a, b, c, d) → (−c_r − d, a − b_r)
// for the four quarters of the block (r the reversal), makes the MDCT a DCT-IV of
// length M; this package's DCT-IV is one M/2 = N/4-point complex FFT between a
// pre- and a post-twiddle, so the cost is N/4 points of FFT plus O(N). The IMDCT
// is the same DCT-IV followed by the unfolding (u, v) → (v, −v_r, −u_r, −u).
// For odd M (N = 2 mod 4) the shift n + 1/2 + M/2 is an integer, the DCT-IV form
// does not apply, and the transform is computed by the N-point route: with
// ω = e^{-iπ/M}, X[k] = Re( ω^{n0(k+1/2)} Σ_n (x[n] ω^{n/2}) ω^{nk} ) for
// n0 = (M+1)/2, which is one complex FFT of length N on a plan of this package
// between two twiddles; the IMDCT is the same sum with the roles of n and k
// exchanged, the M coefficients zero-padded to N. That route is about four times
// the cost of the even one and has the same error bound.

// MDCTPlan is a reusable MDCT of N samples to N/2 coefficients, in float64. A
// plan is immutable and safe for concurrent use; the methods allocate nothing in
// steady state.
type MDCTPlan struct {
	n    int
	k    *r2rKernel                 // even N/2: the DCT-IV kernel (N/4-point plan and twiddles)
	fft  *Plan                      // odd N/2: length N
	pre  []complex128               // odd: e^{-iπ n/M}^{1/2}, n < N
	post []complex128               // odd: e^{-iπ n0 (k+1/2)/M}, k < M
	pool scratchCache[[]complex128] // length N/4 (even N/2) or N
}

// mdctTwiddles returns the pre and post twiddles of the odd route; the angles
// are formed from integers, reduced exactly, so each costs one rounding.
func mdctTwiddles(m int) (pre, post []complex128) {
	n := 2 * m
	pre = make([]complex128, n)
	post = make([]complex128, m)
	for i := range pre {
		s, c := math.Sincos(-math.Pi * float64(i) / float64(2*m))
		pre[i] = complex(c, s)
	}
	for k := range post {
		// int64: (m+1)(2k+1) reaches 2·m² and overflows a 32-bit int from m = 32,769
		r := int((int64(m+1) * int64(2*k+1)) % int64(8*m))
		s, c := math.Sincos(-2 * math.Pi * float64(r) / float64(8*m))
		post[k] = complex(c, s)
	}
	return pre, post
}

func checkMDCT(n int) {
	if n < 2 || n%2 != 0 {
		panic("fft: MDCT needs an even block length of at least 2")
	}
}

// NewMDCTPlan returns a plan for blocks of n samples and n/2 coefficients. It
// panics if n is odd or below 2.
func NewMDCTPlan(n int) *MDCTPlan {
	checkMDCT(n)
	p := &MDCTPlan{n: n}
	if m := n / 2; m%2 == 0 {
		p.k = newR2RKernel(true, 4, m)
		p.pool.init(func() *[]complex128 { s := make([]complex128, m/2); return &s })
	} else {
		p.fft = NewPlan(n)
		p.pre, p.post = mdctTwiddles(m)
		p.pool.init(func() *[]complex128 { s := make([]complex128, n); return &s })
	}
	return p
}

// Len reports N, the number of samples of a block; the number of coefficients is
// Len()/2.
func (p *MDCTPlan) Len() int { return p.n }

// MDCT writes scale times the MDCT of the N samples of src into dst[:N/2] and
// returns it. dst must hold at least N/2 values and src at least N. src is
// read completely before dst is written, so dst may overlap src; src is not
// modified otherwise.
func (p *MDCTPlan) MDCT(dst, src []float64, scale float64) []float64 {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < m || len(src) < n {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for i := 0; i < n; i++ {
			z[i] = complex(src[i], 0) * p.pre[i]
		}
		p.fft.FFT(z, z)
		for k := 0; k < m; k++ {
			a, b := p.post[k], z[k]
			dst[k] = scale * (real(a)*real(b) - imag(a)*imag(b))
		}
		p.pool.put(sp)
		return dst[:m]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	// The fold (a, b, c, d) → (−c_r − d, a − b_r) of the article is read in place:
	// the DCT-IV's pre-twiddle pairs the folded values u[2j] and u[m-1-2j]; for
	// j < j1 = ceil(h/2) the first lies in the first half of u (from c and d) and
	// the second in the other (from a and b), for j >= j1 the reverse.
	j1 := (h + 1) / 2
	src = src[:n]
	pre := k.pre[:h]
	for j := 0; j < j1; j++ {
		ua := -(src[m+h-1-2*j] + src[m+h+2*j])
		ub := src[h-1-2*j] - src[h+2*j]
		c[j] = complex(ua, ub) * pre[j]
	}
	for j := j1; j < h; j++ {
		ua := src[2*j-h] - src[3*h-1-2*j]
		ub := -(src[h+2*j] + src[5*h-1-2*j])
		c[j] = complex(ua, ub) * pre[j]
	}
	k.half.FFT(c, c)
	f := scale / 2 // the package's DCT-IV is twice the sum
	dst = dst[:m]
	post := k.post[:h]
	for i := 0; i < h; i++ {
		w := post[i] * c[i]
		dst[2*i] = real(w) * f
		dst[m-1-2*i] = -imag(w) * f
	}
	p.pool.put(sp)
	return dst
}

// IMDCT writes scale times the inverse MDCT of the N/2 coefficients of src into
// dst[:N] and returns it. dst must hold at least N values and src at least N/2.
// src is read completely before dst is written, so dst may overlap src. A scale
// of 2/M, M = N/2, is the one that makes the overlap-add of windowed blocks
// reconstruct the signal when the MDCT scale is 1 (see above).
func (p *MDCTPlan) IMDCT(dst, src []float64, scale float64) []float64 {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < n || len(src) < m {
		panic("fft: MDCTPlan slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for k := 0; k < m; k++ {
			z[k] = complex(src[k], 0) * p.post[k]
		}
		clear(z[m:])
		p.fft.FFT(z, z)
		for i := 0; i < n; i++ {
			a, b := p.pre[i], z[i]
			dst[i] = scale * (real(a)*real(b) - imag(a)*imag(b))
		}
		p.pool.put(sp)
		return dst[:n]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	src = src[:m]
	pre := k.pre[:h]
	for j := 0; j < h; j++ {
		c[j] = complex(src[2*j], src[m-1-2*j]) * pre[j]
	}
	k.half.FFT(c, c)
	// The DCT-IV output u[k] is the unfolded (u, v) → (v, −v_r, −u_r, −u) of the
	// article, written where it belongs: u[k] with k < h goes to dst[m+h-1-k] and
	// dst[m+h+k] negated, u[h+j] to dst[j] and, negated, dst[m-1-j]. The outputs
	// k = 2i and k = m-1-2i of one post-twiddle fall in the first half for
	// i < j1 and the second half for the other, and the reverse.
	f := scale / 2
	dst = dst[:n]
	post := k.post[:h]
	j1 := (h + 1) / 2
	for i := 0; i < j1; i++ {
		w := post[i] * c[i]
		s := real(w) * f  // u[2i], 2i < h
		t := -imag(w) * f // u[m-1-2i], in the second half: index j = h-1-2i
		dst[m+h-1-2*i] = -s
		dst[m+h+2*i] = -s
		dst[h-1-2*i] = t
		dst[m-h+2*i] = -t
	}
	for i := j1; i < h; i++ {
		w := post[i] * c[i]
		t := real(w) * f  // u[2i], 2i >= h: j = 2i-h
		s := -imag(w) * f // u[m-1-2i], m-1-2i < h
		dst[2*i-h] = t
		dst[m-1-2*i+h] = -t
		dst[m+h-1-(m-1-2*i)] = -s
		dst[m+h+(m-1-2*i)] = -s
	}
	p.pool.put(sp)
	return dst
}

// MDCTPlan32 is the single-precision counterpart of MDCTPlan.
type MDCTPlan32 struct {
	n    int
	k    *f32ndR2RKernel // even N/2: the DCT-IV kernel
	fft  *Plan32
	pre  []complex64
	post []complex64
	pool scratchCache[[]complex64] // length N/4 (even N/2) or N
}

// NewMDCTPlan32 returns a single-precision plan for blocks of n samples. It
// panics if n is odd or below 2.
func NewMDCTPlan32(n int) *MDCTPlan32 {
	checkMDCT(n)
	p := &MDCTPlan32{n: n}
	if m := n / 2; m%2 == 0 {
		p.k = f32ndNewR2RKernel(true, 4, m)
		p.pool.init(func() *[]complex64 { s := make([]complex64, m/2); return &s })
	} else {
		p.fft = NewPlan32(n)
		pre, post := mdctTwiddles(m)
		p.pre, p.post = make([]complex64, len(pre)), make([]complex64, len(post))
		for i, v := range pre {
			p.pre[i] = complex64(v)
		}
		for i, v := range post {
			p.post[i] = complex64(v)
		}
		p.pool.init(func() *[]complex64 { s := make([]complex64, n); return &s })
	}
	return p
}

// Len reports N, the number of samples of a block.
func (p *MDCTPlan32) Len() int { return p.n }

// MDCT is MDCTPlan.MDCT on float32 values.
func (p *MDCTPlan32) MDCT(dst, src []float32, scale float32) []float32 {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < m || len(src) < n {
		panic("fft: MDCTPlan32 slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for i := 0; i < n; i++ {
			z[i] = complex(src[i], 0) * p.pre[i]
		}
		p.fft.FFT(z, z)
		for k := 0; k < m; k++ {
			a, b := p.post[k], z[k]
			dst[k] = scale * (real(a)*real(b) - imag(a)*imag(b))
		}
		p.pool.put(sp)
		return dst[:m]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	// The fold is read in place, as in MDCTPlan.MDCT.
	j1 := (h + 1) / 2
	src = src[:n]
	pre := k.pre[:h]
	for j := 0; j < j1; j++ {
		ua := -(src[m+h-1-2*j] + src[m+h+2*j])
		ub := src[h-1-2*j] - src[h+2*j]
		c[j] = f32Mul(complex(ua, ub), pre[j])
	}
	for j := j1; j < h; j++ {
		ua := src[2*j-h] - src[3*h-1-2*j]
		ub := -(src[h+2*j] + src[5*h-1-2*j])
		c[j] = f32Mul(complex(ua, ub), pre[j])
	}
	k.half.execute(c, c, false)
	f := scale / 2
	dst = dst[:m]
	post := k.post[:h]
	for i := 0; i < h; i++ {
		w := f32Mul(post[i], c[i])
		dst[2*i] = real(w) * f
		dst[m-1-2*i] = -imag(w) * f
	}
	p.pool.put(sp)
	return dst
}

// IMDCT is MDCTPlan.IMDCT on float32 values.
func (p *MDCTPlan32) IMDCT(dst, src []float32, scale float32) []float32 {
	n := p.n
	m, h := n/2, n/4
	if len(dst) < n || len(src) < m {
		panic("fft: MDCTPlan32 slice shorter than the plan's length")
	}
	if p.k == nil {
		sp := p.pool.get()
		z := *sp
		for k := 0; k < m; k++ {
			z[k] = complex(src[k], 0) * p.post[k]
		}
		clear(z[m:])
		p.fft.FFT(z, z)
		for i := 0; i < n; i++ {
			a, b := p.pre[i], z[i]
			dst[i] = scale * (real(a)*real(b) - imag(a)*imag(b))
		}
		p.pool.put(sp)
		return dst[:n]
	}
	sp := p.pool.get()
	c := (*sp)[:h]
	k := p.k
	src = src[:m]
	pre := k.pre[:h]
	for j := 0; j < h; j++ {
		c[j] = f32Mul(complex(src[2*j], src[m-1-2*j]), pre[j])
	}
	k.half.execute(c, c, false)
	// The unfolding is written from the post-twiddle, as in MDCTPlan.IMDCT.
	f := scale / 2
	dst = dst[:n]
	post := k.post[:h]
	j1 := (h + 1) / 2
	for i := 0; i < j1; i++ {
		w := f32Mul(post[i], c[i])
		s := real(w) * f
		t := -imag(w) * f
		dst[m+h-1-2*i] = -s
		dst[m+h+2*i] = -s
		dst[h-1-2*i] = t
		dst[m-h+2*i] = -t
	}
	for i := j1; i < h; i++ {
		w := f32Mul(post[i], c[i])
		t := real(w) * f
		s := -imag(w) * f
		dst[2*i-h] = t
		dst[m-1-2*i+h] = -t
		dst[m+h-1-(m-1-2*i)] = -s
		dst[m+h+(m-1-2*i)] = -s
	}
	p.pool.put(sp)
	return dst
}

// SineWindow returns the length-n sine window w[k] = sin(π(k+1/2)/n), which
// satisfies the Princen-Bradley condition w[k]² + w[k+n/2]² = 1 for even n (the
// MDCT block length). n <= 0 returns an empty slice.
func SineWindow(n int) []float64 {
	w := make([]float64, max(n, 0))
	for k := range w {
		w[k] = math.Sin(math.Pi * (float64(k) + 0.5) / float64(n))
	}
	return w
}

// KBDWindow returns the length-n Kaiser-Bessel-derived window of parameter alpha
// (AAC uses 4 for long and 6 for short blocks), n even:
//
//	w[k] = sqrt( Σ_{j<=k} K[j] / Σ_{j<=n/2} K[j] ),  k < n/2,   w[n-1-k] = w[k],
//
// K being the Kaiser window of length n/2+1 with parameter πα. It satisfies the
// Princen-Bradley condition. It panics if n is odd, or if alpha is NaN, infinite
// or |alpha| > 200 (the Bessel function I0(π·alpha) overflows float64 beyond
// 227, and a NaN window would follow); n <= 0 returns an empty slice.
func KBDWindow(n int, alpha float64) []float64 {
	if n%2 != 0 {
		panic("fft: KBDWindow needs an even length")
	}
	if !(math.Abs(alpha) <= kbdMaxAlpha) { // false for NaN
		panic("fft: KBDWindow needs a finite alpha with |alpha| at most 200")
	}
	w := make([]float64, max(n, 0))
	h := n / 2
	if h == 0 {
		return w
	}
	kaiser := make([]float64, h+1)
	total := 0.0
	for j := range kaiser {
		r := 2*float64(j)/float64(h) - 1
		kaiser[j] = besselI0(math.Pi * alpha * math.Sqrt(1-r*r))
		total += kaiser[j]
	}
	acc := 0.0
	for k := 0; k < h; k++ {
		acc += kaiser[k]
		w[k] = math.Sqrt(acc / total)
		w[n-1-k] = w[k]
	}
	return w
}

// kbdMaxAlpha keeps π·alpha at 628, where I0 is about 1e270, so the sum of any
// number of them that fits in memory stays finite (I0 itself overflows at 713.9).
const kbdMaxAlpha = 200

// besselI0 is the modified Bessel function of the first kind of order 0, by its
// power series Σ ((x/2)^j / j!)², which converges for every x and has positive
// terms, so it is accurate to a few ulps.
func besselI0(x float64) float64 {
	q := x * x / 4
	sum, term := 1.0, 1.0
	for j := 1; term > 1e-17*sum; j++ {
		term *= q / float64(j*j)
		sum += term
	}
	return sum
}
