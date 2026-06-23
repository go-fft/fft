package fft

import (
	"math"
	"sync"
)

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

	// scratch pools a single 2·m complex128 buffer per concurrent caller so the
	// pack/untangle (and the inverse's reverse-untangle/unpack) reuse a private
	// working set instead of allocating two m-length buffers every call. The plan
	// stays immutable and concurrent-safe: each Get hands out a buffer owned by
	// that call for its duration. nil for the odd path.
	scratch *sync.Pool

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
	p.scratch = &sync.Pool{New: func() any {
		s := make([]complex128, 2*m)
		return &s
	}}
	return p
}

// getScratch borrows the plan's pooled 2·m working buffer, split into two
// m-length halves (z for packing, Z for the spectrum). putScratch returns it.
func (p *RealPlan) getScratch() (buf *[]complex128, z, Z []complex128) {
	m := p.n / 2
	buf = p.scratch.Get().(*[]complex128)
	s := *buf
	return buf, s[:m:m], s[m : 2*m : 2*m]
}

func (p *RealPlan) putScratch(buf *[]complex128) { p.scratch.Put(buf) }

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
	buf, z, Z := p.getScratch()
	for j := 0; j < m; j++ {
		z[j] = complex(src[2*j], src[2*j+1])
	}
	if p.half.it != nil {
		// m is a power of two: feed the freshly packed buffer straight to the
		// iterative kernel's scratch entry point, skipping the engine's internal
		// gather buffer (and its allocation). z is private to this call, so the
		// permutation consuming it is safe.
		p.half.it.transformScratch(Z, z, false)
	} else {
		p.half.FFT(Z, z)
	}
	rfftUntangle(dst, Z, p.tw, m)
	p.putScratch(buf)
	return dst[:m+1]
}

// rfftUntangle splits the m-point packed spectrum Z into the m+1 real-FFT bins
// dst[0..m], using the precomputed recombination twiddles tw (tw[k]=W_n^k for
// k=0..m). The k=0 and k=m bins are purely real and handled outside the loop.
// The interior bins are produced in conjugate pairs (k, m-k): both read the same
// Z[k], Z[m-k] and the same twiddle W_n^k, and satisfy
//
//	dst[k]   = xe + W_n^k·xo,
//	dst[m-k] = conj(xe − W_n^k·xo),
//
// because xe(m-k)=conj(xe(k)), xo(m-k)=conj(xo(k)), and W_n^{m-k}=−conj(W_n^k)
// (W_n^m = −1 for n = 2m). Computing the pair together halves the Z reads and
// twiddle lookups of a per-bin loop. The even/odd split and twiddle
// recombination are expanded into real arithmetic to avoid per-bin
// complex-multiply helper calls.
//
// The interior loop runs over k = 1 .. (m-1)/2 with NO data-dependent branch in
// the body (the gc autovectorizer bails on a loop carrying a branch), writing
// both dst[k] and dst[m-k] every iteration. The self-paired middle bin k = m/2
// (present only when m is even, where m-k == k) is finished once after the loop
// from the same formula, so it is never written twice.
func rfftUntangle(dst, Z, tw []complex128, m int) {
	z0r, z0i := real(Z[0]), imag(Z[0])
	dst[0] = complex(z0r+z0i, 0)
	dst[m] = complex(z0r-z0i, 0)
	half := (m - 1) / 2
	for k := 1; k <= half; k++ {
		zk := Z[k]
		zmk := Z[m-k]
		// xe = (zk + conj(zmk))/2, xo = (zk - conj(zmk))·(-i/2).
		xer := (real(zk) + real(zmk)) * 0.5
		xei := (imag(zk) - imag(zmk)) * 0.5
		// (zk - conj(zmk)) = (real(zk)-real(zmk)) + i(imag(zk)+imag(zmk));
		// times -i/2 swaps and scales: xo = (imag-sum)/2 - i(real-diff)/2.
		xor := (imag(zk) + imag(zmk)) * 0.5
		xoi := -(real(zk) - real(zmk)) * 0.5
		// t = W_n^k · xo.
		wr, wi := real(tw[k]), imag(tw[k])
		tr := wr*xor - wi*xoi
		ti := wr*xoi + wi*xor
		// dst[k] = xe + t; dst[m-k] = conj(xe - t).
		dst[k] = complex(xer+tr, xei+ti)
		dst[m-k] = complex(xer-tr, -(xei - ti))
	}
	if m&1 == 0 {
		// Self-paired middle bin k = m/2: here zmk == zk, so xe = real(zk), the
		// odd part vanishes (xor = imag(zk), times W^{m/2} = -i gives a real
		// contribution that the conjugate-pair formula reduces to conj(zk)).
		k := m / 2
		zk := Z[k]
		dst[k] = complex(real(zk), -imag(zk))
	}
}

// IRFFT writes the real signal of length Len() reconstructed from the half
// spectrum src into dst and returns dst. dst must have length Len(); src is
// read up to min(len(src), Len()/2+1) bins. The result is normalized by N. src
// is not modified.
//
// For even N the inverse mirrors the forward packing: the half spectrum is
// pre-processed into an N/2-point complex spectrum, run through one N/2-point
// inverse complex FFT, and unpacked — half the arithmetic and memory traffic of
// promoting to a full conjugate-symmetric length-N inverse transform. For odd N
// (and the trivial N<=1) it falls back to the full conjugate-mirror inverse.
func (p *RealPlan) IRFFT(dst []float64, src []complex128) []float64 {
	n := p.n
	if n <= 0 {
		return dst[:0]
	}
	if p.half == nil {
		// Odd length (and the trivial n==1, whose plan has no half): full
		// conjugate-symmetric inverse of size n.
		return p.irfftFull(dst, src)
	}
	return p.irfftPacked(dst, src)
}

// irfftFull is the size-n conjugate-mirror inverse used for odd lengths (which
// cannot be packed into a half-length transform). It builds the full Hermitian
// spectrum and runs one length-n inverse complex FFT.
func (p *RealPlan) irfftFull(dst []float64, src []complex128) []float64 {
	n := p.n
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

// irfftPacked is the half-length (even-n) inverse. It reverses the forward
// untangle to recover the N/2-point packed spectrum Z[k], runs one N/2-point
// inverse complex FFT (normalized by m = N/2), and unpacks z[j] into the two
// real samples src[2j], src[2j+1] it carried. Missing input bins (len(src) <
// N/2+1) and the imaginary parts of the DC/Nyquist bins are treated as zero,
// matching the full conjugate-mirror inverse for a valid Hermitian spectrum.
func (p *RealPlan) irfftPacked(dst []float64, src []complex128) []float64 {
	n := p.n
	m := n / 2

	// bin returns src[k] (the kept half spectrum), or 0 past its end so a short
	// or partially-specified spectrum behaves exactly like the zero-filled
	// conjugate-mirror inverse.
	bin := func(k int) complex128 {
		if k < len(src) {
			return src[k]
		}
		return 0
	}

	// Build the packed N/2-point spectrum Z. The forward produced, for each pair
	// (k, m-k) with k in 1..m-1:
	//   xe = (Z[k] + conj(Z[m-k]))/2,  xo = (Z[k] - conj(Z[m-k]))·(-i/2)
	//   X[k]   = xe + W_n^k·xo,        X[m-k] = conj(xe - W_n^k·xo).
	// Inverting that pair:
	//   xe = (X[k] + conj(X[m-k]))/2,  W_n^k·xo = (X[k] - conj(X[m-k]))/2,
	//   xo = conj(W_n^k)·(X[k] - conj(X[m-k]))/2,
	//   Z[k]   = xe + i·xo,            Z[m-k] = conj(xe - i·xo).
	// The k=0 / k=m DC and Nyquist bins are purely real for a real signal and map
	// to Z[0] = (X[0]+X[m]) + i·(X[0]-X[m]) (the inverse of X[0]=Z0.r+Z0.i,
	// X[m]=Z0.r-Z0.i); only the real parts are used, discarding any imaginary
	// component just as the full inverse's real() projection does.
	buf, z, Z := p.getScratch()
	x0 := real(bin(0))
	xm := real(bin(m))
	Z[0] = complex((x0+xm)*0.5, (x0-xm)*0.5)
	for k := 1; k <= m-k; k++ {
		xk := bin(k)
		xmk := bin(m - k)
		cmk := complexConj(xmk) // conj(X[m-k])
		// xe = (X[k] + conj(X[m-k]))/2.
		xer := (real(xk) + real(cmk)) * 0.5
		xei := (imag(xk) + imag(cmk)) * 0.5
		// d = (X[k] - conj(X[m-k]))/2  (= W_n^k·xo).
		dr := (real(xk) - real(cmk)) * 0.5
		di := (imag(xk) - imag(cmk)) * 0.5
		// xo = conj(W_n^k)·d = (wr - i·wi)·(dr + i·di).
		wr, wi := real(p.tw[k]), imag(p.tw[k])
		xor := wr*dr + wi*di
		xoi := wr*di - wi*dr
		// Z[k] = xe + i·xo = (xer - xoi) + i·(xei + xor).
		Z[k] = complex(xer-xoi, xei+xor)
		// Z[m-k] = conj(xe - i·xo) = (xer - xoi) - i·... wait, compute directly:
		// xe - i·xo = (xer + xoi) + i·(xei - xor); its conjugate is the next line.
		if k != m-k {
			Z[m-k] = complex(xer+xoi, -(xei - xor))
		}
	}

	// One m-point inverse complex FFT (normalized by m), then unpack:
	// z[j] = x[2j] + i·x[2j+1]. z is the pool buffer's second half (borrowed
	// alongside Z above), private to this call.
	if p.half.it != nil {
		// m is a power of two: run the unnormalized inverse on a private scratch
		// buffer (Z is private here, safe to consume) and normalize on unpack.
		p.half.it.transformScratch(z, Z, true)
		inv := 1 / float64(m)
		for j := 0; j < m; j++ {
			dst[2*j] = real(z[j]) * inv
			dst[2*j+1] = imag(z[j]) * inv
		}
		p.putScratch(buf)
		return dst
	}
	p.half.IFFT(z, Z) // already normalized by m
	for j := 0; j < m; j++ {
		dst[2*j] = real(z[j])
		dst[2*j+1] = imag(z[j])
	}
	p.putScratch(buf)
	return dst
}
