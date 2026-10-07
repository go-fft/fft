package fft

import (
	"math"
	"sync"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
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
// Len()/2+1. src is not modified. dst must not overlap src (which only an
// unsafe conversion can arrange): dst holds the intermediate half-length
// spectrum while src is still being read.
func (p *RealPlan) RFFT(dst []complex128, src []float64) []complex128 {
	n := p.n
	if len(src) < n || (n > 0 && len(dst) < n/2+1) {
		panic("fft: RealPlan slice shorter than the plan's length")
	}
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
	if p.half.it != nil {
		// m is a power of two: fuse the real-pair packing into the iterative
		// kernel's bit-reversal gather (transformRealPacked reads src directly into
		// the bit-reversed Z), removing the separate pack pass and the z buffer
		// write/read entirely — one gather instead of pack + gather.
		buf, _, Z := p.getScratch()
		p.half.it.transformRealPacked(Z, src)
		rfftUntangle(dst, Z, p.tw, m)
		p.putScratch(buf)
		return dst[:m+1]
	}
	// The packing z[j] = src[2j] + i·src[2j+1] is exactly the memory layout of
	// a []complex128, so the half-length FFT reads src through a view of it:
	// no pack pass. It writes the packed spectrum Z straight into dst[:m],
	// which the untangle then turns into the bins in place (each conjugate
	// pair is read before it is written): no buffer of the RealPlan's own, one
	// sync.Pool round trip fewer per call (Round 30: 1.04-1.07× at 256 and
	// 512 points on Zen 3, 1.03× at 256 on Neoverse-N1).
	Z := dst[:m]
	p.half.execute(Z, asComplex(src[:2*m]), false)
	rfftUntangle(dst, Z, p.tw, m)
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
// the body (a branch per bin only costs; gc does not vectorize either form),
// writing
// both dst[k] and dst[m-k] every iteration. The self-paired middle bin k = m/2
// (present only when m is even, where m-k == k) is finished once after the loop
// from the same formula, so it is never written twice.
func rfftUntangle(dst, Z, tw []complex128, m int) {
	z0r, z0i := real(Z[0]), imag(Z[0])
	dst[0] = complex(z0r+z0i, 0)
	dst[m] = complex(z0r-z0i, 0)
	half := (m - 1) / 2
	// An AVX2 (amd64) or NEON (arm64) kernel produces the leading bins,
	// bit-identically.
	for k := kernels.Untangle(dst, Z, tw, m) + 1; k <= half; k++ {
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
	if len(dst) < n {
		panic("fft: RealPlan slice shorter than the plan's length")
	}
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

	if smallIRFFTInPlace(p.half) {
		// Z built in dst, read as complex128 (the layout the inverse writes
		// anyway), and the half-length inverse run in place on it: one
		// sync.Pool round trip fewer (Round 30).
		Z := asComplex(dst[:2*m])
		p.packSpectrum(Z, src, 0.5*(1/float64(m)))
		p.half.execute(Z, Z, true)
		return dst
	}
	buf, z, Z := p.getScratch()
	scale := 1.0
	if p.half.it == nil {
		scale = 1 / float64(m)
	}
	p.packSpectrum(Z, src, 0.5*scale)
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
	// z[j] = x[2j] + i·x[2j+1] is the memory layout of dst read as complex128,
	// so the (already scaled) inverse writes the real signal in place: no
	// unpack pass.
	p.half.execute(asComplex(dst[:2*m]), Z, true)
	p.putScratch(buf)
	return dst
}

// irfftRetangle rebuilds the packed spectrum Z[1..m-1] from the complete half
// spectrum X[0..m] (len(X) == m+1), pair by pair; h = 0.5·scale.
func irfftRetangle(Z, X, tw []complex128, m int, h float64) {
	X = X[:m+1]
	tw = tw[:m+1]
	// An AVX2 (amd64) or NEON (arm64) kernel produces the leading pairs,
	// bit-identically.
	k := kernels.Retangle(Z, X, tw, m, h) + 1
	for ; k < m-k; k++ {
		Z[k], Z[m-k] = retangle(X[k], X[m-k], tw[k], h)
	}
	if k == m-k {
		// Self-paired middle bin: one index, written once.
		Z[k], _ = retangle(X[k], X[k], tw[k], h)
	}
}

// retangle inverts the forward untangle for the conjugate pair (k, m-k): given
// X[k] = xk and X[m-k] = xmk it returns Z[k] and Z[m-k], each scaled by 2h. The forward produced, for each pair (k, m-k) with k in 1..m-1:
//
//	xe = (Z[k] + conj(Z[m-k]))/2,  xo = (Z[k] - conj(Z[m-k]))·(-i/2)
//	X[k]   = xe + W_n^k·xo,        X[m-k] = conj(xe - W_n^k·xo).
//
// Inverting that pair:
//
//	xe = (X[k] + conj(X[m-k]))/2,  W_n^k·xo = (X[k] - conj(X[m-k]))/2,
//	xo = conj(W_n^k)·(X[k] - conj(X[m-k]))/2,
//	Z[k]   = xe + i·xo,            Z[m-k] = conj(xe - i·xo).
func retangle(xk, xmk, w complex128, h float64) (zk, zmk complex128) {
	// xe = (X[k] + conj(X[m-k]))·h.
	xer := (real(xk) + real(xmk)) * h
	xei := (imag(xk) - imag(xmk)) * h
	// d = (X[k] - conj(X[m-k]))·h  (= W_n^k·xo).
	dr := (real(xk) - real(xmk)) * h
	di := (imag(xk) + imag(xmk)) * h
	// xo = conj(W_n^k)·d = (wr - i·wi)·(dr + i·di).
	wr, wi := real(w), imag(w)
	xor := wr*dr + wi*di
	xoi := wr*di - wi*dr
	return complex(xer-xoi, xei+xor), complex(xer+xoi, -(xei - xor))
}

// asComplex views the 2m float64s of f as m complex128 values. A complex128 is
// laid out as its real then its imaginary float64 (the layout gc uses and that
// TestAsComplexLayout pins), so the view is exactly the real-pair packing
// z[j] = f[2j] + i·f[2j+1], at no cost. len(f) must be even.
func asComplex(f []float64) []complex128 {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*complex128)(unsafe.Pointer(&f[0])), len(f)/2)
}

// smallIRFFTInPlace reports whether the inverse of a RealPlan whose half
// plan is half rebuilds the packed spectrum in dst and transforms it there
// (irfftInPlace) rather than in a pooled buffer: wherever the half plan runs
// the Stockham passes in place at no extra cost. The iterative pow2 kernel
// normalizes on unpack and keeps the buffer; an odd pass count in place
// copies the input first where the scratch slides off dst's sets (takesGap:
// IRFFT 4096 lost 5% that way on Neoverse-N1), and the blocked schedule was
// not measured in place, so both keep it too.
func smallIRFFTInPlace(half *Plan) bool {
	sk := half.sk
	if sk == nil || sk.cascT > 0 {
		return false
	}
	return len(sk.stages)%2 == 0 || !takesGap(sk.n)
}

// packSpectrum builds the packed N/2-point spectrum Z from the half spectrum
// src, scaled by 2h (h = 0.5·scale: the inverse's 1/m normalization is
// folded in here, so the inverse FFT runs unnormalized and no separate
// scaling pass is needed). The forward produced, for each pair (k, m-k) with
// k in 1..m-1:
//
//	xe = (Z[k] + conj(Z[m-k]))/2,  xo = (Z[k] - conj(Z[m-k]))·(-i/2)
//	X[k]   = xe + W_n^k·xo,        X[m-k] = conj(xe - W_n^k·xo).
//
// Inverting that pair:
//
//	xe = (X[k] + conj(X[m-k]))/2,  W_n^k·xo = (X[k] - conj(X[m-k]))/2,
//	xo = conj(W_n^k)·(X[k] - conj(X[m-k]))/2,
//	Z[k]   = xe + i·xo,            Z[m-k] = conj(xe - i·xo).
//
// The k=0 / k=m DC and Nyquist bins are purely real for a real signal and map
// to Z[0] = (X[0]+X[m]) + i·(X[0]-X[m]) (the inverse of X[0]=Z0.r+Z0.i,
// X[m]=Z0.r-Z0.i); only the real parts are used, discarding any imaginary
// component just as the full inverse's real() projection does. Missing input
// bins (len(src) < N/2+1) count as zero, so a short or partially-specified
// spectrum behaves exactly like the zero-filled conjugate-mirror inverse.
func (p *RealPlan) packSpectrum(Z, src []complex128, h float64) {
	m := p.n / 2
	bin := func(k int) complex128 {
		if k < len(src) {
			return src[k]
		}
		return 0
	}
	x0 := real(bin(0))
	xm := real(bin(m))
	Z[0] = complex((x0+xm)*h, (x0-xm)*h)
	if len(src) >= m+1 {
		// Complete spectrum (the usual case): no per-bin bounds test.
		irfftRetangle(Z, src[:m+1], p.tw, m, h)
		return
	}
	for k := 1; k <= m-k; k++ {
		zk, zmk := retangle(bin(k), bin(m-k), p.tw[k], h)
		Z[k] = zk
		if k != m-k {
			Z[m-k] = zmk
		}
	}
}
