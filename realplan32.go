package fft

import (
	"sync"
	"unsafe"
)

// A RealPlan32 is the single-precision counterpart of RealPlan: a reusable
// real-input transform of a fixed length N over float32 samples and complex64
// bins. For even N the samples are packed into an N/2-point Plan32 and
// untangled with a table computed in float64 and rounded once; for odd N a
// full-length Plan32 is used. RealPlan32s are immutable after construction
// and safe for concurrent use.
type RealPlan32 struct {
	n int

	half    *Plan32     // length n/2 (even path), nil for odd n
	tw      []complex64 // exp(-2πi·k/n), k = 0 .. n/2 (even path)
	scratch *sync.Pool  // one m-point buffer per concurrent call (even path)

	full *Plan32 // length n (odd path and n == 1)
}

// NewRealPlan32 returns a single-precision real-input plan for length n
// (n >= 0).
func NewRealPlan32(n int) *RealPlan32 {
	p := &RealPlan32{n: n}
	if n <= 1 || n%2 != 0 {
		if n >= 1 {
			p.full = cachedPlan32(n)
		}
		return p
	}
	m := n / 2
	p.half = cachedPlan32(m)
	tw := make([]complex128, m+1)
	full := twiddleTable(n)
	copy(tw, full[:m+1])
	p.tw = toC64(tw)
	p.scratch = &sync.Pool{New: func() any { s := make([]complex64, m); return &s }}
	return p
}

// Len reports the real-input length the plan was built for.
func (p *RealPlan32) Len() int { return p.n }

// RFFT writes the N/2+1 non-redundant bins of the real signal src into dst,
// unnormalized, and returns dst[:N/2+1]. src must have at least Len()
// elements and dst at least Len()/2+1. src is not modified.
func (p *RealPlan32) RFFT(dst []complex64, src []float32) []complex64 {
	return p.RFFTNorm(dst, src, NormBackward)
}

// IRFFT writes the real signal of length Len() reconstructed from the half
// spectrum src into dst, normalized by N, and returns dst[:N]. src is read up
// to min(len(src), Len()/2+1) bins; missing bins count as zero, and the
// imaginary parts of the DC (and, for even N, Nyquist) bins are ignored. src
// is not modified.
func (p *RealPlan32) IRFFT(dst []float32, src []complex64) []float32 {
	return p.IRFFTNorm(dst, src, NormBackward)
}

// RFFTNorm is RFFT scaled as norm says (1, 1/sqrt(N) or 1/N). It panics on a
// Norm that is not one of the three constants.
func (p *RealPlan32) RFFTNorm(dst []complex64, src []float32, norm Norm) []complex64 {
	n := p.n
	s := norm.scale(n, false)
	if len(src) < n || (n > 0 && len(dst) < n/2+1) {
		panic("fft: RealPlan32 slice shorter than the plan's length")
	}
	if n == 0 {
		return dst[:0]
	}
	h := n/2 + 1
	if p.half == nil {
		c := make([]complex64, n)
		for i, v := range src[:n] {
			c[i] = complex(v, 0)
		}
		p.full.execute(c, c, false)
		copy(dst, c[:h])
	} else {
		m := n / 2
		bp := p.scratch.Get().(*[]complex64)
		Z := *bp
		// z[j] = src[2j] + i·src[2j+1] is the memory layout of a []complex64.
		p.half.execute(Z, asComplex64(src[:n]), false)
		f32Untangle(dst, Z, p.tw, m)
		p.scratch.Put(bp)
	}
	scale32(dst[:h], s)
	return dst[:h]
}

// IRFFTNorm is IRFFT scaled as norm says (1/N, 1/sqrt(N) or 1). It panics on
// a Norm that is not one of the three constants.
func (p *RealPlan32) IRFFTNorm(dst []float32, src []complex64, norm Norm) []float32 {
	n := p.n
	s := norm.scale(n, true)
	if len(dst) < n {
		panic("fft: RealPlan32 slice shorter than the plan's length")
	}
	if n == 0 {
		return dst[:0]
	}
	bin := func(k int) complex64 {
		if k < len(src) {
			return src[k]
		}
		return 0
	}
	if p.half == nil {
		// Odd n (or 1): the full Hermitian spectrum through one inverse.
		full := make([]complex64, n)
		for k := 0; k <= n/2; k++ {
			full[k] = bin(k)
			if k > 0 {
				full[n-k] = complex(real(full[k]), -imag(full[k]))
			}
		}
		p.full.execute(full, full, true)
		f := float32(s)
		for i, v := range full {
			dst[i] = real(v) * f
		}
		return dst[:n]
	}
	// Even n = 2m: rebuild the packed m-point spectrum (the inverse of
	// f32Untangle), with the normalization folded in. The unnormalized m-point
	// inverse of the packed spectrum is m·z, so the factor that turns it into
	// s·n·x is 2s, which is h = 2s·0.5 = s in the pair formula.
	m := n / 2
	h := float32(s)
	bp := p.scratch.Get().(*[]complex64)
	Z := *bp
	x0, xm := real(bin(0)), real(bin(m))
	Z[0] = complex((x0+xm)*h, (x0-xm)*h)
	k := 1
	for ; k < m-k; k++ {
		Z[k], Z[m-k] = f32Retangle(bin(k), bin(m-k), p.tw[k], h)
	}
	if k == m-k {
		// Self-paired middle bin: one index, written once.
		Z[k], _ = f32Retangle(bin(k), bin(k), p.tw[k], h)
	}
	// z[j] = x[2j] + i·x[2j+1] is the memory layout of dst read as complex64.
	p.half.execute(asComplex64(dst[:n]), Z, true)
	p.scratch.Put(bp)
	return dst[:n]
}

// f32Untangle is rfftUntangle in single precision: it splits the m-point
// packed spectrum Z into the m+1 real-FFT bins dst[0..m].
func f32Untangle(dst, Z, tw []complex64, m int) {
	z0r, z0i := real(Z[0]), imag(Z[0])
	dst[0] = complex(z0r+z0i, 0)
	dst[m] = complex(z0r-z0i, 0)
	for k := 1; k <= (m-1)/2; k++ {
		zk, zmk := Z[k], Z[m-k]
		xer := (real(zk) + real(zmk)) * 0.5
		xei := (imag(zk) - imag(zmk)) * 0.5
		xor := (imag(zk) + imag(zmk)) * 0.5
		xoi := -(real(zk) - real(zmk)) * 0.5
		wr, wi := real(tw[k]), imag(tw[k])
		tr := wr*xor - wi*xoi
		ti := wr*xoi + wi*xor
		dst[k] = complex(xer+tr, xei+ti)
		dst[m-k] = complex(xer-tr, -(xei - ti))
	}
	if m&1 == 0 {
		zk := Z[m/2]
		dst[m/2] = complex(real(zk), -imag(zk))
	}
}

// f32Retangle is retangle in single precision: given X[k] and X[m-k] it
// returns Z[k] and Z[m-k], each scaled by 2h.
func f32Retangle(xk, xmk, w complex64, h float32) (zk, zmk complex64) {
	xer := (real(xk) + real(xmk)) * h
	xei := (imag(xk) - imag(xmk)) * h
	dr := (real(xk) - real(xmk)) * h
	di := (imag(xk) + imag(xmk)) * h
	wr, wi := real(w), imag(w)
	xor := wr*dr + wi*di
	xoi := wr*di - wi*dr
	return complex(xer-xoi, xei+xor), complex(xer+xoi, -(xei - xor))
}

// asComplex64 views the 2m float32s of f as m complex64 values: a complex64
// is its real then its imaginary float32 (the layout TestAsComplex64Layout
// pins). len(f) must be even and non-zero.
func asComplex64(f []float32) []complex64 {
	return unsafe.Slice((*complex64)(unsafe.Pointer(&f[0])), len(f)/2)
}

// RFFT32 returns the N/2+1 non-redundant bins of the real signal x in single
// precision, unnormalized like RFFT. Empty input returns an empty (non-nil)
// slice. Plans are cached per length.
func RFFT32(x []float32) []complex64 {
	n := len(x)
	if n == 0 {
		return []complex64{}
	}
	return cachedRealPlan32(n).RFFT(make([]complex64, n/2+1), x)
}

// IRFFT32 inverts RFFT32, returning n real samples normalized by n like
// IRFFT. n <= 0 returns an empty (non-nil) slice. n sets the allocation:
// bound it when it comes from untrusted input (see SECURITY.md).
func IRFFT32(spectrum []complex64, n int) []float32 {
	if n <= 0 {
		return []float32{}
	}
	return cachedRealPlan32(n).IRFFT(make([]float32, n), spectrum)
}

var (
	realPlan32Mu    sync.Mutex
	realPlan32Cache = map[int]*RealPlan32{}
)

// cachedRealPlan32 returns a shared RealPlan32 for length n.
func cachedRealPlan32(n int) *RealPlan32 {
	realPlan32Mu.Lock()
	defer realPlan32Mu.Unlock()
	p, ok := realPlan32Cache[n]
	if !ok {
		p = NewRealPlan32(n)
		realPlan32Cache[n] = p
	}
	return p
}
