package fft

// This file implements the discrete sine transforms of types I–IV, as
// scipy.fft.dst/idst define them (scipy 1.18, orthogonalize at its default):
//
//	DST-I   y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(k+1)(n+1)/(N+1))                    logical size 2(N+1)
//	DST-II  y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(k+1)(2n+1)/(2N))                    logical size 2N
//	DST-III y[k] = (-1)^k x[N-1] + 2 Σ_{n=0}^{N-2} x[n] sin(π(2k+1)(n+1)/(2N))    logical size 2N
//	DST-IV  y[k] = 2 Σ_{n=0}^{N-1} x[n] sin(π(2k+1)(2n+1)/(4N))                   logical size 2N
//
// FFTW's RODFT00, RODFT10, RODFT01 and RODFT11 (FFTW manual §4.8.4). Under
// NormOrtho, DST-II divides y[N-1] by √2 and DST-III multiplies x[N-1] by √2;
// DST-I and DST-IV need no correction.
//
// DST-I runs the real FFT of the odd extension (r2rKernel.dst1). Types II–IV
// reuse the DCT kernels through the classical index identities, each checked
// by substituting n → N-1-n (or k → N-1-k) in the definitions:
//
//	DST-II(x)[k]  = DCT-II((-1)^n x[n])[N-1-k]
//	DST-III(x)[k] = (-1)^k DCT-III(x[N-1-n])[k]
//	DST-IV(x)[k]  = (-1)^k DCT-IV(x[N-1-n])[k]
//
// (the same reductions pocketfft uses). The ortho corrections then fall on
// the DCT's own: DST-II's y[N-1] is DCT-II's y[0], DST-III's x[N-1] is
// DCT-III's x[0].

// A DSTPlan is a reusable discrete sine transform of a fixed length and type
// (1 to 4), matching scipy.fft.dst and scipy.fft.idst. It is immutable and
// safe for concurrent use.
type DSTPlan struct {
	typ int
	k   *r2rKernel
}

// NewDSTPlan returns a plan for the DST of type typ (1, 2, 3 or 4) of length
// n >= 1. It panics if typ is not 1..4 or n < 1.
func NewDSTPlan(n, typ int) *DSTPlan {
	checkR2R(false, typ, n, NormBackward)
	return &DSTPlan{typ: typ, k: newR2RKernel(false, typ, n)}
}

// Len reports the transform length the plan was built for.
func (p *DSTPlan) Len() int { return p.k.n }

// Type reports the DST type (1 to 4) the plan was built for.
func (p *DSTPlan) Type() int { return p.typ }

// DST writes the DST of src into dst and returns dst[:Len()], as
// scipy.fft.dst(src, type=Type(), norm=norm). dst and src must have at least
// Len() elements; dst may alias src. src is not modified unless it aliases
// dst.
func (p *DSTPlan) DST(dst, src []float64, norm Norm) []float64 {
	return p.run(dst, src, norm, false)
}

// IDST writes the inverse DST of src into dst and returns dst[:Len()], as
// scipy.fft.idst(src, type=Type(), norm=norm): IDST(DST(x, norm), norm) == x
// for every norm. The same slice rules as DST apply.
func (p *DSTPlan) IDST(dst, src []float64, norm Norm) []float64 {
	return p.run(dst, src, norm, true)
}

func (p *DSTPlan) run(dst, src []float64, norm Norm, inverse bool) []float64 {
	k := p.k
	n := k.n
	checkR2R(false, p.typ, n, norm)
	if len(dst) < n || len(src) < n {
		panic("fft: DSTPlan slice shorter than the plan's length")
	}
	typ := p.typ
	if inverse {
		typ = inverseType(typ)
	}
	f := norm.scale(logicalSize(false, typ, n), inverse)
	ortho := norm == NormOrtho
	s := k.pool.Get().(*r2rScratch)
	t := s.t
	alternate := false
	switch typ {
	case 1:
		k.dst1(dst, src, s)
	case 2:
		for i := 0; i < n; i++ {
			if i%2 == 0 {
				t[i] = src[i]
			} else {
				t[i] = -src[i]
			}
		}
		k.mk.dct2(dst, t, s.r, s.c)
		for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
			dst[i], dst[j] = dst[j], dst[i]
		}
		if ortho {
			dst[n-1] /= sqrt2
		}
	case 3:
		reverseInto(t, src[:n])
		a0 := 1.0
		if ortho {
			a0 = sqrt2
		}
		k.mk.dct3(dst, t, a0, s.r, s.c)
		alternate = true
	default:
		reverseInto(t, src[:n])
		k.dct4(dst, t, s)
		alternate = true
	}
	k.pool.Put(s)
	scaleReal(dst[:n], f, alternate)
	return dst[:n]
}

// reverseInto writes src in reverse order into dst (len(dst) >= len(src)).
func reverseInto(dst, src []float64) {
	n := len(src)
	for i, v := range src {
		dst[n-1-i] = v
	}
}

// cachedDSTPlan returns the shared plan of type typ and length n, building it
// on first use. Arguments are validated by the caller.
func cachedDSTPlan(n, typ int) *DSTPlan {
	r2rMu.Lock()
	defer r2rMu.Unlock()
	key := r2rKey{false, typ, n}
	p, ok := r2rCache[key].(*DSTPlan)
	if !ok {
		p = NewDSTPlan(n, typ)
		r2rCache[key] = p
	}
	return p
}

// DST returns the discrete sine transform of type typ (1, 2, 3 or 4) of x,
// scaled by norm, as scipy.fft.dst(x, type=typ, norm=norm); NormOrtho also
// applies scipy's default orthogonalisation. x is not modified.
//
// It panics if typ is not 1..4, if x is empty or if norm is unknown.
func DST(x []float64, typ int, norm Norm) []float64 {
	checkR2R(false, typ, len(x), norm)
	return cachedDSTPlan(len(x), typ).DST(make([]float64, len(x)), x, norm)
}

// IDST returns the inverse discrete sine transform of type typ of x, as
// scipy.fft.idst(x, type=typ, norm=norm), so that IDST(DST(x, t, m), t, m)
// reproduces x. The same panics as DST apply.
func IDST(x []float64, typ int, norm Norm) []float64 {
	checkR2R(false, typ, len(x), norm)
	return cachedDSTPlan(len(x), typ).IDST(make([]float64, len(x)), x, norm)
}
