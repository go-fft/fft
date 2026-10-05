package fft

import (
	"math"
	"sync"
)

// This file holds the discrete cosine and sine transforms of types I–IV in
// single precision: DCT32/IDCT32/DST32/IDST32, their N-D forms and their
// plans, as scipy.fft.dct/idct/dst/idst compute them for float32 input (scipy
// keeps float32: a float32 array in, a float32 array out). The definitions,
// the norms, the ortho corrections and the algorithms are those of dct.go and
// dst.go, run on the single-precision FFTs (RealPlan32, Plan32); the
// validation (checkR2R) and the size bookkeeping (logicalSize, inverseType)
// are shared with the float64 path. Every twiddle — Makhoul's exp(-iπk/(2N)),
// the DCT-IV pre- and post-twiddles — is computed in float64 and rounded once
// to float32, as the 1-D single-precision plans do.
//
// Where the float64 code scales by N before an inverse real FFT that divides
// by N (Makhoul's DCT-III), this code runs the inverse unscaled instead: in
// float32 each of those two scalings would round.

// f32ndSqrt2 is √2 rounded to float32, the ortho correction factor.
const f32ndSqrt2 = float32(sqrt2)

// f32ndMakhoul holds the tables of Makhoul's N-point DCT-II/DCT-III in single
// precision; see makhoul.
type f32ndMakhoul struct {
	n  int
	rp *RealPlan32
	w  []complex64 // exp(-iπk/(2N)), k = 0 .. N/2
}

func f32ndNewMakhoul(n int) f32ndMakhoul {
	w := make([]complex64, n/2+1)
	for k := range w {
		s, c := math.Sincos(-math.Pi * float64(k) / float64(2*n))
		w[k] = complex64(complex(c, s))
	}
	return f32ndMakhoul{n: n, rp: cachedRealPlan32(n), w: w}
}

// dct2 is makhoul.dct2 in single precision: the unnormalised DCT-II of
// src[:n] into dst[:n] (dst may alias src), v and V scratch.
func (m f32ndMakhoul) dct2(dst, src, v []float32, V []complex64) {
	n := m.n
	for i := 0; 2*i < n; i++ {
		v[i] = src[2*i]
	}
	for i := 0; 2*i+1 < n; i++ {
		v[n-1-i] = src[2*i+1]
	}
	V = m.rp.RFFT(V, v)
	dst[0] = 2 * real(V[0])
	for k := 1; k < (n+1)/2; k++ {
		z := f32Mul(m.w[k], V[k])
		dst[k] = 2 * real(z)
		dst[n-k] = -2 * imag(z)
	}
	if n%2 == 0 {
		dst[n/2] = 2 * real(f32Mul(m.w[n/2], V[n/2]))
	}
}

// dct3 is makhoul.dct3 in single precision, src[0] taken times a0. The
// half spectrum exp(+iπk/(2N))·(X[k] - i·X[N-k]) goes through an UNSCALED
// inverse real FFT, which yields the reordered output directly.
func (m f32ndMakhoul) dct3(dst, src []float32, a0 float32, v []float32, V []complex64) {
	n := m.n
	V = V[:n/2+1]
	V[0] = complex(a0*src[0], 0)
	for k := 1; k <= n/2; k++ {
		w := m.w[k]
		V[k] = f32Mul(complex(real(w), -imag(w)), complex(src[k], -src[n-k]))
	}
	m.rp.IRFFTNorm(v, V, NormForward)
	for i := 0; 2*i < n; i++ {
		dst[2*i] = v[i]
	}
	for i := 0; 2*i+1 < n; i++ {
		dst[2*i+1] = v[n-1-i]
	}
}

// f32ndR2RKernel is r2rKernel in single precision: the machinery behind a
// DCTPlan32 or DSTPlan32 of one length and one family of types.
type f32ndR2RKernel struct {
	n         int
	ext       *RealPlan32 // DCT-I/DST-I: the symmetric extension
	mk        f32ndMakhoul
	half      *Plan32 // DCT-IV, even n
	pre, post []complex64
	pool      sync.Pool // *f32ndR2RScratch
}

type f32ndR2RScratch struct {
	r []float32
	c []complex64
	t []float32
}

// f32ndNewR2RKernel is newR2RKernel in single precision.
func f32ndNewR2RKernel(cosine bool, typ, n int) *f32ndR2RKernel {
	k := &f32ndR2RKernel{n: n}
	var nr, nc int
	switch typ {
	case 1:
		l := logicalSize(cosine, 1, n)
		k.ext = cachedRealPlan32(l)
		nr, nc = l, l/2+1
	case 2, 3:
		k.mk = f32ndNewMakhoul(n)
		nr, nc = n, n/2+1
	default:
		if n%2 == 0 {
			h := n / 2
			k.half = cachedPlan32(h)
			k.pre = make([]complex64, h)
			k.post = make([]complex64, h)
			for i := 0; i < h; i++ {
				s, c := math.Sincos(-math.Pi * float64(i) / float64(n))
				k.pre[i] = complex64(complex(c, s))
				s, c = math.Sincos(-math.Pi * float64(4*i+1) / float64(4*n))
				k.post[i] = complex64(complex(2*c, 2*s))
			}
			nc = h
		} else {
			k.mk = f32ndNewMakhoul(2 * n)
			nr, nc = 4*n, n+1
		}
	}
	k.pool.New = func() any {
		return &f32ndR2RScratch{r: make([]float32, nr), c: make([]complex64, nc), t: make([]float32, n)}
	}
	return k
}

// dct1 is r2rKernel.dct1 in single precision.
func (k *f32ndR2RKernel) dct1(dst, src []float32, a float32, s *f32ndR2RScratch) {
	n := k.n
	l := 2 * (n - 1)
	e := s.r[:l]
	e[0] = a * src[0]
	e[n-1] = a * src[n-1]
	for i := 1; i < n-1; i++ {
		e[i] = src[i]
		e[l-i] = src[i]
	}
	F := k.ext.RFFT(s.c, e)
	for i := 0; i < n; i++ {
		dst[i] = real(F[i])
	}
}

// dst1 is r2rKernel.dst1 in single precision.
func (k *f32ndR2RKernel) dst1(dst, src []float32, s *f32ndR2RScratch) {
	n := k.n
	l := 2 * (n + 1)
	e := s.r[:l]
	e[0], e[n+1] = 0, 0
	for i := 0; i < n; i++ {
		e[i+1] = src[i]
		e[l-1-i] = -src[i]
	}
	F := k.ext.RFFT(s.c, e)
	for i := 0; i < n; i++ {
		dst[i] = -imag(F[i+1])
	}
}

// dct4 is r2rKernel.dct4 in single precision; dst may alias src.
func (k *f32ndR2RKernel) dct4(dst, src []float32, s *f32ndR2RScratch) {
	n := k.n
	if k.half != nil {
		// The half-length complex FFT; see r2rKernel.dct4Even.
		h := n / 2
		c := s.c[:h]
		for m := 0; m < h; m++ {
			c[m] = f32Mul(complex(src[2*m], src[n-1-2*m]), k.pre[m])
		}
		k.half.execute(c, c, false)
		for i := 0; i < h; i++ {
			z := f32Mul(k.post[i], c[i])
			dst[2*i] = real(z)
			dst[n-1-2*i] = -imag(z)
		}
		return
	}
	// Odd n: outputs 2k+1 of the 2n-point DCT-II of x padded with n zeros.
	z := s.r[:2*n]
	copy(z, src[:n])
	clear(z[n:])
	k.mk.dct2(z, z, s.r[2*n:4*n], s.c)
	for i := 0; i < n; i++ {
		dst[i] = z[2*i+1]
	}
}

// A DCTPlan32 is the single-precision counterpart of DCTPlan: a reusable DCT
// of a fixed length and type (1 to 4) over float32 data, matching
// scipy.fft.dct and scipy.fft.idct on float32 input. It is immutable and safe
// for concurrent use.
type DCTPlan32 struct {
	typ int
	k   *f32ndR2RKernel
}

// NewDCTPlan32 returns a single-precision plan for the DCT of type typ (1, 2, 3
// or 4) of length n. It panics if typ is not 1..4 or if n is below the type's
// minimum: 2 for DCT-I, 1 for the others.
func NewDCTPlan32(n, typ int) *DCTPlan32 {
	checkR2R(true, typ, n, NormBackward)
	return &DCTPlan32{typ: typ, k: f32ndNewR2RKernel(true, typ, n)}
}

// Len reports the transform length the plan was built for.
func (p *DCTPlan32) Len() int { return p.k.n }

// Type reports the DCT type (1 to 4) the plan was built for.
func (p *DCTPlan32) Type() int { return p.typ }

// DCT writes the DCT of src into dst and returns dst[:Len()], as
// scipy.fft.dct(src, type=Type(), norm=norm) on float32 input. dst and src
// must have at least Len() elements; dst may alias src.
func (p *DCTPlan32) DCT(dst, src []float32, norm Norm) []float32 {
	return p.run(dst, src, norm, false)
}

// IDCT writes the inverse DCT of src into dst and returns dst[:Len()], as
// scipy.fft.idct(src, type=Type(), norm=norm). The same slice rules as DCT
// apply.
func (p *DCTPlan32) IDCT(dst, src []float32, norm Norm) []float32 {
	return p.run(dst, src, norm, true)
}

func (p *DCTPlan32) run(dst, src []float32, norm Norm, inverse bool) []float32 {
	k := p.k
	n := k.n
	checkR2R(true, p.typ, n, norm)
	if len(dst) < n || len(src) < n {
		panic("fft: DCTPlan32 slice shorter than the plan's length")
	}
	typ := p.typ
	if inverse {
		typ = inverseType(typ)
	}
	f := norm.scale(logicalSize(true, typ, n), inverse)
	ortho := norm == NormOrtho
	a := float32(1)
	if ortho {
		a = f32ndSqrt2
	}
	s := k.pool.Get().(*f32ndR2RScratch)
	switch typ {
	case 1:
		k.dct1(dst, src, a, s)
		if ortho {
			dst[0] /= f32ndSqrt2
			dst[n-1] /= f32ndSqrt2
		}
	case 2:
		k.mk.dct2(dst, src, s.r, s.c)
		if ortho {
			dst[0] /= f32ndSqrt2
		}
	case 3:
		k.mk.dct3(dst, src, a, s.r, s.c)
	default:
		k.dct4(dst, src, s)
	}
	k.pool.Put(s)
	scaleReal32(dst[:n], f)
	return dst[:n]
}

// A DSTPlan32 is the single-precision counterpart of DSTPlan, matching
// scipy.fft.dst and scipy.fft.idst on float32 input. It is immutable and safe
// for concurrent use.
type DSTPlan32 struct {
	typ int
	k   *f32ndR2RKernel
}

// NewDSTPlan32 returns a single-precision plan for the DST of type typ (1, 2,
// 3 or 4) of length n >= 1. It panics if typ is not 1..4 or n < 1.
func NewDSTPlan32(n, typ int) *DSTPlan32 {
	checkR2R(false, typ, n, NormBackward)
	return &DSTPlan32{typ: typ, k: f32ndNewR2RKernel(false, typ, n)}
}

// Len reports the transform length the plan was built for.
func (p *DSTPlan32) Len() int { return p.k.n }

// Type reports the DST type (1 to 4) the plan was built for.
func (p *DSTPlan32) Type() int { return p.typ }

// DST writes the DST of src into dst and returns dst[:Len()], as
// scipy.fft.dst(src, type=Type(), norm=norm) on float32 input. dst and src
// must have at least Len() elements; dst may alias src.
func (p *DSTPlan32) DST(dst, src []float32, norm Norm) []float32 {
	return p.run(dst, src, norm, false)
}

// IDST writes the inverse DST of src into dst and returns dst[:Len()], as
// scipy.fft.idst(src, type=Type(), norm=norm). The same slice rules as DST
// apply.
func (p *DSTPlan32) IDST(dst, src []float32, norm Norm) []float32 {
	return p.run(dst, src, norm, true)
}

func (p *DSTPlan32) run(dst, src []float32, norm Norm, inverse bool) []float32 {
	k := p.k
	n := k.n
	checkR2R(false, p.typ, n, norm)
	if len(dst) < n || len(src) < n {
		panic("fft: DSTPlan32 slice shorter than the plan's length")
	}
	typ := p.typ
	if inverse {
		typ = inverseType(typ)
	}
	f := norm.scale(logicalSize(false, typ, n), inverse)
	ortho := norm == NormOrtho
	s := k.pool.Get().(*f32ndR2RScratch)
	t := s.t
	alternate := false
	switch typ {
	case 1:
		k.dst1(dst, src, s)
	case 2:
		// DST-II(x)[k] = DCT-II((-1)^n x[n])[N-1-k]; see dst.go.
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
			dst[n-1] /= f32ndSqrt2
		}
	case 3:
		f32ndReverseInto(t, src[:n])
		a0 := float32(1)
		if ortho {
			a0 = f32ndSqrt2
		}
		k.mk.dct3(dst, t, a0, s.r, s.c)
		alternate = true
	default:
		f32ndReverseInto(t, src[:n])
		k.dct4(dst, t, s)
		alternate = true
	}
	k.pool.Put(s)
	scaleReal32(dst[:n], f)
	if alternate {
		for i := 1; i < n; i += 2 {
			dst[i] = -dst[i]
		}
	}
	return dst[:n]
}

// f32ndReverseInto writes src in reverse order into dst.
func f32ndReverseInto(dst, src []float32) {
	n := len(src)
	for i, v := range src {
		dst[n-1-i] = v
	}
}

var (
	r2r32Mu    sync.Mutex
	r2r32Cache = map[r2rKey]any{}
)

// cachedDCTPlan32 returns the shared plan of type typ and length n (already
// validated), building it on first use.
func cachedDCTPlan32(n, typ int) *DCTPlan32 {
	r2r32Mu.Lock()
	defer r2r32Mu.Unlock()
	key := r2rKey{true, typ, n}
	p, ok := r2r32Cache[key].(*DCTPlan32)
	if !ok {
		p = NewDCTPlan32(n, typ)
		r2r32Cache[key] = p
	}
	return p
}

// cachedDSTPlan32 returns the shared DST plan of type typ and length n.
func cachedDSTPlan32(n, typ int) *DSTPlan32 {
	r2r32Mu.Lock()
	defer r2r32Mu.Unlock()
	key := r2rKey{false, typ, n}
	p, ok := r2r32Cache[key].(*DSTPlan32)
	if !ok {
		p = NewDSTPlan32(n, typ)
		r2r32Cache[key] = p
	}
	return p
}

// DCT32 returns the DCT of type typ (1, 2, 3 or 4) of the float32 signal x,
// scaled by norm, in single precision: scipy.fft.dct(x, type=typ, norm=norm)
// for a float32 x, which scipy returns as float32. x is not modified. The
// same panics as DCT apply.
func DCT32(x []float32, typ int, norm Norm) []float32 {
	checkR2R(true, typ, len(x), norm)
	return cachedDCTPlan32(len(x), typ).DCT(make([]float32, len(x)), x, norm)
}

// IDCT32 returns the inverse DCT of type typ of x in single precision, as
// scipy.fft.idct(x, type=typ, norm=norm). The same panics as DCT apply.
func IDCT32(x []float32, typ int, norm Norm) []float32 {
	checkR2R(true, typ, len(x), norm)
	return cachedDCTPlan32(len(x), typ).IDCT(make([]float32, len(x)), x, norm)
}

// DST32 returns the DST of type typ (1, 2, 3 or 4) of x in single precision,
// as scipy.fft.dst(x, type=typ, norm=norm) for a float32 x. The same panics as
// DST apply.
func DST32(x []float32, typ int, norm Norm) []float32 {
	checkR2R(false, typ, len(x), norm)
	return cachedDSTPlan32(len(x), typ).DST(make([]float32, len(x)), x, norm)
}

// IDST32 returns the inverse DST of type typ of x in single precision, as
// scipy.fft.idst(x, type=typ, norm=norm). The same panics as DST apply.
func IDST32(x []float32, typ int, norm Norm) []float32 {
	checkR2R(false, typ, len(x), norm)
	return cachedDSTPlan32(len(x), typ).IDST(make([]float32, len(x)), x, norm)
}

// DCTN32 is DCTN in single precision: the DCT of type typ along every axis of
// the row-major float32 array x, as scipy.fft.dctn(x, type=typ, norm=norm).
// The same rules as DCTN apply.
func DCTN32(x []float32, shape []int, typ int, norm Norm) []float32 {
	return f32ndR2RN(x, shape, true, typ, norm, false)
}

// IDCTN32 inverts DCTN32, as scipy.fft.idctn(x, type=typ, norm=norm).
func IDCTN32(x []float32, shape []int, typ int, norm Norm) []float32 {
	return f32ndR2RN(x, shape, true, typ, norm, true)
}

// DSTN32 is DSTN in single precision, as scipy.fft.dstn(x, type=typ,
// norm=norm). The same rules as DCTN apply.
func DSTN32(x []float32, shape []int, typ int, norm Norm) []float32 {
	return f32ndR2RN(x, shape, false, typ, norm, false)
}

// IDSTN32 inverts DSTN32, as scipy.fft.idstn(x, type=typ, norm=norm).
func IDSTN32(x []float32, shape []int, typ int, norm Norm) []float32 {
	return f32ndR2RN(x, shape, false, typ, norm, true)
}

// f32ndR2RN is r2rN in single precision: everything is validated first, then
// the 1-D transform runs along every axis of a copy of x.
func f32ndR2RN(x []float32, shape []int, cosine bool, typ int, norm Norm, inverse bool) []float32 {
	checkR2R(cosine, typ, 2, norm)
	total := validateShape(shape, len(x))
	for _, s := range shape {
		checkR2R(cosine, typ, s, norm)
	}
	out := make([]float32, total)
	copy(out, x)
	stride := rowMajorStrides(shape)
	for ax, n := range shape {
		var run func(dst, src []float32, norm Norm) []float32
		switch {
		case cosine && inverse:
			run = cachedDCTPlan32(n, typ).IDCT
		case cosine:
			run = cachedDCTPlan32(n, typ).DCT
		case inverse:
			run = cachedDSTPlan32(n, typ).IDST
		default:
			run = cachedDSTPlan32(n, typ).DST
		}
		line := make([]float32, n)
		sa := stride[ax]
		for c := 0; c < total/n; c++ {
			base := lineBase(c, shape, stride, ax)
			for i := range line {
				line[i] = out[base+i*sa]
			}
			run(line, line, norm)
			for i, v := range line {
				out[base+i*sa] = v
			}
		}
	}
	return out
}
