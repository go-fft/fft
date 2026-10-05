package fft

import (
	"math"
	"strconv"
	"sync"
)

// This file implements the discrete cosine transforms of types I–IV, and the
// fast kernels the discrete sine transforms of dst.go reuse. The definitions,
// the logical sizes used for normalisation and the "ortho" corrections are
// those of scipy.fft.dct/idct (scipy 1.18, orthogonalize left at its default,
// which is True exactly when norm="ortho"):
//
//	DCT-I   y[k] = x[0] + (-1)^k x[N-1] + 2 Σ_{n=1}^{N-2} x[n] cos(πkn/(N-1))      logical size 2(N-1)
//	DCT-II  y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(πk(2n+1)/(2N))                         logical size 2N
//	DCT-III y[k] = x[0] + 2 Σ_{n=1}^{N-1} x[n] cos(π(2k+1)n/(2N))                  logical size 2N
//	DCT-IV  y[k] = 2 Σ_{n=0}^{N-1} x[n] cos(π(2k+1)(2n+1)/(4N))                    logical size 2N
//
// These are FFTW's REDFT00, REDFT10, REDFT01 and REDFT11 exactly (FFTW manual,
// §4.8.3 "1d Real-even DFTs (DCTs)", whose logical size N is the one used
// here). Norm m scales the forward transform by m.scale(M, false) and the
// inverse by m.scale(M, true), M the logical size. Under NormOrtho, as scipy
// does by default, DCT-I multiplies x[0], x[N-1] by √2 and divides y[0],
// y[N-1] by √2; DCT-II divides y[0] by √2; DCT-III multiplies x[0] by √2. The
// inverse of type t is the transform of type 1, 3, 2, 4 (for t = 1, 2, 3, 4)
// with the inverse scaling, which is how scipy.fft.idct is defined.
//
// Every path is O(N log N) through the package's RealPlan/Plan:
//
//   - DCT-I: the real FFT of the even extension x[0..N-1], x[N-2..1] (length
//     2(N-1)), whose spectrum is real and is the DCT-I (FFTW manual §4.8.3,
//     REDFT00 "corresponds to a DFT of real-even data of logical size
//     2(N-1)"; the same reduction pocketfft/ducc0 use).
//   - DCT-II and DCT-III: Makhoul's N-point algorithm (J. Makhoul, "A fast
//     cosine transform in one and two dimensions", IEEE Trans. ASSP 28(1),
//     27–34, 1980): reorder v = x[0], x[2], …, x[3], x[1], one N-point real
//     FFT V, then y[k] = 2·Re(exp(-iπk/(2N))·V[k]); DCT-III runs the same steps
//     backwards through one N-point inverse real FFT.
//   - DCT-IV, even N: one N/2-point complex FFT of
//     (x[2m] + i·x[N-1-2m])·exp(-iπm/N) with a post-twiddle exp(-iπ(4k+1)/(4N))
//     that yields y[2k] and y[N-1-2k] together (the standard "half-length
//     complex FFT" DCT-IV; derivation in the comment of dct4Even).
//   - DCT-IV, odd N: y[k] is output 2k+1 of the 2N-point DCT-II of x padded
//     with N zeros, computed by Makhoul's algorithm on a 2N-point real FFT.

// sqrt2 is √2, the ortho correction factor.
const sqrt2 = math.Sqrt2

// minR2RLen is the smallest length scipy accepts for each type, indexed by
// type: DCT-I needs N >= 2 (its logical size 2(N-1) must be positive), every
// other DCT/DST type N >= 1. scipy 1.18 accepts DST-I of length 1 although its
// docstring says otherwise; we follow what it computes.
func minR2RLen(cosine bool, typ int) int {
	if cosine && typ == 1 {
		return 2
	}
	return 1
}

// r2rName names a transform for messages: "DCT-II", "DST-IV".
func r2rName(cosine bool, typ int) string {
	name := "DST-"
	if cosine {
		name = "DCT-"
	}
	return name + [...]string{"I", "II", "III", "IV"}[typ-1]
}

// checkR2R panics with the package's message if typ is not 1..4, if n is
// below the type's minimum, or if norm is not one of the three modes. It runs
// before anything is allocated.
func checkR2R(cosine bool, typ, n int, norm Norm) {
	if typ < 1 || typ > 4 {
		kind := "DST"
		if cosine {
			kind = "DCT"
		}
		panic("fft: unknown " + kind + " type " + strconv.Itoa(typ) + " (want 1, 2, 3 or 4)")
	}
	if lo := minR2RLen(cosine, typ); n < lo {
		panic("fft: " + r2rName(cosine, typ) + " length " + strconv.Itoa(n) + " is below its minimum " + strconv.Itoa(lo))
	}
	norm.scale(1, false) // panics on an unknown Norm
}

// inverseType is the type whose transform inverts type typ: I and IV are their
// own inverses, II and III invert each other.
func inverseType(typ int) int {
	return [...]int{1, 3, 2, 4}[typ-1]
}

// logicalSize is the length M the normalisation divides by: 2(N-1) for DCT-I,
// 2(N+1) for DST-I, 2N for every other type (FFTW's "logical size").
func logicalSize(cosine bool, typ, n int) int {
	switch {
	case typ != 1:
		return 2 * n
	case cosine:
		return 2 * (n - 1)
	}
	return 2 * (n + 1)
}

// makhoul holds the tables of Makhoul's N-point DCT-II/DCT-III: an N-point
// real plan and the twiddles w[k] = exp(-iπk/(2N)) for k = 0 .. N/2.
type makhoul struct {
	n  int
	rp *RealPlan
	w  []complex128
}

func newMakhoul(n int) makhoul {
	w := make([]complex128, n/2+1)
	for k := range w {
		s, c := math.Sincos(-math.Pi * float64(k) / float64(2*n))
		w[k] = complex(c, s)
	}
	return makhoul{n: n, rp: NewRealPlan(n), w: w}
}

// dct2 writes the unnormalised DCT-II of src[:n] into dst[:n], using v (n
// reals) and V (n/2+1 complex) as scratch. dst may alias src: src is read in
// full before dst is written.
//
// Makhoul (1980), eqs. (20)–(23): with v[i] = x[2i] and v[n-1-i] = x[2i+1],
// V = DFT(v) and z = exp(-iπk/(2N))·V[k], y[k] = 2·Re z. The bin n-k needs no
// second DFT value: since V[n-k] = conj(V[k]) and exp(-iπ(n-k)/(2N)) =
// -i·conj(exp(-iπk/(2N))), y[n-k] = -2·Im z. One real FFT of the same length
// therefore yields all n outputs, two per half-spectrum bin.
func (m makhoul) dct2(dst, src, v []float64, V []complex128) {
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
		z := m.w[k] * V[k]
		dst[k] = 2 * real(z)
		dst[n-k] = -2 * imag(z)
	}
	if n%2 == 0 {
		dst[n/2] = 2 * real(m.w[n/2]*V[n/2])
	}
}

// dct3 writes the unnormalised DCT-III of src[:n] into dst[:n], src[0] taken
// times a0 (√2 for the ortho correction, 1 otherwise). v and V are scratch as
// for dct2; dst may alias src.
//
// It inverts dct2's steps (Makhoul 1980, §III.B): the unnormalised DCT-III is
// 2N times the inverse of the DCT-II, and given X = DCT-II(x) the DFT of the
// reordered v is V[k] = exp(+iπk/(2N))·(X[k] - i·X[N-k])/2, with X[N] = 0.
// That V is Hermitian for any real X, so one inverse real FFT (which divides
// by N) of N·exp(iπk/(2N))·(X[k] - i·X[N-k]) gives v, and undoing the reorder
// gives y.
func (m makhoul) dct3(dst, src []float64, a0 float64, v []float64, V []complex128) {
	n := m.n
	fn := float64(n)
	V = V[:n/2+1]
	V[0] = complex(fn*a0*src[0], 0)
	for k := 1; k <= n/2; k++ {
		w := m.w[k]
		V[k] = complex(fn*real(w), -fn*imag(w)) * complex(src[k], -src[n-k])
	}
	m.rp.IRFFT(v, V)
	for i := 0; 2*i < n; i++ {
		dst[2*i] = v[i]
	}
	for i := 0; 2*i+1 < n; i++ {
		dst[2*i+1] = v[n-1-i]
	}
}

// r2rKernel is the precomputed machinery for one length and one family of
// types: what a DCTPlan or DSTPlan of that length runs. Only the fields its
// family needs are set. It is immutable after construction; per-call working
// memory comes from pool.
type r2rKernel struct {
	n int

	// ext is the real plan of the symmetric extension: length 2(n-1) for
	// DCT-I, 2(n+1) for DST-I.
	ext *RealPlan

	// mk serves DCT-II/III of length n, and DCT-IV of odd n at length 2n.
	mk makhoul

	// DCT-IV, even n: the n/2-point complex plan and its twiddles
	// pre[m] = exp(-iπm/n), post[k] = 2·exp(-iπ(4k+1)/(4n)).
	half      *Plan
	pre, post []complex128

	pool sync.Pool // *r2rScratch
}

// r2rScratch is one call's working memory: r and c for the kernel, t for the
// reordered input of a DST.
type r2rScratch struct {
	r []float64
	c []complex128
	t []float64
}

// newR2RKernel builds the kernel of family typ (1 = the type-I extension of
// DCT-I if cosine or DST-I otherwise, 2 or 3 = Makhoul, 4 = DCT-IV) for
// length n, already validated.
func newR2RKernel(cosine bool, typ, n int) *r2rKernel {
	k := &r2rKernel{n: n}
	var nr, nc int
	switch typ {
	case 1:
		l := logicalSize(cosine, 1, n)
		k.ext = NewRealPlan(l)
		nr, nc = l, l/2+1
	case 2, 3:
		k.mk = newMakhoul(n)
		nr, nc = n, n/2+1
	default:
		if n%2 == 0 {
			h := n / 2
			k.half = NewPlan(h)
			k.pre = make([]complex128, h)
			k.post = make([]complex128, h)
			for i := 0; i < h; i++ {
				s, c := math.Sincos(-math.Pi * float64(i) / float64(n))
				k.pre[i] = complex(c, s)
				s, c = math.Sincos(-math.Pi * float64(4*i+1) / float64(4*n))
				k.post[i] = complex(2*c, 2*s)
			}
			nc = h
		} else {
			k.mk = newMakhoul(2 * n)
			nr, nc = 4*n, n+1
		}
	}
	k.pool.New = func() any {
		return &r2rScratch{r: make([]float64, nr), c: make([]complex128, nc), t: make([]float64, n)}
	}
	return k
}

// dct1 writes the unnormalised DCT-I of src[:n] into dst[:n], the end samples
// taken times a (√2 for the ortho correction). The real FFT of the even
// extension e = x[0], …, x[n-1], x[n-2], …, x[1] (length 2(n-1)) is real and
// equals the DCT-I (FFTW manual §4.8.3, REDFT00).
func (k *r2rKernel) dct1(dst, src []float64, a float64, s *r2rScratch) {
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

// dst1 writes the unnormalised DST-I of src[:n] into dst[:n]. The real FFT of
// the odd extension e = 0, x[0], …, x[n-1], 0, -x[n-1], …, -x[0] (length
// 2(n+1)) is F[k] = -2i·Σ x[j]·sin(π(j+1)k/(n+1)), so y[k] = -Im F[k+1]
// (FFTW manual §4.8.4, RODFT00).
func (k *r2rKernel) dst1(dst, src []float64, s *r2rScratch) {
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

// dct4 writes the unnormalised DCT-IV of src[:n] into dst[:n]; dst may alias
// src.
func (k *r2rKernel) dct4(dst, src []float64, s *r2rScratch) {
	if k.half != nil {
		k.dct4Even(dst, src, s)
		return
	}
	// Odd n: y[k] = Y[2k+1] where Y is the 2n-point DCT-II of x padded with n
	// zeros, since cos(π(2k+1)(2j+1)/(4n)) is the DCT-II kernel of length 2n
	// at output 2k+1.
	n := k.n
	z := s.r[:2*n]
	copy(z, src[:n])
	clear(z[n:])
	k.mk.dct2(z, z, s.r[2*n:4*n], s.c)
	for i := 0; i < n; i++ {
		dst[i] = z[2*i+1]
	}
}

// dct4Even is the DCT-IV of even n through one n/2-point complex FFT.
//
// Derivation. Let φ(k,m) = π(4k+1)(4m+1)/(4n) and c[m] = x[2m] + i·x[n-1-2m].
// Splitting the sum over j into even j = 2m and odd j = n-1-2m, and using
// cos(π(4k+1)/2 - φ) = sin φ and, for the mirrored outputs, the same identity
// with the roles of cos and sin exchanged,
//
//	y[2k]     =  2·Σ_m (x[2m] cos φ + x[n-1-2m] sin φ) =  2·Re S[k]
//	y[n-1-2k] =  2·Σ_m (x[2m] sin φ - x[n-1-2m] cos φ) = -2·Im S[k]
//
// with S[k] = Σ_m c[m]·exp(-iφ(k,m)). Since φ = 2πkm/(n/2) + π(4k+1)/(4n) +
// πm/n, S[k] = exp(-iπ(4k+1)/(4n)) · DFT_{n/2}(c[m]·exp(-iπm/n))[k].
func (k *r2rKernel) dct4Even(dst, src []float64, s *r2rScratch) {
	n := k.n
	h := n / 2
	c := s.c[:h]
	for m := 0; m < h; m++ {
		c[m] = complex(src[2*m], src[n-1-2*m]) * k.pre[m]
	}
	k.half.FFT(c, c)
	for i := 0; i < h; i++ {
		z := k.post[i] * c[i]
		dst[2*i] = real(z)
		dst[n-1-2*i] = -imag(z)
	}
}

// A DCTPlan is a reusable discrete cosine transform of a fixed length and
// type (1 to 4), matching scipy.fft.dct and scipy.fft.idct. Like Plan it
// precomputes every twiddle factor once; its methods write into a
// caller-supplied destination. A DCTPlan is immutable and safe for concurrent
// use.
type DCTPlan struct {
	typ int
	k   *r2rKernel
}

// NewDCTPlan returns a plan for the DCT of type typ (1, 2, 3 or 4) of length
// n. It panics if typ is not 1..4 or if n is below the type's minimum: 2 for
// DCT-I, 1 for the others.
func NewDCTPlan(n, typ int) *DCTPlan {
	checkR2R(true, typ, n, NormBackward)
	return &DCTPlan{typ: typ, k: newR2RKernel(true, typ, n)}
}

// Len reports the transform length the plan was built for.
func (p *DCTPlan) Len() int { return p.k.n }

// Type reports the DCT type (1 to 4) the plan was built for.
func (p *DCTPlan) Type() int { return p.typ }

// DCT writes the DCT of src into dst and returns dst[:Len()], as
// scipy.fft.dct(src, type=Type(), norm=norm). dst and src must have at least
// Len() elements; dst may alias src. src is not modified unless it aliases
// dst.
func (p *DCTPlan) DCT(dst, src []float64, norm Norm) []float64 {
	return p.run(dst, src, norm, false)
}

// IDCT writes the inverse DCT of src into dst and returns dst[:Len()], as
// scipy.fft.idct(src, type=Type(), norm=norm): IDCT(DCT(x, norm), norm) == x
// for every norm. The same slice rules as DCT apply.
func (p *DCTPlan) IDCT(dst, src []float64, norm Norm) []float64 {
	return p.run(dst, src, norm, true)
}

func (p *DCTPlan) run(dst, src []float64, norm Norm, inverse bool) []float64 {
	k := p.k
	n := k.n
	checkR2R(true, p.typ, n, norm)
	if len(dst) < n || len(src) < n {
		panic("fft: DCTPlan slice shorter than the plan's length")
	}
	typ := p.typ
	if inverse {
		typ = inverseType(typ)
	}
	f := norm.scale(logicalSize(true, typ, n), inverse)
	ortho := norm == NormOrtho
	a := 1.0
	if ortho {
		a = sqrt2
	}
	s := k.pool.Get().(*r2rScratch)
	switch typ {
	case 1:
		k.dct1(dst, src, a, s)
		if ortho {
			dst[0] /= sqrt2
			dst[n-1] /= sqrt2
		}
	case 2:
		k.mk.dct2(dst, src, s.r, s.c)
		if ortho {
			dst[0] /= sqrt2
		}
	case 3:
		k.mk.dct3(dst, src, a, s.r, s.c)
	default:
		k.dct4(dst, src, s)
	}
	k.pool.Put(s)
	scaleReal(dst[:n], f)
	return dst[:n]
}

// scaleAlternating multiplies x by f at even indices and by -f at odd ones.
func scaleAlternating(x []float64, f float64) {
	for i := range x {
		if i%2 == 0 {
			x[i] *= f
		} else {
			x[i] *= -f
		}
	}
}

// r2rKey identifies a cached plan: the family, the type and the length.
type r2rKey struct {
	cosine bool
	typ, n int
}

var (
	r2rMu    sync.Mutex
	r2rCache = map[r2rKey]any{}
)

// cachedDCTPlan returns the shared plan of type typ and length n, building it
// on first use. Arguments are validated by the caller.
func cachedDCTPlan(n, typ int) *DCTPlan {
	r2rMu.Lock()
	defer r2rMu.Unlock()
	key := r2rKey{true, typ, n}
	p, ok := r2rCache[key].(*DCTPlan)
	if !ok {
		p = NewDCTPlan(n, typ)
		r2rCache[key] = p
	}
	return p
}

// DCT returns the discrete cosine transform of type typ (1, 2, 3 or 4) of x,
// scaled by norm, as scipy.fft.dct(x, type=typ, norm=norm). The definitions
// are those of scipy (see the DCTPlan methods); NormOrtho also applies
// scipy's default orthogonalisation. x is not modified.
//
// It panics if typ is not 1..4, if len(x) is below the type's minimum (2 for
// DCT-I, 1 otherwise) or if norm is unknown.
func DCT(x []float64, typ int, norm Norm) []float64 {
	checkR2R(true, typ, len(x), norm)
	return cachedDCTPlan(len(x), typ).DCT(make([]float64, len(x)), x, norm)
}

// IDCT returns the inverse discrete cosine transform of type typ of x, as
// scipy.fft.idct(x, type=typ, norm=norm), so that IDCT(DCT(x, t, m), t, m)
// reproduces x. The same panics as DCT apply.
func IDCT(x []float64, typ int, norm Norm) []float64 {
	checkR2R(true, typ, len(x), norm)
	return cachedDCTPlan(len(x), typ).IDCT(make([]float64, len(x)), x, norm)
}
