package fft

import (
	"sync"

	"github.com/go-fft/fft/internal/kernels"
)

// A Plan32 is the single-precision counterpart of Plan: a reusable transform
// of a fixed length N over complex64 data, as scipy.fft computes when it is
// given complex64 input (and numpy.fft since NumPy 2.0). Every
// length is accepted and routed as Plan routes it: smooth lengths (prime
// factors <= 13) to the Stockham passes, a prime whose N-1 is 7-smooth to
// Rader's algorithm, anything else to Bluestein's chirp-z.
//
// Precision. The arithmetic is float32 throughout, as in FFTW's fftwf_ and
// pocketfft's float instantiation, but every precomputed table — twiddles,
// Rader's and Bluestein's kernel spectra, Bluestein's chirp — is computed in
// float64 by the double-precision engine and rounded once to float32. A table
// built in float32 (or by a float32 sin/cos recurrence) would carry its own
// O(eps32·log N) error into every transform; rounded once, each entry is
// within half an ulp. The error of a transform is then O(eps32·log2 N)
// relative to the vector's 2-norm; the tests hold it to eps32·log2 N against
// the DFT of the same input computed in float64, for every length up to 130
// and for powers of two up to 2^20, composites and primes beyond. The worst
// error measured is 0.49 of that bound (N = 2, NormOrtho), 0.14 from 1024
// points up.
//
// The engines are separate float32 code, not a generic instantiation of the
// float64 ones, because a generic engine measured slower on the float64 path.
// Go does not let real, imag or complex take a type-parameter argument
// (go.dev/issue/50937), so a generic butterfly has to reach the parts through
// a conversion to complex128 — free for T = complex128 in itself, but it
// raises the inliner's cost of the butterflies (bfly4 from 56 to 88, bfly3
// from 69 to 175, over the budget of 80), they stop being inlined into the
// pass loops, and the float64 transforms got 1.44× slower (geometric mean of
// the Complex/Real/CReal benchmarks, interleaved A/B, Apple M4 Max). The
// float64 path also dispatches to amd64 assembly through internal/kernels,
// which a complex64 instantiation would have to bypass anyway.
//
// Plans are immutable after construction and safe for concurrent use.
type Plan32 struct {
	n         int
	sk        *skPlan32        // smooth length
	rader     *raderPlan32     // prime with 7-smooth n-1
	bluestein *bluesteinPlan32 // any other length
}

// NewPlan32 returns a single-precision transform plan for length n,
// precomputing all twiddle factors. n may be any non-negative integer; n == 0
// and n == 1 produce a trivial plan.
func NewPlan32(n int) *Plan32 {
	p := &Plan32{n: n}
	if n <= 1 {
		return p
	}
	if factorsAreSmall(n) {
		p.sk = newSKPlan32(n)
		return p
	}
	if isPrime(n) {
		if _, ok := convCost(n - 1); ok {
			p.rader = newRaderPlan32(n)
			return p
		}
	}
	p.bluestein = newBluesteinPlan32(n)
	return p
}

// Len reports the transform length the plan was built for.
func (p *Plan32) Len() int { return p.n }

// FFT writes the forward DFT of src into dst, unnormalized, and returns dst.
// dst and src must each have at least Len() elements; dst may alias src.
func (p *Plan32) FFT(dst, src []complex64) []complex64 {
	return p.FFTNorm(dst, src, NormBackward)
}

// IFFT writes the inverse DFT of src into dst, normalized by N, and returns
// dst. dst and src must each have at least Len() elements; dst may alias src.
func (p *Plan32) IFFT(dst, src []complex64) []complex64 {
	return p.IFFTNorm(dst, src, NormBackward)
}

// FFTNorm writes the forward DFT of src into dst, scaled as norm says (1 for
// NormBackward, 1/sqrt(N) for NormOrtho, 1/N for NormForward), and returns
// dst. It panics on a Norm that is not one of the three constants.
func (p *Plan32) FFTNorm(dst, src []complex64, norm Norm) []complex64 {
	s := norm.scale(p.n, false)
	p.checkLen(dst, src)
	p.execute(dst, src, false)
	scale32(dst[:p.n], s)
	return dst
}

// IFFTNorm writes the inverse DFT of src into dst, scaled as norm says (1/N
// for NormBackward, 1/sqrt(N) for NormOrtho, 1 for NormForward), and returns
// dst. It panics on a Norm that is not one of the three constants.
func (p *Plan32) IFFTNorm(dst, src []complex64, norm Norm) []complex64 {
	s := norm.scale(p.n, true)
	p.checkLen(dst, src)
	p.execute(dst, src, true)
	scale32(dst[:p.n], s)
	return dst
}

// checkLen panics with the package's message when dst or src is shorter than
// the plan's length.
func (p *Plan32) checkLen(dst, src []complex64) {
	if len(dst) < p.n || len(src) < p.n {
		panic("fft: Plan32 slice shorter than the plan's length")
	}
}

// execute leaves the unnormalized transform of src in dst.
func (p *Plan32) execute(dst, src []complex64, inverse bool) {
	switch {
	case p.n <= 1:
		copy(dst[:p.n], src[:p.n])
	case p.sk != nil:
		p.sk.transform(dst, src, inverse)
	case p.rader != nil:
		p.rader.transform(dst, src, inverse)
	default:
		p.bluestein.transform(dst, src, inverse)
	}
}

// scale32 multiplies x by s, rounded once to float32; s == 1 is a no-op.
func scale32(x []complex64, s float64) {
	if s == 1 {
		return
	}
	f := float32(s)
	for i, v := range x {
		x[i] = complex(real(v)*f, imag(v)*f)
	}
}

// FFT32 returns the forward discrete Fourier transform of x in single
// precision, unnormalized like FFT. The input is not modified; empty input
// returns an empty (non-nil) slice. Plans are cached per length.
func FFT32(x []complex64) []complex64 {
	out := make([]complex64, len(x))
	return cachedPlan32(len(x)).FFT(out, x)
}

// IFFT32 returns the inverse discrete Fourier transform of x in single
// precision, normalized by N like IFFT. The input is not modified.
func IFFT32(x []complex64) []complex64 {
	out := make([]complex64, len(x))
	return cachedPlan32(len(x)).IFFT(out, x)
}

var (
	plan32Mu    sync.Mutex
	plan32Cache = map[int]*Plan32{}
)

// cachedPlan32 returns a shared Plan32 for length n, building it on first
// use. Unlike cachedPlan it builds under the lock: NewPlan32 never re-enters
// this cache (its Rader and Bluestein sub-transforms are private Stockham
// plans, and their kernels come from the float64 cache).
func cachedPlan32(n int) *Plan32 {
	plan32Mu.Lock()
	defer plan32Mu.Unlock()
	p, ok := plan32Cache[n]
	if !ok {
		p = NewPlan32(n)
		plan32Cache[n] = p
	}
	return p
}

// toC64 rounds every entry of t to complex64.
func toC64(t []complex128) []complex64 {
	out := make([]complex64, len(t))
	for i, v := range t {
		out[i] = complex64(v)
	}
	return out
}

// --- Stockham -----------------------------------------------------------------

// skStage32 is one single-precision Stockham pass; see skStage.
type skStage32 struct {
	r, l1, ido int
	tw, twc    []complex64
	rt, rtc    []complex64
	// twK, twKc are the pass's twiddles in the layout of the float32 pass
	// kernels (kernels.StockhamTwiddles32), nil when no kernel runs it.
	twK, twKc []complex64
}

// skPlan32 is the single-precision Stockham plan for one smooth length.
type skPlan32 struct {
	n       int
	stages  []skStage32
	scratch sync.Pool
}

// newSKPlan32 builds the plan with the float64 engine's factorization. Every
// twiddle is taken from the float64 root table and rounded once.
func newSKPlan32(n int) *skPlan32 {
	// The pocketfft order: the odd-first order of skFactorize on amd64 was
	// measured for the complex128 AVX2 pass kernels only (Round 17).
	return newSKPlan32Factors(n, skFactorizeOrder(n, false))
}

// newSKPlan32Factors builds the plan for the given radix order.
func newSKPlan32Factors(n int, factors []int) *skPlan32 {
	root := twiddleTable(n)
	p := &skPlan32{n: n}
	l1 := 1
	for _, r := range factors {
		ido := n / (l1 * r)
		st := skStage32{r: r, l1: l1, ido: ido}
		if ido > 1 {
			st.tw = make([]complex64, (r-1)*(ido-1))
			st.twc = make([]complex64, (r-1)*(ido-1))
			for j := 1; j < r; j++ {
				for i := 1; i < ido; i++ {
					w := root[(j*l1*i)%n]
					st.tw[(j-1)*(ido-1)+i-1] = complex64(w)
					st.twc[(j-1)*(ido-1)+i-1] = complex64(complexConj(w))
				}
			}
		}
		st.twK, st.twKc = kernels.StockhamTwiddles32(r, ido, l1, root)
		switch r {
		case 2, 3, 4, 5, 7, 8:
		default:
			st.rt = make([]complex64, r)
			st.rtc = make([]complex64, r)
			for k := 0; k < r; k++ {
				w := root[k*(n/r)]
				st.rt[k] = complex64(w)
				st.rtc[k] = complex64(complexConj(w))
			}
		}
		p.stages = append(p.stages, st)
		l1 *= r
	}
	p.scratch.New = func() any { b := make([]complex64, n); return &b }
	return p
}

// transform writes the unnormalized DFT of src into dst (conjugate roots when
// inverse). dst may alias src.
func (p *skPlan32) transform(dst, src []complex64, inverse bool) {
	bp := p.scratch.Get().(*[]complex64)
	scr := (*bp)[:p.n]
	s := len(p.stages)
	in := src
	if s%2 == 1 && &dst[0] == &src[0] {
		copy(scr, src[:p.n])
		in = scr
	}
	for k := range p.stages {
		out := scr
		if (s-1-k)%2 == 0 {
			out = dst
		}
		p.stages[k].pass(out, in, inverse)
		in = out
	}
	p.scratch.Put(bp)
}

// pass runs the stage on a float32 pass kernel when one runs it
// (kernels.StockhamPass32), otherwise on the Go pass; the two are
// bit-identical.
func (st *skStage32) pass(ch, cc []complex64, inverse bool) {
	twK := st.twK
	if inverse {
		twK = st.twKc
	}
	if kernels.StockhamPass32(st.r, st.ido, st.l1, cc, ch, twK, inverse) {
		return
	}
	st.passScalar(ch, cc, inverse)
}

// passScalar runs the stage on its Go pass.
func (st *skStage32) passScalar(ch, cc []complex64, inverse bool) {
	tw := st.tw
	if inverse {
		tw = st.twc
	}
	switch st.r {
	case 2:
		f32Pass2(st.ido, st.l1, cc, ch, tw)
	case 3:
		f32Pass3(st.ido, st.l1, cc, ch, tw, inverse)
	case 4:
		f32Pass4(st.ido, st.l1, cc, ch, tw, inverse)
	case 5:
		f32Pass5(st.ido, st.l1, cc, ch, tw, inverse)
	case 7:
		f32Pass7(st.ido, st.l1, cc, ch, tw, inverse)
	case 8:
		f32Pass8(st.ido, st.l1, cc, ch, tw, inverse)
	default:
		rt := st.rt
		if inverse {
			rt = st.rtc
		}
		f32Passg(st.r, st.ido, st.l1, cc, ch, tw, rt)
	}
}

// --- Rader ----------------------------------------------------------------------

// raderPlan32 is Rader's algorithm in single precision; see raderPlan. Its
// permutations and kernel spectra come from the float64 plan for the same
// prime, the spectra rounded once.
type raderPlan32 struct {
	n           int
	perm, iperm []int
	bF, bI      []complex64
	conv        *skPlan32 // length n-1
	scratch     sync.Pool
}

func newRaderPlan32(n int) *raderPlan32 {
	d := newRaderPlan(n)
	q := n - 1
	p := &raderPlan32{n: n, perm: d.perm, iperm: d.iperm, bF: toC64(d.bF), bI: toC64(d.bI), conv: newSKPlan32(q)}
	p.scratch.New = func() any { b := make([]complex64, q); return &b }
	return p
}

func (p *raderPlan32) transform(dst, src []complex64, inverse bool) {
	n, q := p.n, p.n-1
	bSpec := p.bF
	if inverse {
		bSpec = p.bI
	}
	// The DC bin sums n inputs: accumulate in float64, so its error does not
	// grow with n the way a float32 running sum's would.
	x0 := src[0]
	var sr, si float64
	for _, v := range src[:n] {
		sr += float64(real(v))
		si += float64(imag(v))
	}
	bp := p.scratch.Get().(*[]complex64)
	defer p.scratch.Put(bp)
	a := (*bp)[:q]
	for k, j := range p.perm[:q] {
		a[k] = src[j]
	}
	p.conv.transform(a, a, false)
	bSpec = bSpec[:q]
	for i := range a {
		a[i] = f32Mul(a[i], bSpec[i])
	}
	p.conv.transform(a, a, true)
	dst[0] = complex(float32(sr), float32(si))
	for qi, j := range p.iperm[:q] {
		dst[j] = x0 + a[qi]
	}
}

// --- Bluestein --------------------------------------------------------------------

// bluesteinPlan32 is Bluestein's chirp-z in single precision; see
// bluesteinPlan. Chirps and kernel spectra come from the float64 plan for the
// same length, rounded once.
type bluesteinPlan32 struct {
	n, m    int
	wF, wI  []complex64
	bF, bI  []complex64
	conv    *skPlan32 // length m
	scratch sync.Pool
}

func newBluesteinPlan32(n int) *bluesteinPlan32 {
	d := newBluesteinPlan(n)
	m := d.m
	p := &bluesteinPlan32{n: n, m: m, wF: toC64(d.wF), wI: toC64(d.wI), bF: toC64(d.bF), bI: toC64(d.bI), conv: newSKPlan32(m)}
	p.scratch.New = func() any { b := make([]complex64, m); return &b }
	return p
}

func (p *bluesteinPlan32) transform(dst, src []complex64, inverse bool) {
	n := p.n
	w, bSpec := p.wF, p.bF
	if inverse {
		w, bSpec = p.wI, p.bI
	}
	bp := p.scratch.Get().(*[]complex64)
	defer p.scratch.Put(bp)
	a := *bp
	for j := 0; j < n; j++ {
		a[j] = f32Mul(src[j], w[j])
	}
	clear(a[n:])
	p.conv.transform(a, a, false)
	bSpec = bSpec[:len(a)]
	for i := range a {
		a[i] = f32Mul(a[i], bSpec[i])
	}
	p.conv.transform(a, a, true)
	for k := 0; k < n; k++ {
		dst[k] = f32Mul(a[k], w[k])
	}
}
