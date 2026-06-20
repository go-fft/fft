package fft

import (
	"math"
	"sync"
)

// A Plan is a reusable, precomputed transform of a fixed length N. Building a
// plan factors N and precomputes every twiddle factor once; the resulting
// tables are then amortized across an unbounded number of FFT/IFFT calls, so
// repeated transforms of the same length pay no sin/cos cost. Plans are
// immutable after construction and safe for concurrent use by multiple
// goroutines (the transform methods write only into the caller-supplied
// destination and per-call scratch).
//
// For one-off transforms the package-level FFT/IFFT functions are more
// convenient; they consult an internal plan cache so even they avoid
// recomputing twiddles for a length they have seen before.
type Plan struct {
	n int

	// kind selects the engine. A length is resolved by mixed-radix Cooley–Tukey
	// (factors small enough), by Rader's algorithm (a large prime), or by
	// Bluestein (any other length with a too-large prime factor). n <= 1 is the
	// trivial copy.
	bluestein *bluesteinPlan // non-nil iff this length uses Bluestein
	ct        *ctPlan        // non-nil iff this length uses mixed-radix CT
	rader     *raderPlan     // non-nil iff this length uses Rader
	sr        *srPlan        // non-nil iff this length uses split-radix (pow2)
}

// maxRadix bounds the largest prime factor handled by a direct radix-p
// butterfly. Factors above this make the radix-p inner DFT (O(p^2)) costlier
// than reducing the whole transform with Bluestein, so such a length is routed
// to Bluestein instead. Highly-composite lengths (products of 2,3,5,7,...) stay
// on the fast Cooley–Tukey path.
const maxRadix = 13

// NewPlan returns a transform plan for length n, precomputing all twiddle
// factors. n may be any non-negative integer; n == 0 and n == 1 produce a
// trivial plan. The same plan serves both the forward FFT and the inverse IFFT.
func NewPlan(n int) *Plan {
	p := &Plan{n: n}
	if n <= 1 {
		return p
	}
	if n&(n-1) == 0 {
		// Pure power of two: the split-radix kernel (≈⅓ fewer real multiplies
		// than radix-4) wins the mid-range; route it here.
		p.sr = newSRPlan(n)
		return p
	}
	if factorsAreSmall(n) {
		p.ct = newCTPlan(n)
		return p
	}
	// A large prime goes to Rader once it is big enough that Rader's lack of a
	// chirp pre/post-multiply outweighs its index-permutation overhead;
	// otherwise (and for non-prime lengths with a large prime factor) Bluestein.
	// Both reduce the prime to the same power-of-two convolution, but measured
	// head-to-head on the benchmark host Bluestein wins below raderThreshold and
	// Rader wins above it (see docs/perf.md), so the engine is picked per length.
	if n >= raderThreshold && isPrime(n) {
		p.rader = newRaderPlan(n)
		return p
	}
	p.bluestein = newBluesteinPlan(n)
	return p
}

// raderThreshold is the smallest prime routed to Rader instead of Bluestein.
// Below it the two share a convolution size and Bluestein's contiguous chirp
// passes beat Rader's permuted gather/scatter; at and above it Rader's lack of
// a chirp pre/post-multiply and its direct length-(N−1) convolution win. The
// crossover dropped sharply once the split-radix engine sped up the pow2 FFTs
// both algorithms convolve with: re-measured on the 4-core arm64 benchmark host
// it is now ~700 (Bluestein still faster at N=641, Rader faster from N=769 up;
// e.g. 9973: Bluestein 1.41ms vs Rader 0.66ms — Rader is ~2× ahead at N≈10⁴).
const raderThreshold = 700

// Len reports the transform length the plan was built for.
func (p *Plan) Len() int { return p.n }

// FFT writes the forward DFT of src into dst and returns dst. dst and src must
// each have length Len(); dst may alias src. src is not modified unless it
// aliases dst.
func (p *Plan) FFT(dst, src []complex128) []complex128 {
	p.execute(dst, src, false)
	return dst
}

// IFFT writes the inverse DFT of src into dst (normalized by N) and returns
// dst. dst and src must each have length Len(); dst may alias src.
func (p *Plan) IFFT(dst, src []complex128) []complex128 {
	p.execute(dst, src, true)
	n := p.n
	if n == 0 {
		return dst
	}
	inv := complex(1/float64(n), 0)
	for i := range dst {
		dst[i] *= inv
	}
	return dst
}

// execute dispatches to the selected engine, leaving the unnormalized transform
// in dst.
func (p *Plan) execute(dst, src []complex128, inverse bool) {
	n := p.n
	if n <= 1 {
		copy(dst, src)
		return
	}
	switch {
	case p.sr != nil:
		p.sr.transform(dst, src, inverse)
	case p.ct != nil:
		p.ct.transform(dst, src, inverse)
	case p.rader != nil:
		p.rader.transform(dst, src, inverse)
	default:
		p.bluestein.transform(dst, src, inverse)
	}
}

// factorsAreSmall reports whether every prime factor of n is <= maxRadix, i.e.
// whether n can be handled entirely by mixed-radix Cooley–Tukey without a
// Bluestein step.
func factorsAreSmall(n int) bool {
	for _, prime := range []int{2, 3, 5, 7, 11, 13} {
		for n%prime == 0 {
			n /= prime
		}
	}
	return n == 1
}

// factorize splits n into an ordered list of radices, preferring 4 over 2·2 (a
// radix-4 butterfly needs fewer multiplies than two radix-2 stages) and pulling
// out small primes in ascending order. Every returned factor is <= maxRadix and
// their product is n. factorsAreSmall(n) must hold.
func factorize(n int) []int {
	var f []int
	// Pull out 4s first (radix-4 is cheaper than 2·2), leaving at most one 2.
	for n%4 == 0 {
		f = append(f, 4)
		n /= 4
	}
	for _, prime := range []int{2, 3, 5, 7, 11, 13} {
		for n%prime == 0 {
			f = append(f, prime)
			n /= prime
		}
	}
	return f
}

// --- package-level plan cache -------------------------------------------------

// planCache memoizes plans by length so the convenience FFT/IFFT functions and
// the RFFT/IRFFT helpers avoid rebuilding twiddle tables for a repeated length.
// It is keyed by n only (a plan serves both directions), bounded implicitly by
// the set of distinct lengths the program uses.
var (
	planMu    sync.Mutex
	planCache = map[int]*Plan{}
)

// cachedPlan returns a shared plan for length n, building and memoizing it on
// first use. The returned plan is immutable and safe for concurrent transforms.
func cachedPlan(n int) *Plan {
	planMu.Lock()
	p, ok := planCache[n]
	if !ok {
		p = NewPlan(n)
		planCache[n] = p
	}
	planMu.Unlock()
	return p
}

// twiddleTable returns the n forward roots of unity exp(-2πi·k/n), k=0..n-1.
func twiddleTable(n int) []complex128 {
	t := make([]complex128, n)
	for k := 0; k < n; k++ {
		ang := -2 * math.Pi * float64(k) / float64(n)
		t[k] = complex(math.Cos(ang), math.Sin(ang))
	}
	return t
}
