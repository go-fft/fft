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

	// kind selects the engine. A length is resolved by the iterative mixed-radix
	// Stockham engine (factors small enough), by Rader's algorithm (a large prime), or by
	// Bluestein (any other length with a too-large prime factor). n <= 1 is the
	// trivial copy.
	bluestein *bluesteinPlan // non-nil iff this length uses Bluestein
	sk        *skPlan        // non-nil iff this length uses mixed-radix Stockham
	rader     *raderPlan     // non-nil iff this length uses Rader
	it        *itPlan        // non-nil iff this length uses the iterative pow2 kernel
}

// pow2StockhamMax is the largest power of two routed to the Stockham engine
// instead of the iterative pow2 kernel. It is per-architecture (route_*.go)
// and a variable only so the tests can run both routes on every architecture.
var pow2StockhamMax = pow2StockhamMaxDefault()

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
	if n&(n-1) == 0 && n > pow2StockhamMax {
		// Pure power of two: the iterative cache-friendly kernel (one bit-reversal
		// + radix-4 DIT stages, twiddles laid out for sequential reads) measured
		// faster than the recursive split-radix engine at every power-of-two length
		// on the benchmark host — it matches the operation count but wins the memory
		// schedule, which is the documented mid-range gap vs pocketfft (see
		// docs/perf.md and iterative.go). Route every power of two here.
		p.it = newITPlan(n)
		return p
	}
	if factorsAreSmall(n) {
		// Smooth length: the iterative Stockham passes (stockham.go) replaced the
		// recursive Cooley–Tukey engine here, 1.5–1.9× faster at N=1000…1920 on
		// the benchmark host; mixedradix.go stays as an independent test oracle.
		p.sk = newSKPlan(n)
		return p
	}
	// A prime whose n-1 is 7-smooth goes to Rader (a cyclic convolution at
	// length exactly n-1); every other length with a large prime factor goes to
	// Bluestein (a linear convolution at a smooth length >= 2n-1). Measured on
	// arm64 and amd64 for every prime up to 6000, that split picks the faster
	// engine in all but a handful of cases — see rader.go.
	if isPrime(n) {
		if _, ok := convCost(n - 1); ok {
			p.rader = newRaderPlan(n)
			return p
		}
	}
	p.bluestein = newBluesteinPlan(n)
	return p
}

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
	case p.it != nil:
		p.it.transform(dst, src, inverse)
	case p.sk != nil:
		p.sk.transform(dst, src, inverse)
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

// nextSmoothConv returns the smallest integer >= lo whose only prime factors are
// 2, 3 and 5. Such a length is handled entirely by the mixed-radix engine's
// specialized radix-2/3/4/5 butterflies (no slow general radix-p step) and is
// substantially smaller than the next power of two for the convolution sizes the
// prime engines pad to (~0.6× at N≈10⁴), so it is the cheapest linear-convolution
// length. lo >= 1.
//
// It is retained as the lower-bound helper bestConvLen searches from (and is the
// reference the radix7 test pins). The Rader/Bluestein convolution length itself
// is chosen by bestConvLen, which also admits radix-7 lengths and ranks
// candidates by a measured cost model rather than just taking the smallest.
func nextSmoothConv(lo int) int {
	if lo < 1 {
		lo = 1
	}
	for m := lo; ; m++ {
		x := m
		for x%2 == 0 {
			x /= 2
		}
		for x%3 == 0 {
			x /= 3
		}
		for x%5 == 0 {
			x /= 5
		}
		if x == 1 {
			return m
		}
	}
}

// convCost estimates the relative wall-clock cost of one Stockham FFT of a
// 7-smooth length m. It returns (cost, ok); ok is false when m is not 7-smooth
// (a prime factor above 7 would force the slow general radix pass, never a good
// convolution length). The cost is m·Σ(per-pass weight) over the passes the
// engine actually runs for m — skFactorize's radix-8/4/2/3/5/7 sequence — so the
// model follows the engine's own factorization.
//
// The weights were fitted by least squares (relative error) to the measured
// time of every 7-smooth length in [500, 45000] on two hosts — Apple M4 Max
// (arm64) and Xeon E5-2620 v3 (amd64) — and averaged; each host's own fit
// agrees with the other's within the noise. Scored on the linear-convolution
// windows [2q-1, 1.4·(2q-1)] of 260 primes q in 260..20000, the pick lands on
// average 4.3% (arm64) / 2.6% (amd64) above the measured-fastest length in its
// window, median exactly on it. The weights calibrated for the recursive engine
// this replaced were 26% / 13% above, worse than just taking the smallest
// smooth length (11% / 9%). Measured 2026-09-29.
func convCost(m int) (float64, bool) {
	x := m
	for _, p := range []int{2, 3, 5, 7} {
		for x%p == 0 {
			x /= p
		}
	}
	if x != 1 {
		return 0, false
	}
	cost := 0.0
	for _, r := range skFactorize(m) {
		cost += passWeight[r]
	}
	return float64(m) * cost, true
}

// passWeight is the fitted relative cost per point of one Stockham pass of
// each radix (see convCost), normalized to radix 4.
var passWeight = map[int]float64{8: 1.51, 4: 1.0, 2: 1.11, 3: 1.05, 5: 1.44, 7: 1.93}

// bestConvLen returns the cheapest 7-smooth convolution length >= lo for
// Bluestein's linear convolution. It scans the window [lo, ⌈1.4·lo⌉]
// of 7-smooth candidates and returns the one minimizing convCost. The lower
// bound lo is the linear-convolution minimum (2N-1 for a length-N transform); the 1.4× window is wide enough to always contain a highly
// 2/4-composite length yet narrow enough that a larger-but-smoother length never
// out-costs the small dense ones. lo >= 1.
//
// The window is guaranteed non-empty: the largest ratio between consecutive
// 7-smooth integers is < 1.4 for every lo >= 1 (verified exhaustively up to far
// beyond any convolution length the prime engines produce), so the loop always
// finds at least one candidate and best is always assigned.
//
// Ranking by the cost model rather than taking the smallest smooth length is
// what the measurements behind convCost support: over 260 prime windows the
// smallest length was 9–11% slower than the fastest on average, the cost-model
// pick 3–4%.
func bestConvLen(lo int) int {
	if lo < 1 {
		lo = 1
	}
	hi := lo + (lo*4+9)/10 // ⌈1.4·lo⌉ upper bound for the candidate window
	best := lo
	bestCost, _ := convCost(lo)
	haveBest := false
	for m := lo; m <= hi; m++ {
		c, ok := convCost(m)
		if !ok {
			continue
		}
		if !haveBest || c < bestCost {
			best, bestCost, haveBest = m, c, true
		}
	}
	return best
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
//
// The plan is built *outside* the cache lock: NewPlan for a prime length itself
// re-enters cachedPlan to build the power-of-two/smooth convolution sub-plan its
// Rader/Bluestein engine uses, so holding the lock across NewPlan would
// self-deadlock. Building unlocked admits a benign race where two goroutines
// construct the same length concurrently; plans are immutable and identical, so
// either may win the store with no observable difference.
func cachedPlan(n int) *Plan {
	planMu.Lock()
	p, ok := planCache[n]
	planMu.Unlock()
	if ok {
		return p
	}
	p = NewPlan(n)
	planMu.Lock()
	if existing, ok := planCache[n]; ok {
		p = existing
	} else {
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
