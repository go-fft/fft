package fft

import (
	"math"
	"sync"
)

// Rader's algorithm for a prime length N.
//
// For prime N the DFT of the N-1 non-DC outputs is a cyclic convolution of
// length N-1. Picking a primitive root g of the multiplicative group mod N, the
// input and output indices 1..N-1 are reordered by the powers of g; in that
// ordering the size-N DFT becomes
//
//	X[g^{-q}] = x[0] + sum_p x[g^p] · W_N^{g^{p-q}}
//
// i.e. X (minus its DC term) equals the cyclic convolution of the permuted
// input with the permuted roots W_N^{g^k}. The convolution is evaluated
// directly at length q = N-1 with the library's own FFTs: a ⊛ ker =
// IFFT_q(FFT_q(a) · FFT_q(ker)). The kernel spectrum (pre-scaled by 1/q, so the
// inverse needs no normalization pass) and both permutations are precomputed
// once into the plan, so a transform costs two length-q FFTs, a pointwise
// product and the permuted gather/scatter — no per-call trig.
//
// NewPlan routes a prime here only when q is 7-smooth, which is when this wins.
// Measured against Bluestein for every prime up to 6000 on arm64 (M4 Max) and
// amd64 (Xeon E5-2620 v3), 2026-09-29: with q 7-smooth, Rader was faster for
// every prime on amd64 and for all but a handful on arm64 (median Bluestein
// time 1.4–1.9× Rader's); with a factor 11 or 13 in q it lost every time; and
// with q not smooth — the linear-convolution Rader this file used to carry,
// zero-padded to a smooth length >= 2q — it lost to Bluestein at the same pad
// in all but 15 of about 1000 primes (median ~5–10%), because its permuted
// gather/scatter costs more than Bluestein's contiguous chirp passes. That path
// is gone: those primes go to Bluestein.
type raderPlan struct {
	n     int
	perm  []int        // perm[p] = g^p mod N, p = 0 .. N-2 (input gather order)
	iperm []int        // iperm[q] = g^{-q} mod N, q = 0 .. N-2 (output scatter order)
	bF    []complex128 // FFT of the forward convolution kernel / q, length q
	bI    []complex128 // FFT of the inverse convolution kernel / q, length q

	// scratch lends each concurrent transform its length-q convolution
	// buffer, so a steady-state transform allocates nothing.
	scratch sync.Pool
}

// newRaderPlan builds a Rader plan for prime n (n >= 3). The caller guarantees
// primality and that n-1 is a length the mixed-radix engine handles directly
// (factorsAreSmall); NewPlan further restricts it to a 7-smooth n-1.
func newRaderPlan(n int) *raderPlan {
	g := primitiveRoot(n)
	q := n - 1
	p := &raderPlan{n: n}

	p.perm = make([]int, q)
	p.iperm = make([]int, q)
	x := 1
	for k := 0; k < q; k++ {
		p.perm[k] = x
		x = x * g % n
	}
	// g^{-q} = g^{(n-1)-q} for q in 1..n-1; iperm[q] = perm[(n-1-q) mod (n-1)].
	for qi := 0; qi < q; qi++ {
		p.iperm[qi] = p.perm[(q-qi)%q]
	}

	p.bF = p.buildKernel(n, false)
	p.bI = p.buildKernel(n, true)
	p.scratch.New = func() any { b := make([]complex128, q); return &b }
	return p
}

// buildKernel forms the convolution kernel ker[m] = W_N^{g^{-m}}, m = 0 .. q-1
// (forward sign, or its conjugate for the inverse) and returns its length-q FFT
// scaled by 1/q. The length-q cyclic convolution result[qi] = sum_p
// a[p]·ker[(qi-p) mod q] is the Rader correlation.
func (p *raderPlan) buildKernel(n int, inverse bool) []complex128 {
	q := n - 1
	b := make([]complex128, q)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	// ker[m] = W_N^{g^{-m}} = W_N^{perm[(q-m) mod q]}.
	for m := 0; m < q; m++ {
		e := p.perm[(q-m)%q]
		ang := sign * 2 * math.Pi * float64(e) / float64(n)
		b[m] = complex(math.Cos(ang), math.Sin(ang))
	}
	cachedPlan(q).FFT(b, b)
	inv := complex(1/float64(q), 0)
	for i := range b {
		b[i] *= inv
	}
	return b
}

// transform writes the unnormalized length-n DFT of src into dst via Rader's
// convolution. dst may alias src.
func (p *raderPlan) transform(dst, src []complex128, inverse bool) {
	n, q := p.n, p.n-1
	bSpec := p.bF
	if inverse {
		bSpec = p.bI
	}

	// DC bin: X[0] = sum of all inputs. Capture x[0] before any aliasing write.
	x0 := src[0]
	var sum complex128
	for i := 0; i < n; i++ {
		sum += src[i]
	}

	// Gather the permuted inputs a[p] = x[g^p]; every one of the q entries is
	// written, so the pooled buffer needs no clearing.
	bp := p.scratch.Get().(*[]complex128)
	defer p.scratch.Put(bp)
	a := (*bp)[:q]
	perm := p.perm[:q]
	for k, j := range perm {
		a[k] = src[j]
	}

	// Cyclic convolution a ⊛ kernel via FFTs, in place: IFFT(FFT(a)·bSpec),
	// with the 1/q carried by bSpec.
	plan := cachedPlan(q)
	plan.execute(a, a, false)
	bSpec = bSpec[:q]
	for i := range a {
		a[i] *= bSpec[i]
	}
	plan.execute(a, a, true)

	dst[0] = sum
	iperm := p.iperm[:q]
	for qi, j := range iperm {
		dst[j] = x0 + a[qi]
	}
}

// primitiveRoot returns the smallest primitive root g of the prime p (a
// generator of the multiplicative group Z/pZ*). p must be a prime; for the only
// even prime, 2, the group is trivial and the generator is 1. A primitive root
// is guaranteed to exist for every prime, so the search always succeeds.
func primitiveRoot(p int) int {
	phi := p - 1
	factors := primeFactorsDistinct(phi)
	if len(factors) == 0 {
		// phi == 1, i.e. p == 2: the group {1} is trivial, generator 1.
		return 1
	}
	for g := 2; ; g++ {
		isGenerator := true
		for _, f := range factors {
			if modPow(g, phi/f, p) == 1 {
				isGenerator = false
				break
			}
		}
		if isGenerator {
			return g
		}
	}
}

// primeFactorsDistinct returns the distinct prime factors of m (m >= 1).
func primeFactorsDistinct(m int) []int {
	var f []int
	for d := 2; d*d <= m; d++ {
		if m%d == 0 {
			f = append(f, d)
			for m%d == 0 {
				m /= d
			}
		}
	}
	if m > 1 {
		f = append(f, m)
	}
	return f
}

// modPow returns base^exp mod m by binary exponentiation.
func modPow(base, exp, m int) int {
	result := 1
	base %= m
	for exp > 0 {
		if exp&1 == 1 {
			result = result * base % m
		}
		exp >>= 1
		base = base * base % m
	}
	return result
}

// isPrime reports whether n is prime (trial division; n only ever a transform
// length here, so this is not on any hot path).
func isPrime(n int) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for d := 3; d*d <= n; d += 2 {
		if n%d == 0 {
			return false
		}
	}
	return true
}
