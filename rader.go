package fft

import "math"

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
// input with the permuted roots W_N^{g^k}. The convolution is evaluated with
// the library's own fast FFTs of length L, where L is the smallest power of two
// >= 2(N-1)-1 so a linear convolution wraps to the length-(N-1) cyclic one. The
// kernel spectrum and both permutations are precomputed once into the plan, so
// a transform costs two length-L FFTs and a pointwise product — no per-call
// trig.
//
// Rader replaces Bluestein for a prime N only above a measured size threshold
// (see raderThreshold); both reduce a prime to the same power-of-two
// convolution, but Rader transforms the length-(N-1) data directly (no chirp
// pre/post multiply and a shorter useful span), which wins on the larger primes.
type raderPlan struct {
	n     int
	l     int          // convolution length, a power of two >= 2(N-1)-1
	perm  []int        // perm[p] = g^p mod N, p = 0 .. N-2 (input gather order)
	iperm []int        // iperm[q] = g^{-q} mod N, q = 0 .. N-2 (output scatter order)
	bF    []complex128 // FFT of the forward convolution kernel, length l
	bI    []complex128 // FFT of the inverse convolution kernel, length l
}

// newRaderPlan builds a Rader plan for prime n (n >= 3). The caller guarantees
// primality; isPrime/factorsAreSmall route composites elsewhere.
func newRaderPlan(n int) *raderPlan {
	g := primitiveRoot(n)
	l := 1
	for l < 2*(n-1)-1 {
		l <<= 1
	}
	p := &raderPlan{n: n, l: l}

	p.perm = make([]int, n-1)
	p.iperm = make([]int, n-1)
	x := 1
	for k := 0; k < n-1; k++ {
		p.perm[k] = x
		x = x * g % n
	}
	// g^{-q} = g^{(n-1)-q} for q in 1..n-1; iperm[q] = perm[(n-1-q) mod (n-1)].
	for q := 0; q < n-1; q++ {
		p.iperm[q] = p.perm[(n-1-q)%(n-1)]
	}

	p.bF = p.buildKernel(n, false)
	p.bI = p.buildKernel(n, true)
	return p
}

// buildKernel forms the length-l cyclic kernel from the roots W_N^{g^k} (forward
// sign) or their conjugates (inverse), padded so a linear convolution of length
// l wraps to the length-(n-1) cyclic convolution, and returns its FFT.
func (p *raderPlan) buildKernel(n int, inverse bool) []complex128 {
	q := n - 1
	b := make([]complex128, p.l)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	// result[qi] = sum_p a[p]·W_N^{g^{(p-qi) mod q}} is the length-q cyclic
	// correlation of a with ker[m] = W_N^{g^m}; written as a convolution it is
	// result[qi] = sum_p a[p]·kerC[(qi-p) mod q] with kerC[s] = ker[(q-s) mod q].
	// Replicating kerC over indices 0..2q-1 makes a linear convolution recover
	// the cyclic one: result[qi] = conv[qi+q] (the read window in transform),
	// whose largest kernel index is 2q-1. l (>= 2q) holds the replicas.
	for m := 0; m < 2*q; m++ {
		s := m % q
		e := p.perm[(q-s)%q] // g^{-s} mod n  (kerC[s])
		ang := sign * 2 * math.Pi * float64(e) / float64(n)
		b[m] = complex(math.Cos(ang), math.Sin(ang))
	}
	return cachedPlan(p.l).FFT(make([]complex128, p.l), b)
}

// transform writes the unnormalized length-n DFT of src into dst via Rader's
// convolution. dst may alias src.
func (p *raderPlan) transform(dst, src []complex128, inverse bool) {
	n, l, q := p.n, p.l, p.n-1
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

	// Gather the permuted inputs a[p] = x[g^p], zero-padded to l.
	a := make([]complex128, l)
	for k := 0; k < q; k++ {
		a[k] = src[p.perm[k]]
	}

	// Cyclic convolution a ⊛ kernel via FFTs: a = IFFT(FFT(a)·bSpec).
	plan := cachedPlan(l)
	A := make([]complex128, l)
	plan.FFT(A, a)
	for i := 0; i < l; i++ {
		A[i] *= bSpec[i]
	}
	conv := make([]complex128, l)
	plan.IFFT(conv, A)

	// The cyclic correlation result[qi] is read from the linear convolution at
	// index qi+q (the kernel was laid out so this window holds the cyclic value).
	dst[0] = sum
	for qi := 0; qi < q; qi++ {
		// Output index is g^{-qi}; add the x[0] term that every non-DC bin gets.
		dst[p.iperm[qi]] = x0 + conv[qi+q]
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
