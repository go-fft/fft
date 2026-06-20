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
//
// Convolution length. The Rader convolution is *cyclic* of length q = N-1. When
// q itself is a length the library transforms cheaply (all prime factors small,
// so mixed-radix Cooley–Tukey applies — see factorsAreSmall), the cyclic
// convolution is evaluated directly at length q: a ⊛ ker = IFFT_q(FFT_q(a) ·
// FFT_q(ker)). This is FFTW's tuned Rader path and avoids the zero-pad to a
// power of two >= 2q-1, which for these primes is 2–3.3× larger than q itself
// (e.g. N=769: q=768=2⁸·3 vs a 2048-point pad; N=2017: q=2016=2⁵·3²·7 vs 4096).
// When q is *not* smooth (its largest prime factor exceeds maxRadix, e.g.
// N=9973 has q=9972=2²·3²·277) the direct length-q transform would itself be a
// Bluestein/Rader call, so the plan falls back to the classic linear
// convolution: zero-pad to the smallest power of two cl >= 2q-1 and read the
// cyclic result out of the linear convolution's wrap window.
type raderPlan struct {
	n      int
	cl     int          // convolution length (q if cyclic, else pow2 >= 2q-1)
	cyclic bool         // true: direct length-q cyclic conv; false: padded linear
	perm   []int        // perm[p] = g^p mod N, p = 0 .. N-2 (input gather order)
	iperm  []int        // iperm[q] = g^{-q} mod N, q = 0 .. N-2 (output scatter order)
	bF     []complex128 // FFT of the forward convolution kernel, length cl
	bI     []complex128 // FFT of the inverse convolution kernel, length cl
}

// newRaderPlan builds a Rader plan for prime n (n >= 3). The caller guarantees
// primality; isPrime/factorsAreSmall route composites elsewhere.
func newRaderPlan(n int) *raderPlan {
	g := primitiveRoot(n)
	q := n - 1
	p := &raderPlan{n: n}

	// Prefer the direct length-q cyclic convolution when q is smooth enough for
	// the mixed-radix engine; otherwise zero-pad for the classic linear
	// convolution to the smallest highly-composite (2·3·5-smooth) length >= 2q-1.
	// A 2·3·5-smooth pad rides the specialized radix-2/3/4/5 butterflies and is
	// ~0.6× the size of the next power of two for these primes (e.g. N=9973:
	// 20000 = 2⁵·5⁴ vs a 32768-point pad), so the linear convolution is markedly
	// cheaper than the historical power-of-two pad.
	if factorsAreSmall(q) {
		p.cyclic = true
		p.cl = q
	} else {
		// The linear convolution spans indices 0..2q-2; the cyclic value is read
		// from the wrap window at qi+q (max index 2q-1), so the buffer must hold at
		// least 2q points (>= 2q-1 alone leaves the last window read out of range).
		p.cl = nextSmoothConv(2 * q)
	}

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
	return p
}

// buildKernel forms the convolution kernel from the roots W_N^{g^k} (forward
// sign) or their conjugates (inverse) and returns its FFT at length cl.
//
// For the cyclic path the kernel is ker[m] = W_N^{g^{-m}}, m = 0 .. q-1: the
// length-q cyclic convolution result[qi] = sum_p a[p]·ker[(qi-p) mod q] is the
// Rader correlation. For the padded-linear path the kernel is the same cyclic
// kernel replicated over indices 0 .. 2q-1, so a linear convolution of length cl
// (>= 2q) recovers the cyclic value in its wrap window: result[qi] = conv[qi+q].
func (p *raderPlan) buildKernel(n int, inverse bool) []complex128 {
	q := n - 1
	b := make([]complex128, p.cl)
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	if p.cyclic {
		// ker[m] = W_N^{g^{-m}} = W_N^{perm[(q-m) mod q]}.
		for m := 0; m < q; m++ {
			e := p.perm[(q-m)%q]
			ang := sign * 2 * math.Pi * float64(e) / float64(n)
			b[m] = complex(math.Cos(ang), math.Sin(ang))
		}
	} else {
		// Replicate the cyclic kernel over 0..2q-1 (kerC[s] = W_N^{g^{-s}}).
		for m := 0; m < 2*q; m++ {
			s := m % q
			e := p.perm[(q-s)%q]
			ang := sign * 2 * math.Pi * float64(e) / float64(n)
			b[m] = complex(math.Cos(ang), math.Sin(ang))
		}
	}
	return cachedPlan(p.cl).FFT(make([]complex128, p.cl), b)
}

// transform writes the unnormalized length-n DFT of src into dst via Rader's
// convolution. dst may alias src.
func (p *raderPlan) transform(dst, src []complex128, inverse bool) {
	n, cl, q := p.n, p.cl, p.n-1
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

	// Gather the permuted inputs a[p] = x[g^p] (zero-padded to cl in the linear
	// path; the cyclic path uses exactly q == cl entries).
	a := make([]complex128, cl)
	for k := 0; k < q; k++ {
		a[k] = src[p.perm[k]]
	}

	// Cyclic convolution a ⊛ kernel via FFTs: a = IFFT(FFT(a)·bSpec).
	plan := cachedPlan(cl)
	A := make([]complex128, cl)
	plan.FFT(A, a)
	for i := range A {
		A[i] *= bSpec[i]
	}
	conv := make([]complex128, cl)
	plan.IFFT(conv, A)

	dst[0] = sum
	if p.cyclic {
		// Direct length-q cyclic convolution: result[qi] = conv[qi].
		for qi := 0; qi < q; qi++ {
			dst[p.iperm[qi]] = x0 + conv[qi]
		}
		return
	}
	// Padded linear convolution: the cyclic value is in the wrap window at qi+q.
	for qi := 0; qi < q; qi++ {
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
