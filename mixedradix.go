package fft

// Mixed-radix Cooley–Tukey engine.
//
// A length N whose prime factors are all small (<= maxRadix) is transformed by
// recursively decimating in time. At each level the leading factor p of the
// remaining length splits the sub-transform into p interleaved sequences of
// length m = len/p; each is transformed recursively, then recombined by a
// radix-p butterfly that multiplies in the size-N roots of unity (twiddles).
//
// Specialized straight-line butterflies handle the common radices 2, 3, 4 and
// 5 (fewer multiplies, no inner loop); any other small prime factor uses a
// general radix-p butterfly. All trig is precomputed once into the plan's
// twiddle table, so transforms cost no sin/cos at run time.

// ctPlan is the mixed-radix Cooley–Tukey plan for one length.
type ctPlan struct {
	n       int          // transform length
	factors []int        // ordered radices, product == n
	tw      []complex128 // forward roots exp(-2πi·k/n), k = 0 .. n-1
	twConj  []complex128 // conjugate (inverse) roots, precomputed once
}

// newCTPlan factors n and precomputes the size-n forward and conjugate twiddle
// tables. Holding the conjugate table too lets every butterfly index the active
// table directly — no per-element conjugate branch, and no modulo, since the
// specialized radix-2/3/4/5 kernels only ever form indices < n.
func newCTPlan(n int) *ctPlan {
	tw := twiddleTable(n)
	conj := make([]complex128, n)
	for i, w := range tw {
		conj[i] = complex(real(w), -imag(w))
	}
	return &ctPlan{
		n:       n,
		factors: factorize(n),
		tw:      tw,
		twConj:  conj,
	}
}

// roots returns the forward or conjugate twiddle table for the requested
// direction, so the butterflies pick once per call instead of branching per
// element.
func (p *ctPlan) roots(inverse bool) []complex128 {
	if inverse {
		return p.twConj
	}
	return p.tw
}

// transform writes the unnormalized DFT of src into dst. When inverse is true
// the conjugate roots are used (the caller applies 1/N). dst may alias src; the
// recursion reads src through an index map and never writes src.
func (p *ctPlan) transform(dst, src []complex128, inverse bool) {
	n := p.n
	scratch := make([]complex128, n)
	// Copy src so aliasing dst==src is safe and the recursion has a stable
	// read-only source.
	copy(scratch, src)
	p.rec(dst, scratch, n, 1, p.factors, p.roots(inverse), inverse)
}

// rec computes a length-len DFT of the sub-sequence src[0], src[stride],
// src[2·stride], … and writes it contiguously into out[0:len]. factors lists
// the remaining radices whose product is len (factors[0] is applied at this
// level). The size-n twiddle table is indexed with the global step len/n
// scaled appropriately at each level via the stride/len relationship.
func (p *ctPlan) rec(out, src []complex128, length, stride int, factors []int, tw []complex128, inverse bool) {
	// length is always > 1 here: the top-level call passes n > 1, and the
	// recursion only descends with length = m when m > 1 (the m == 1 leaf is
	// handled inline below), so a length-1 sub-DFT is never reached.
	r := factors[0]
	m := length / r

	if m == 1 {
		// Leaf level: the r sub-transforms are single samples, so gather the
		// strided inputs straight into out and let the butterfly do the size-r
		// DFT. This saves the r trivial length-1 recursive calls that dominate
		// the call count of a deep power-of-two factorization.
		for i := 0; i < r; i++ {
			out[i] = src[i*stride]
		}
	} else {
		// Transform the r interleaved sub-sequences of length m. Sub-sequence i
		// starts at offset i·stride and has step r·stride within src; its result
		// lands in out[i·m : (i+1)·m].
		for i := 0; i < r; i++ {
			p.rec(out[i*m:(i+1)*m], src[i*stride:], m, r*stride, factors[1:], tw, inverse)
		}
	}

	// Combine: a radix-r butterfly over the m groups. The twiddle for the
	// (j, i) term is the size-len root raised to (i·j); since the plan holds
	// size-n roots, index it with step n/len.
	p.butterfly(out, length, r, m, stride, tw, inverse)
}

// butterfly applies the radix-r recombination in place on out[0:length], where
// out currently holds the r sub-transforms laid out as r contiguous blocks of
// length m. step = n/length maps a size-len exponent to the size-n twiddle
// table. Dispatch picks a specialized kernel for r in {2,3,4,5}.
func (p *ctPlan) butterfly(out []complex128, length, r, m, stride int, tw []complex128, inverse bool) {
	step := p.n / length
	switch r {
	case 2:
		p.radix2(out, m, step, tw)
	case 3:
		p.radix3(out, m, step, tw, inverse)
	case 4:
		p.radix4(out, m, step, tw, inverse)
	case 5:
		p.radix5(out, m, step, tw, inverse)
	default:
		p.radixP(out, length, r, m, step, tw)
	}
}

// radix2 recombines two length-m sub-transforms stored as out[0:m] and
// out[m:2m] into the length-2m result, in place. tw is the active (forward or
// conjugate) size-n root table; the index k·step is always < n here, so it is
// read directly with no modulo or per-element conjugate branch.
func (p *ctPlan) radix2(out []complex128, m, step int, tw []complex128) {
	for k := 0; k < m; k++ {
		w := tw[k*step]
		a := out[k]
		b := out[k+m] * w
		out[k] = a + b
		out[k+m] = a - b
	}
}

// radix4 recombines four length-m sub-transforms into the length-4m result, in
// place, using a straight-line radix-4 butterfly (the i factor is applied by a
// 90° rotation rather than a complex multiply). tw is the active root table.
func (p *ctPlan) radix4(out []complex128, m, step int, tw []complex128, inverse bool) {
	for k := 0; k < m; k++ {
		ks := k * step
		w1 := tw[ks]
		w2 := tw[2*ks]
		w3 := tw[3*ks]
		a := out[k]
		b := out[k+m] * w1
		c := out[k+2*m] * w2
		d := out[k+3*m] * w3
		// DFT-4 of (a,b,c,d). For the forward transform the rotation is -i; for
		// the inverse it is +i. rot(z) multiplies z by -i (clockwise).
		t0 := a + c
		t1 := a - c
		t2 := b + d
		t3 := b - d
		t3r := rotNeg90(t3, inverse)
		out[k] = t0 + t2
		out[k+m] = t1 + t3r
		out[k+2*m] = t0 - t2
		out[k+3*m] = t1 - t3r
	}
}

// rotNeg90 multiplies z by -i for the forward transform (and by +i for the
// inverse), the unit rotation a radix-4 butterfly needs without a full complex
// multiply.
func rotNeg90(z complex128, inverse bool) complex128 {
	if inverse {
		return complex(-imag(z), real(z)) // ×(+i)
	}
	return complex(imag(z), -real(z)) // ×(−i)
}

// sin120 = sin(2π/3) = √3/2, the rotation magnitude in the radix-3 butterfly.
const sin120 = 0.8660254037844386467637231707529361834714026269051903140279

// radix3 recombines three length-m sub-transforms into the length-3m result in
// place, with a straight-line radix-3 butterfly (no inner DFT loop): the cube
// roots of unity collapse to one real add and one imaginary rotation.
func (p *ctPlan) radix3(out []complex128, m, step int, tw []complex128, inverse bool) {
	// Rotation sign: forward uses -2π/3 roots, inverse uses +.
	s := -sin120
	if inverse {
		s = sin120
	}
	for k := 0; k < m; k++ {
		ks := k * step
		w1 := tw[ks]
		w2 := tw[2*ks]
		a := out[k]
		b := out[k+m] * w1
		c := out[k+2*m] * w2
		t := b + c
		out[k] = a + t
		// a - t/2 ± i·s·(b-c). With W_3 = -1/2 ∓ i·√3/2.
		u := a - complex(real(t)*0.5, imag(t)*0.5)
		d := b - c
		v := complex(-s*imag(d), s*real(d)) // ±i·s·(b-c)
		out[k+m] = u + v
		out[k+2*m] = u - v
	}
}

// radix5 recombines five length-m sub-transforms into the length-5m result in
// place with a straight-line radix-5 butterfly using the standard four real
// constants (cos/sin of 2π/5 and 4π/5).
func (p *ctPlan) radix5(out []complex128, m, step int, tw []complex128, inverse bool) {
	// W_5^1 = c1 + i·s1, W_5^2 = c2 + i·s2 (forward: negative angles).
	const (
		c1 = 0.30901699437494742410229341718281905886015458990288  // cos(2π/5)
		s1 = 0.95105651629515357211643933337938214340569863412575  // sin(2π/5)
		c2 = -0.80901699437494742410229341718281905886015458990289 // cos(4π/5)
		s2 = 0.58778525229247312916870595463907276859765243764314  // sin(4π/5)
	)
	sg := -1.0
	if inverse {
		sg = 1.0
	}
	for k := 0; k < m; k++ {
		ks := k * step
		a := out[k]
		b := out[k+m] * tw[ks]
		c := out[k+2*m] * tw[2*ks]
		d := out[k+3*m] * tw[3*ks]
		e := out[k+4*m] * tw[4*ks]

		// Symmetric sums/differences for the conjugate root pairs (1,4) and (2,3).
		t1 := b + e
		t2 := b - e
		t3 := c + d
		t4 := c - d
		out[k] = a + t1 + t3
		// Real combinations weighted by the cosines.
		r1 := a + complex(c1*real(t1)+c2*real(t3), c1*imag(t1)+c2*imag(t3))
		r2 := a + complex(c2*real(t1)+c1*real(t3), c2*imag(t1)+c1*imag(t3))
		// Imaginary combinations weighted by the sines; rot multiplies by i.
		i1 := rotI(complex(sg*(s1*real(t2)+s2*real(t4)), sg*(s1*imag(t2)+s2*imag(t4))))
		i2 := rotI(complex(sg*(s2*real(t2)-s1*real(t4)), sg*(s2*imag(t2)-s1*imag(t4))))
		out[k+m] = r1 + i1
		out[k+4*m] = r1 - i1
		out[k+2*m] = r2 + i2
		out[k+3*m] = r2 - i2
	}
}

// rotI multiplies z by i (90° counter-clockwise).
func rotI(z complex128) complex128 {
	return complex(-imag(z), real(z))
}

// radixP recombines r length-m sub-transforms with a general radix-r butterfly
// (r prime, 3 <= r <= maxRadix, used for 3,5,7,11,13). It evaluates, for each
// k in [0,m) and output digit q in [0,r), the size-r DFT of the twiddled inputs
// — O(r^2) per group, which is cheap for small r and avoids special-casing
// every prime.
func (p *ctPlan) radixP(out []complex128, length, r, m, step int, tw []complex128) {
	n := p.n
	buf := make([]complex128, r)
	rstep := n / r
	for k := 0; k < m; k++ {
		// Twiddle the r inputs: in[i] = sub_i[k] · W_n^{i·k·step}. Here i·k·step
		// can reach (r-1)·(m-1)·step < n, but the product is formed modulo n for
		// safety since the general path is not on the pow2 fast lane.
		for i := 0; i < r; i++ {
			buf[i] = out[k+i*m] * tw[i*k*step%n]
		}
		// Output q: sum_i in[i] · W_r^{i·q} = sum_i in[i] · W_n^{i·q·(n/r)}.
		for q := 0; q < r; q++ {
			var sum complex128
			for i := 0; i < r; i++ {
				sum += buf[i] * tw[i*q*rstep%n]
			}
			out[k+q*m] = sum
		}
	}
}
