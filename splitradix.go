package fft

// Split-radix engine for power-of-two lengths.
//
// The split-radix FFT is the classic low-operation-count power-of-two kernel:
// it computes the same DFT as radix-2/radix-4 Cooley–Tukey but with roughly a
// third fewer real multiplies, because it decimates the transform into one
// half-length DFT (the even-indexed samples) and two quarter-length DFTs (the
// indices ≡1 and ≡3 mod 4), recombined with an L-shaped butterfly that needs
// only the twiddles W_N^k and W_N^{3k}. This is pocketfft's mid-range advantage
// over a plain mixed-radix engine, so go-fft routes every pure power of two
// here in place of the general ctPlan path.
//
// Like ctPlan it amortizes all trig: the size-n forward and conjugate root
// tables are precomputed once, and the recursion reads the source through an
// index map (stride/offset) so it never bit-reverses or copies between levels
// beyond the single read-only scratch the public transform makes.

// srPlan is the split-radix Cooley–Tukey plan for one power-of-two length.
type srPlan struct {
	n      int          // transform length, a power of two
	tw     []complex128 // forward roots exp(-2πi·k/n), k = 0 .. n-1
	twConj []complex128 // conjugate (inverse) roots, precomputed once
}

// newSRPlan precomputes the size-n forward and conjugate twiddle tables for a
// power-of-two length n (n >= 2). The caller guarantees n is a power of two.
func newSRPlan(n int) *srPlan {
	tw := twiddleTable(n)
	conj := make([]complex128, n)
	for i, w := range tw {
		conj[i] = complex(real(w), -imag(w))
	}
	return &srPlan{n: n, tw: tw, twConj: conj}
}

// roots returns the forward or conjugate twiddle table for the requested
// direction.
func (p *srPlan) roots(inverse bool) []complex128 {
	if inverse {
		return p.twConj
	}
	return p.tw
}

// transform writes the unnormalized DFT of src into dst. When inverse is true
// the conjugate roots are used (the caller applies 1/N). dst may alias src; the
// recursion reads src through an index map and never writes src.
func (p *srPlan) transform(dst, src []complex128, inverse bool) {
	n := p.n
	scratch := make([]complex128, n)
	copy(scratch, src)
	p.rec(dst, scratch, n, 0, 1, p.roots(inverse), inverse)
}

// rec computes a length-len split-radix DFT of the sub-sequence that starts at
// src[off] and advances by stride, writing the result contiguously into
// out[0:len]. tw is the active size-n root table; step = n/len maps a size-len
// exponent k to the size-n table index k·step. inverse selects the sign of the
// quarter-rotation in the L-butterfly.
//
// The decimation is the textbook split-radix split:
//   - the even-indexed samples form one length-len/2 sub-DFT (E),
//   - the ≡1 mod 4 samples one length-len/4 sub-DFT (O1),
//   - the ≡3 mod 4 samples one length-len/4 sub-DFT (O3),
//
// recombined over k in [0,len/4) by the L-shaped butterfly below.
func (p *srPlan) rec(out, src []complex128, length, off, stride int, tw []complex128, inverse bool) {
	// length is always >= 2 and a power of two: the top-level call passes n >= 2,
	// and the recursion descends only to the length-2 and length-4 leaves below
	// (a length-4 parent is the radix-4 leaf, so neither the even half nor either
	// quarter is ever a length-1 sub-DFT).
	switch length {
	case 2:
		a := src[off]
		b := src[off+stride]
		out[0] = a + b
		out[1] = a - b
		return
	case 4:
		// Hardcoded radix-4 DFT leaf: no twiddles (all roots are ±1, ±i), so this
		// terminates the recursion two levels early and removes the deepest, most
		// numerous calls — the dominant overhead of a recursive split-radix.
		a := src[off]
		b := src[off+stride]
		c := src[off+stride*2]
		d := src[off+stride*3]
		t0 := a + c
		t1 := a - c
		t2 := b + d
		t3 := rotNeg90(b-d, inverse)
		out[0] = t0 + t2
		out[1] = t1 + t3
		out[2] = t0 - t2
		out[3] = t1 - t3
		return
	}

	q := length / 4
	h := length / 2

	// E occupies out[0:h]; O1 and O3 occupy out[h:h+q] and out[h+q:length].
	p.rec(out[:h], src, h, off, stride*2, tw, inverse)
	p.rec(out[h:h+q], src, q, off+stride, stride*4, tw, inverse)
	p.rec(out[h+q:length], src, q, off+stride*3, stride*4, tw, inverse)

	step := p.n / length
	for k := 0; k < q; k++ {
		o1 := out[h+k] * tw[k*step]
		o3 := out[h+q+k] * tw[3*k*step]
		// sum = W^k·O1 + W^{3k}·O3 ; dif = W^k·O1 − W^{3k}·O3.
		sum := o1 + o3
		dif := o1 - o3
		// The N/4-shifted outputs apply −i·dif (forward) / +i·dif (inverse).
		rdif := rotNeg90(dif, inverse)

		e0 := out[k]
		e1 := out[k+q]
		out[k] = e0 + sum
		out[k+h] = e0 - sum
		out[k+q] = e1 + rdif
		out[k+q+h] = e1 - rdif
	}
}
