package fft

// Iterative, cache-friendly power-of-two FFT engine.
//
// The split-radix engine (splitradix.go) already matches pocketfft's *operation
// count* in the mid-range, but loses on the *memory schedule*: a recursive
// kernel revisits the data through a deep call tree, and the compiler cannot keep
// a working set resident across recursion frames. pocketfft instead runs an
// iterative kernel — a single bit-reversal permutation followed by log-N
// butterfly stages, each one sequential pass over the array — so the hot data and
// the twiddles stream linearly and stay cache-resident.
//
// This file ports that schedule to pure Go:
//
//   - A precomputed bit-reversal permutation reorders the input once.
//   - Decimation-in-time stages then double the sub-transform length each pass.
//     A radix-4 stage advances the length ×4 with one twiddle triple per group;
//     a single radix-2 stage handles the leftover factor of two when log2(n) is
//     odd. Radix-4 halves the number of stages (passes over memory) versus
//     radix-2, which is the whole point on a memory-bound kernel.
//   - Each stage's twiddles are laid out contiguously in the exact order the
//     stage consumes them (stagePlan.tw), so the twiddle read is a linear scan
//     instead of a strided gather into the size-n root table.
//
// A note on blocking: an explicit inner-loop block (sweeping butterfly positions
// in L1-sized chunks across groups) was implemented and measured — and it *lost*
// to leaving each stage as one long contiguous inner loop, at every size and most
// at the large ones (e.g. N=65536: ~450 µs unblocked vs ~480 µs blocked). The gc
// autovectorizer extracts more from the simple long loop than the fragmented
// blocked one, the same lesson the SIMD round taught (docs/perf.md). So the win
// here is the iterative *schedule* — one linear pass per stage, radix-4 to halve
// the pass count, twiddles pre-laid for sequential reads — not manual blocking.
//
// The recursive split-radix kernel is kept; plan.go routes each power-of-two
// length to whichever of the two measured faster for that size (see docs/perf.md
// and the size-routing in NewPlan). The iterative kernel measured faster at every
// power-of-two length on the benchmark host, so NewPlan routes all of them here.

// stagePlan is one decimation-in-time stage of the iterative kernel: a radix-2 or
// radix-4 pass with its twiddles pre-laid-out for a sequential read.
type stagePlan struct {
	radix int          // 2 or 4
	span  int          // sub-transform length entering the stage (the "m" below)
	tw    []complex128 // forward twiddles, contiguous in consumption order
	twC   []complex128 // conjugate (inverse) twiddles, same layout
}

// itPlan is the iterative plan for one power-of-two length.
type itPlan struct {
	n      int
	revPos []int       // bit-reversal permutation: out[i] = in[revPos[i]]
	stages []stagePlan // ordered DIT stages, product of radices == n
}

// newITPlan builds the bit-reversal permutation and the per-stage twiddle layout
// for a power-of-two length n (n >= 2, a power of two — guaranteed by the caller).
func newITPlan(n int) *itPlan {
	p := &itPlan{n: n}

	// Decompose log2(n) into radix-4 stages plus an optional leading radix-2 when
	// the exponent is odd. A leading radix-2 (done first, on the shortest spans)
	// keeps every later stage a clean radix-4.
	log2 := 0
	for (1 << log2) < n {
		log2++
	}
	radices := make([]int, 0, log2)
	if log2%2 == 1 {
		radices = append(radices, 2)
	}
	for i := 0; i < log2/2; i++ {
		radices = append(radices, 4)
	}

	p.revPos = bitReversalForRadices(n, radices)
	p.stages = buildStages(n, radices)
	return p
}

// bitReversalForRadices computes the input permutation for a decimation-in-time
// schedule whose stages use the given ordered radices (read first-to-last as the
// digits of the mixed-radix index, least-significant first). out[i] gathers
// in[revPos[i]]. Because the stages are applied in this order, the permutation is
// the mixed-radix digit reversal of the index.
func bitReversalForRadices(n int, radices []int) []int {
	rev := make([]int, n)
	for i := 0; i < n; i++ {
		x := i
		r := 0
		// Reverse the mixed-radix digits: the first stage consumes the
		// least-significant digit, so reading digits low-to-high here and packing
		// them high-to-low yields the gather index.
		for s := 0; s < len(radices); s++ {
			rad := radices[s]
			r = r*rad + x%rad
			x /= rad
		}
		rev[i] = r
	}
	return rev
}

// buildStages lays out the per-stage twiddle tables in consumption order for the
// given radix schedule. span starts at 1 (the leaf sub-transform length) and is
// multiplied by each stage's radix.
func buildStages(n int, radices []int) []stagePlan {
	root := twiddleTable(n) // size-n forward roots exp(-2πi k/n)
	stages := make([]stagePlan, 0, len(radices))
	span := 1
	for _, rad := range radices {
		next := span * rad
		st := stagePlan{radix: rad, span: span}
		switch rad {
		case 2:
			// One twiddle per butterfly position k in [0, span): W_next^k.
			st.tw = make([]complex128, span)
			st.twC = make([]complex128, span)
			for k := 0; k < span; k++ {
				w := root[(k*(n/next))%n]
				st.tw[k] = w
				st.twC[k] = complex(real(w), -imag(w))
			}
		case 4:
			// Three twiddles per butterfly position k: W^k, W^{2k}, W^{3k} over the
			// length-next transform, stored as consecutive triples for a linear read.
			st.tw = make([]complex128, span*3)
			st.twC = make([]complex128, span*3)
			step := n / next
			for k := 0; k < span; k++ {
				w1 := root[(k*step)%n]
				w2 := root[(2*k*step)%n]
				w3 := root[(3*k*step)%n]
				st.tw[k*3+0] = w1
				st.tw[k*3+1] = w2
				st.tw[k*3+2] = w3
				st.twC[k*3+0] = complex(real(w1), -imag(w1))
				st.twC[k*3+1] = complex(real(w2), -imag(w2))
				st.twC[k*3+2] = complex(real(w3), -imag(w3))
			}
		}
		stages = append(stages, st)
		span = next
	}
	return stages
}

// transform writes the unnormalized DFT of src into dst. dst may alias src. The
// permutation gathers src into dst, then the stages run in place on dst.
func (p *itPlan) transform(dst, src []complex128, inverse bool) {
	if aliases(dst, src) {
		// In-place call: the gather dst[i]=src[j] would overwrite source samples
		// before they are read (the digit-reversal permutation is not an
		// involution in general), so read through a private copy.
		src = append([]complex128(nil), src...)
	}
	rev := p.revPos
	for i, j := range rev {
		dst[i] = src[j]
	}
	p.run(dst, inverse)
}

// aliases reports whether two slices share backing storage at the same start,
// i.e. an in-place transform where dst == src.
func aliases(a, b []complex128) bool {
	return len(a) > 0 && len(b) > 0 && &a[0] == &b[0]
}

// transformScratch is transform with a caller-supplied source buffer that the
// permutation reads (so a caller owning a private length-n buffer skips a copy,
// matching srPlan.transformScratch). dst must not alias scratch.
func (p *itPlan) transformScratch(dst, scratch []complex128, inverse bool) {
	rev := p.revPos
	for i, j := range rev {
		dst[i] = scratch[j]
	}
	p.run(dst, inverse)
}

// run applies the DIT stages in place on a bit-reversed buffer a.
func (p *itPlan) run(a []complex128, inverse bool) {
	n := p.n
	for si := range p.stages {
		st := &p.stages[si]
		tw := st.tw
		if inverse {
			tw = st.twC
		}
		if st.radix == 2 {
			radix2Stage(a, n, st.span, tw)
		} else {
			radix4Stage(a, n, st.span, tw, inverse)
		}
	}
}

// radix2Stage runs one radix-2 DIT pass in place. span is the sub-transform
// length entering the stage; groups of 2*span are combined, each butterfly k in
// [0, span) using tw[k] = W_{2*span}^k.
func radix2Stage(a []complex128, n, span int, tw []complex128) {
	step := 2 * span
	for base := 0; base < n; base += step {
		for k := 0; k < span; k++ {
			i := base + k
			j := i + span
			t := a[j] * tw[k]
			u := a[i]
			a[i] = u + t
			a[j] = u - t
		}
	}
}

// radix4Stage runs one radix-4 DIT pass in place. span is the sub-transform
// length entering the stage; groups of 4*span are combined. tw holds the
// per-position twiddle triples (W^k, W^{2k}, W^{3k}). Each group is a single
// contiguous inner loop over the butterfly positions, which the gc autovectorizer
// schedules better than a manually blocked variant (measured — see the file
// header).
func radix4Stage(a []complex128, n, span int, tw []complex128, inverse bool) {
	step := 4 * span
	for base := 0; base < n; base += step {
		for k := 0; k < span; k++ {
			i0 := base + k
			i1 := i0 + span
			i2 := i1 + span
			i3 := i2 + span
			ti := k * 3
			b1 := a[i1] * tw[ti]
			b2 := a[i2] * tw[ti+1]
			b3 := a[i3] * tw[ti+2]
			a0 := a[i0]
			// Radix-4 butterfly (DIT). t0,t1 even half; t2,t3 odd half rotated ∓i.
			t0 := a0 + b2
			t1 := a0 - b2
			t2 := b1 + b3
			t3 := rotNeg90(b1-b3, inverse)
			a[i0] = t0 + t2
			a[i1] = t1 + t3
			a[i2] = t0 - t2
			a[i3] = t1 - t3
		}
	}
}
