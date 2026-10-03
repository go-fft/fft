package fft

import (
	"math/cmplx"
	"testing"
)

// TestIterativeAgainstNaive validates the iterative cache-blocked engine
// differentially against the O(N²) DFT oracle at every small power-of-two
// length, exercising both the even (radix-4 only) and odd (leading radix-2)
// log2(n) schedules and the block boundary.
func TestIterativeAgainstNaive(t *testing.T) {
	for _, n := range srSizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		dst := make([]complex128, n)
		it.transform(dst, x, false)
		closeVec(t, dst, naiveDFT(x))
	}
}

// TestIterativeVsSplitRadix cross-checks the iterative engine against the
// independent split-radix engine on the mid-range power-of-two sizes (where the
// naive oracle is too slow), for both directions.
func TestIterativeVsSplitRadix(t *testing.T) {
	sizes := append(append([]int{}, srSizes...), srLargeSizes...)
	sizes = append(sizes, 2048, 16384, 32768)
	for _, n := range sizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		sr := newSRPlan(n)
		for _, inverse := range []bool{false, true} {
			a := make([]complex128, n)
			b := make([]complex128, n)
			it.transform(a, x, inverse)
			sr.transform(b, x, inverse)
			tolN := 1e-12 * float64(n)
			for i := range a {
				if d := cmplx.Abs(a[i] - b[i]); d > tolN {
					t.Fatalf("n=%d inverse=%v index %d: iterative %v vs split-radix %v (|diff|=%g)",
						n, inverse, i, a[i], b[i], d)
				}
			}
		}
	}
}

// TestIterativeRoundTrip checks IFFT(FFT(x)) ≈ x through the iterative engine,
// including the transformScratch entry point.
func TestIterativeRoundTrip(t *testing.T) {
	sizes := append(append([]int{}, srSizes...), srLargeSizes...)
	sizes = append(sizes, 2048)
	for _, n := range sizes {
		x := cmplxSignal(n)
		it := newITPlan(n)
		fwd := make([]complex128, n)
		back := make([]complex128, n)
		scr := make([]complex128, n)
		copy(scr, x)
		it.transformScratch(fwd, scr, false)
		it.transform(back, fwd, true)
		inv := complex(1/float64(n), 0)
		for i := range back {
			back[i] *= inv
		}
		tolN := 1e-12 * float64(n)
		for i := range back {
			if d := cmplx.Abs(back[i] - x[i]); d > tolN {
				t.Fatalf("n=%d index %d: round-trip %v vs %v (|diff|=%g)", n, i, back[i], x[i], d)
			}
		}
	}
}

// TestCacheBlockingIsBitIdentical runs every entry point of the pow2 kernel
// with its leading stages cache-blocked at several block sizes and unblocked
// (one block = the whole array), and requires identical bits: a block is
// transformed by exactly the operations the same points get in a full sweep.
func TestCacheBlockingIsBitIdentical(t *testing.T) {
	saved := itBlock
	defer func() { itBlock = saved }()
	for _, n := range []int{2, 4, 8, 64, 512, 2048, 1 << 13, 1 << 16} {
		x := cmplxSignal(n)
		r := make([]float64, 2*n)
		for i := range r {
			r[i] = real(x[i/2]) - float64(i%3)
		}
		run := func(block int) (fwd, inv, scr, packed []complex128) {
			itBlock = block
			p := newITPlan(n)
			fwd = make([]complex128, n)
			p.transform(fwd, x, false)
			inv = append([]complex128(nil), x...)
			p.transform(inv, inv, true)
			scr = make([]complex128, n)
			p.transformScratch(scr, append([]complex128(nil), x...), true)
			packed = make([]complex128, n)
			p.transformRealPacked(packed, r)
			return
		}
		f0, i0, s0, p0 := run(n) // unblocked
		for _, block := range []int{4, 64, 1024, 4096} {
			f, i, s, pk := run(block)
			for k := range f0 {
				if f[k] != f0[k] || i[k] != i0[k] || s[k] != s0[k] || pk[k] != p0[k] {
					t.Fatalf("n=%d block=%d index %d differs from the unblocked run", n, block, k)
				}
			}
		}
	}
}
