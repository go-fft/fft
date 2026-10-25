//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The FMA kernels against their Go reference (Round 41). Restricted, like
// TestStockhamPassMatchesScalar, to GOAMD64 < v3: the Go passes an FMA plan
// keeps for its untwiddled final passes are the default ones, which the
// compiler fuses at v3.

// fmaKernelsHere skips a test that runs the FMA kernels on a CPU without
// FMA3, where they would fault.
func fmaKernelsHere(t *testing.T) {
	t.Helper()
	if !kernels.UseFMA {
		t.Skip("no AVX2 and FMA3 on this CPU: the FMA kernels cannot run")
	}
}

// r41Layouts are the kernel configurations an FMA plan is built under: the
// machine's own, and every narrower one it can run (the 256-bit split
// layout, interleaved passes only), so that each kernel family runs on a
// machine that has it.
func r41Layouts() []struct {
	name                 string
	avx512, split, sp512 bool
} {
	all := []struct {
		name                 string
		avx512, split, sp512 bool
	}{
		{"AVX2 interleaved", false, false, false},
		{"AVX2 split", false, true, false},
	}
	if kernels.UseStockhamAVX512 {
		all = append(all,
			struct {
				name                 string
				avx512, split, sp512 bool
			}{"AVX-512 interleaved", true, false, false},
			struct {
				name                 string
				avx512, split, sp512 bool
			}{"AVX-512 split", true, false, true})
	}
	return all
}

// withLayout builds under one configuration and restores the machine's.
func withLayout(avx512, split, sp512 bool, f func()) {
	defer func(a, b, c bool) {
		kernels.UseStockhamAVX512, kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = a, b, c
	}(kernels.UseStockhamAVX512, kernels.UseStockhamSplit, kernels.UseStockhamSplit512)
	kernels.UseStockhamAVX512, kernels.UseStockhamSplit, kernels.UseStockhamSplit512 = avx512, split, sp512
	f()
}

// TestFMAPassMatchesReference runs every smooth length up to 2100 and a few
// larger ones through an FMA plan twice, kernels on and off, under every
// layout the machine runs, and requires the same bits, forward and inverse,
// in place and not. Kernels off, an FMA stage runs passScalarFMA, so this
// holds every FMA kernel (interleaved at 256 and 512 bits, split at both
// widths, radix 9, 10, 12, 15, 16 and 20, the frame plans) to cmulFMA.
func TestFMAPassMatchesReference(t *testing.T) {
	fmaKernelsHere(t)
	defer func(a bool) { kernels.UseStockhamAVX2 = a }(kernels.UseStockhamAVX2)
	sizes := []int{4096, 8192, 16384, 20160, 45000, 65536}
	for n := 2; n <= 2100; n++ {
		sizes = append(sizes, n)
	}
	for _, l := range r41Layouts() {
		withLayout(l.avx512, l.split, l.sp512, func() {
			fused := 0
			for _, n := range sizes {
				if !factorsAreSmall(n) {
					continue
				}
				p := newSKPlanOpt(n, true)
				for _, st := range p.stages {
					if st.fma && st.ido > 1 {
						fused++
					}
				}
				for s, x := range simdSignals(n) {
					for _, inverse := range []bool{false, true} {
						kernels.UseStockhamAVX2 = false
						ref := make([]complex128, n)
						p.transform(ref, x, inverse)
						kernels.UseStockhamAVX2 = true
						got := make([]complex128, n)
						p.transform(got, x, inverse)
						alias := append([]complex128(nil), x...)
						p.transform(alias, alias, inverse)
						for i := range ref {
							if !sameBits(got[i], ref[i]) || !sameBits(alias[i], ref[i]) {
								t.Fatalf("%s n=%d signal %d inverse=%v index %d: %v (in place %v), reference %v (factors %v)",
									l.name, n, s, inverse, i, got[i], alias[i], ref[i], skFactorize(n))
							}
						}
					}
				}
			}
			if fused == 0 {
				t.Fatalf("%s: no FMA stage was built", l.name)
			}
		})
	}
}

// TestFMABatchMatchesReference holds the FMA batched kernels (the strips of
// an N-D FMA plan) to passBatchScalarFMA, pass by pass: every batched
// radix, the final pass (radix 8 fuses its √2/2 products there too) and
// twiddled ones, batches of 1 to 9 lines, forward and inverse.
func TestFMABatchMatchesReference(t *testing.T) {
	fmaKernelsHere(t)
	defer func(b bool) { kernels.UseStockhamBatchAVX2 = b }(kernels.UseStockhamBatchAVX2)
	for _, f := range [][]int{{8, 8}, {2, 8, 4}, {3, 8}, {5, 8}, {4, 4, 8}, {8, 3, 5}} {
		n := factorProduct(f)
		p := newSKPlanFactorsOpt(n, f, true)
		for k := range p.stages {
			st := &p.stages[k]
			tf, tc := st.batchTwiddles()
			for _, w := range []int{1, 2, 7, 9} {
				x := fmaSignal(n*w, uint64(31*n+w))
				for _, inverse := range []bool{false, true} {
					tw := tf
					if inverse {
						tw = tc
					}
					kernels.UseStockhamBatchAVX2 = true
					got := make([]complex128, n*w)
					st.passBatch(got, x, tw, w, w, w, inverse)
					kernels.UseStockhamBatchAVX2 = false
					want := make([]complex128, n*w)
					st.passBatch(want, x, tw, w, w, w, inverse)
					if !sameAll(got, want) {
						t.Fatalf("%v stage %d (radix %d, ido %d) w=%d inverse=%v: FMA batched kernel differs from its reference", f, k, st.r, st.ido, w, inverse)
					}
				}
			}
		}
	}
}

// TestFMARotatingPassFallback: arm64's rotating passes on an FMA stage fall
// back to its Go reference when the kernels are off, and agree with them.
func TestFMARotatingPassFallback(t *testing.T) {
	fmaKernelsHere(t)
	defer func(a bool) { kernels.UseStockhamAVX2 = a }(kernels.UseStockhamAVX2)
	st := &newSKPlanFactorsOpt(32, []int{4, 8}, true).stages[0]
	in := fmaSignal(32*6, 17)
	a, b := make([]complex128, len(in)), make([]complex128, len(in))
	st.armrealPassL(a, in, 6*st.l1, false)
	kernels.UseStockhamAVX2 = false
	st.armrealPassL(b, in, 6*st.l1, false)
	if !sameAll(a, b) {
		t.Fatal("armrealPassL: the FMA kernel and its Go reference differ")
	}
}

// TestFMACascadeGoPasses: an FMA plan on the blocked schedule with the
// kernels off (its Go passes over l1 blocks, strided pieces and final runs)
// is its kernels' bits.
func TestFMACascadeGoPasses(t *testing.T) {
	fmaKernelsHere(t)
	defer func(a int, b bool) { cascadeMin, kernels.UseStockhamAVX2 = a, b }(cascadeMin, kernels.UseStockhamAVX2)
	cascadeMin = 1
	n := 1 << 14
	p := newSKPlanFactorsOpt(n, skFactorize(n), true)
	x := fmaSignal(n, 3)
	a, b := make([]complex128, n), make([]complex128, n)
	p.transform(a, x, false)
	kernels.UseStockhamAVX2 = false
	p.transform(b, x, false)
	if p.cascT == 0 || !sameAll(a, b) {
		t.Fatal("an FMA blocked schedule's Go passes differ from its kernels")
	}
}
