//go:build amd64 && !amd64.v3

package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// intelSplitToBlocks8 returns x in the 512-bit split layout: eight points
// 8q .. 8q+7 as their real parts in the order 0, 4, 1, 5, 2, 6, 3, 7, then
// their imaginary parts in the same order (len(x) a multiple of 8).
func intelSplitToBlocks8(x []complex128) []complex128 {
	order := [8]int{0, 4, 1, 5, 2, 6, 3, 7}
	y := make([]complex128, len(x))
	for q := 0; q < len(x); q += 8 {
		for k := 0; k < 4; k++ {
			a, b := x[q+order[2*k]], x[q+order[2*k+1]]
			y[q+k] = complex(real(a), real(b))
			y[q+4+k] = complex(imag(a), imag(b))
		}
	}
	return y
}

// intelSplitFromBlocks8 is intelSplitToBlocks8's inverse.
func intelSplitFromBlocks8(y []complex128) []complex128 {
	order := [8]int{0, 4, 1, 5, 2, 6, 3, 7}
	x := make([]complex128, len(y))
	for q := 0; q < len(y); q += 8 {
		for k := 0; k < 4; k++ {
			x[q+order[2*k]] = complex(real(y[q+k]), real(y[q+4+k]))
			x[q+order[2*k+1]] = complex(imag(y[q+k]), imag(y[q+4+k]))
		}
	}
	return x
}

func TestIntelSplitBlocks8RoundTrip(t *testing.T) {
	x := make([]complex128, 16)
	for i := range x {
		x[i] = complex(float64(i), -float64(i)-0.5)
	}
	y := intelSplitToBlocks8(x)
	// Block 0: re(0) re(4) re(1) re(5) re(2) re(6) re(3) re(7), then the im.
	if y[0] != complex(0, 4) || y[1] != complex(1, 5) || y[3] != complex(3, 7) || y[4] != complex(-0.5, -4.5) {
		t.Fatalf("layout %v", y[:8])
	}
	z := intelSplitFromBlocks8(y)
	for i := range x {
		if z[i] != x[i] {
			t.Fatalf("round trip %d: %v vs %v", i, z[i], x[i])
		}
	}
}

// intelSplit512On builds plans with the 512-bit split layout on (or off)
// until the returned function restores the setting. It skips the test where
// the CPU has no AVX-512 (the layout then never runs).
func intelSplit512Plans(t *testing.T, on bool) func() {
	t.Helper()
	if !kernels.UseStockhamAVX512 || !kernels.UseStockhamAVX2 {
		t.Skip("no AVX-512 on this CPU: the 512-bit split kernels never run")
	}
	old := kernels.UseStockhamSplit512
	kernels.UseStockhamSplit512 = on
	return func() { kernels.UseStockhamSplit512 = old }
}

// TestIntelSplit512EachPassMatchesScalar runs every 512-bit split kernel
// alone against the Go pass, output for output, bit for bit, in all four
// layouts (the input converted to the split layout where the kernel reads it
// so, the output converted back where it writes it so): radix 4 and 8, l1
// from 1 to 9, ido from 8 (one group, the first only) to 72, forward and
// inverse, on radix16Signals' seven signals (generic, signed zeros, a ±0/−1
// mix, an infinity, random draws from {±0} and {±0, ±1}, odd multiples of the
// smallest subnormal).
func TestIntelSplit512EachPassMatchesScalar(t *testing.T) {
	defer intelSplit512Plans(t, true)()
	for _, r := range []int{4, 8} {
		for _, l1 := range []int{1, 2, 3, 4, 5, 9} {
			for _, ido := range []int{8, 16, 24, 40, 72} {
				var f []int
				stage := 0
				if l1 > 1 {
					f, stage = append(f, l1), 1
				}
				f = append(f, r, ido)
				n := r * ido * l1
				p := newSKPlanFactors(n, f)
				st := p.stages[stage]
				if st.r != r || st.l1 != l1 || st.ido != ido {
					t.Fatalf("stage %+v, want r=%d l1=%d ido=%d", st, r, l1, ido)
				}
				st.twX, st.twXc = kernels.StockhamSplitTwiddles(r, ido, l1, twiddleTable(n))
				for mode := uint8(1); mode <= 4; mode++ {
					st.split = 4 + mode
					for s, x := range radix16Signals(n) {
						for _, inverse := range []bool{false, true} {
							scalar := make([]complex128, n)
							st.passScalar(scalar, x, inverse)
							in := x
							if mode == 2 || mode == 3 {
								in = intelSplitToBlocks8(x)
							}
							simd := make([]complex128, n)
							st.pass(simd, in, inverse)
							if mode == 1 || mode == 2 {
								simd = intelSplitFromBlocks8(simd)
							}
							for i := range scalar {
								if !sameBits(simd[i], scalar[i]) {
									t.Fatalf("r=%d l1=%d ido=%d mode %d signal %d inverse=%v index %d: %v vs Go %v",
										r, l1, ido, 4+mode, s, inverse, i, simd[i], scalar[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// intelSplit512FactorLists are powers of two whose passes take the 512-bit
// split layout in runs of one, two and more, next to radix-2 and radix-16
// passes and passes whose ido (4) leaves them interleaved, with the run first,
// in the middle and before a final pass of every radix.
var intelSplit512FactorLists = [][]int{
	{4, 4, 16}, {8, 4, 8}, {4, 8, 8}, {8, 8, 4}, {4, 4, 4, 4}, {4, 8, 4, 8},
	{8, 8, 8}, {16, 4, 4, 4}, {2, 4, 4, 8}, {4, 4, 8, 2}, {4, 4, 4, 4, 4},
	{4, 8, 4, 8, 4}, {8, 8, 8, 8}, {8, 8, 16}, {4, 16, 16}, {4, 4, 4, 4, 4, 4, 4},
	{8, 8, 8, 4, 4, 4}, {16, 16}, {8, 8, 8, 8, 4},
}

// TestIntelSplit512TransformMatchesScalar runs whole transforms with
// 512-bit split runs, the kernels on and off, in place and not, bit for bit,
// forward and inverse, through plans built by NewPlan's path too.
func TestIntelSplit512TransformMatchesScalar(t *testing.T) {
	defer func(a, b bool) { kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = a, b }(kernels.UseStockhamAVX2, kernels.UseStockhamAVX512)
	defer intelSplit512Plans(t, true)()
	for _, f := range intelSplit512FactorLists {
		n := product(f)
		p := newSKPlanFactors(n, f)
		split := false
		for _, st := range p.stages {
			split = split || st.split > 4
			if st.split != 0 && st.split <= 4 {
				t.Fatalf("factors %v: a 256-bit split pass %+v with the 512-bit layout on", f, st)
			}
		}
		if !split && f[0] != 16 && f[0] != 2 {
			t.Fatalf("factors %v: no 512-bit split pass", f)
		}
		for s, x := range radix16Signals(n) {
			for _, inverse := range []bool{false, true} {
				kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = false, false
				scalar := make([]complex128, n)
				p.transform(scalar, x, inverse)
				for _, wide := range []bool{false, true} {
					kernels.UseStockhamAVX2, kernels.UseStockhamAVX512 = true, wide
					simd := make([]complex128, n)
					p.transform(simd, x, inverse)
					alias := append([]complex128(nil), x...)
					p.transform(alias, alias, inverse)
					for i := range scalar {
						if !sameBits(simd[i], scalar[i]) || !sameBits(alias[i], scalar[i]) {
							t.Fatalf("avx512=%v factors %v signal %d inverse=%v index %d: %v (in place %v) vs Go %v",
								wide, f, s, inverse, i, simd[i], alias[i], scalar[i])
						}
					}
				}
			}
		}
	}
}
