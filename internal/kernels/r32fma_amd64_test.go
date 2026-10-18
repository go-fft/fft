package kernels

import (
	"math"
	"testing"
)

// r32Data returns n deterministic complex values with signed zeros among
// them.
func r32Data(n int, seed float64) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		re, im := math.Sin(seed*float64(i+1)), math.Cos(seed*float64(2*i+3))
		if i%7 == 3 {
			re = math.Copysign(0, -1)
		}
		x[i] = complex(re, im)
	}
	return x
}

func r32SameBits(a, b []complex128) bool {
	for i := range a {
		if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) ||
			math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
			return false
		}
	}
	return true
}

// TestR32LastWidths: the final radix-32 kernel at 512 bits gives the 256-bit
// kernel's bits for every l1 from 1 to 13 (the four-block steps and both
// tails), both directions; the fft package holds the 256-bit kernel to its
// Go reference. And the radix-32 final pass runs a kernel (StockhamPassFMA
// reports true) except in place, where it hands the pass to Go.
func TestR32LastWidths(t *testing.T) {
	if !UseStockhamAVX2 || !UseFMA {
		t.Skip("no AVX2 and FMA3")
	}
	for l1 := 1; l1 <= 13; l1++ {
		x := r32Data(32*l1, 0.37)
		for _, inverse := range []bool{false, true} {
			narrow := make([]complex128, len(x))
			if !StockhamPassFMA(32, 1, l1, x, narrow, nil, inverse, false) {
				t.Fatalf("l1=%d: the final radix-32 pass ran no kernel", l1)
			}
			if !UseStockhamAVX512 {
				continue
			}
			wide := make([]complex128, len(x))
			if !StockhamPassFMA(32, 1, l1, x, wide, nil, inverse, true) {
				t.Fatalf("l1=%d: the final radix-32 pass ran no 512-bit kernel", l1)
			}
			if !r32SameBits(wide, narrow) {
				t.Fatalf("l1=%d inverse=%v: the 512-bit final radix-32 kernel differs from the 256-bit one", l1, inverse)
			}
		}
	}
	x := r32Data(32, 0.5)
	if StockhamPassFMA(32, 1, 1, x, x, nil, false, false) {
		t.Error("an in-place final radix-32 pass ran the kernel")
	}
	if StockhamPassFMA(32, 2, 1, r32Data(64, 0.5), make([]complex128, 64), nil, false, false) {
		t.Error("an interleaved radix-32 pass that is not the last ran a kernel")
	}
	if StockhamPassFMA(32, 1, 0, x, make([]complex128, 32), nil, false, false) {
		t.Error("a pass of no blocks ran a kernel")
	}
}

// TestR32SplitFrames: the in-frame split radix-32 kernels give the bits of
// the ones that keep their values in the output, for every mode, both
// directions and both widths, with several groups and blocks.
func TestR32SplitFrames(t *testing.T) {
	if !UseStockhamAVX2 || !UseFMA {
		t.Skip("no AVX2 and FMA3")
	}
	for _, wide := range []bool{false, true} {
		if wide && !UseStockhamAVX512 {
			continue
		}
		base, ido := uint8(0), 8
		if wide {
			base, ido = intelSplit512Base, 16
		}
		for _, l1 := range []int{1, 3} {
			n := 32 * ido * l1
			x := r32Data(n, 0.29)
			tw := r32Data(31*ido, 0.11)
			for m := uint8(splitOut); m <= splitNone; m++ {
				for d, inverse := range []bool{false, true} {
					a, b := make([]complex128, n), make([]complex128, n)
					var plain, frame skPassFn
					if wide {
						plain, frame = intelSplit512R32FMA[d][m], intelSplit512R32FMAF[d][m]
					} else {
						plain, frame = skSplit32FMA[d][m], skSplit32FMAF[d][m]
					}
					plain(&x[0], &a[0], &tw[0], &splitK[0][0], ido, l1)
					frame(&x[0], &b[0], &tw[0], &splitK[0][0], ido, l1)
					if !r32SameBits(a, b) {
						t.Fatalf("wide=%v mode %d inverse=%v l1=%d: the in-frame kernel differs", wide, base+m, inverse, l1)
					}
				}
			}
		}
	}
}
