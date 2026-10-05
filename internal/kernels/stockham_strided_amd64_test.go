package kernels

import (
	"math"
	"math/cmplx"
	"reflect"
	"testing"
)

// stridedSignal is a generic signal with signed zeros, infinities and
// subnormals mixed in: an infinity times the i = 0 twiddle (1, 0) gives a NaN,
// so a missing blend shows.
func stridedSignal(n int) []complex128 {
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(math.Sin(float64(i)*0.7), math.Cos(float64(i)*1.3))
	}
	x[0] = complex(math.Inf(1), math.Copysign(0, -1))
	x[5] = complex(5e-324, math.Inf(-1))
	x[n/2] = complex(math.Copysign(0, -1), 0)
	return x
}

func sameC(a, b complex128) bool {
	eq := func(x, y float64) bool {
		return math.Float64bits(x) == math.Float64bits(y) || (math.IsNaN(x) && math.IsNaN(y))
	}
	return eq(real(a), real(b)) && eq(imag(a), imag(b))
}

// TestStockhamStridedSplitsThePass: a pass cut into strided pieces, chunk by
// chunk and block by block or all blocks at once, gives the pass's bits.
func TestStockhamStridedSplitsThePass(t *testing.T) {
	a2, a512 := UseStockhamAVX2, UseStockhamAVX512
	defer func() { UseStockhamAVX2, UseStockhamAVX512 = a2, a512 }()
	if !a2 {
		t.Skip("no AVX2")
	}
	const ido, l1, cnt = 64, 4, 16
	for _, wide := range []bool{false, true} {
		for _, r := range []int{4, 8} {
			n := r * ido * l1
			root := make([]complex128, n)
			for k := range root {
				root[k] = cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n)))
			}
			fwd, conj := StockhamTwiddles(r, ido, l1, root)
			for _, inverse := range []bool{false, true} {
				tw := fwd
				if inverse {
					tw = conj
				}
				cc := stridedSignal(n)
				want := make([]complex128, n)
				if !StockhamPass(r, ido, l1, cc, want, tw, inverse, wide) {
					t.Fatal("no pass kernel")
				}
				one, all := make([]complex128, n), make([]complex128, n)
				for i0 := 0; i0 < ido; i0 += cnt {
					for k := 0; k < l1; k++ {
						if !StockhamStrided(r, cc[r*ido*k+i0:], one[ido*k+i0:], tw[i0*(r-1):], cnt, 1, ido, r*ido, l1*ido, ido, 0, i0 == 0, false, inverse, wide) {
							t.Fatal("StockhamStrided reported no kernel")
						}
					}
					StockhamStrided(r, cc[i0:], all[i0:], tw[i0*(r-1):], cnt, l1, ido, r*ido, l1*ido, ido, 0, i0 == 0, i0 == 0, inverse, wide)
				}
				for i := range want {
					if !sameC(one[i], want[i]) || !sameC(all[i], want[i]) {
						t.Fatalf("wide=%v r=%d inverse=%v: [%d] = %v (one block) %v (all), pass %v", wide, r, inverse, i, one[i], all[i], want[i])
					}
				}
			}
		}
	}
}

// TestStockhamStridedDeclines: the cases with no kernel report false.
func TestStockhamStridedDeclines(t *testing.T) {
	a2 := UseStockhamAVX2
	defer func() { UseStockhamAVX2 = a2 }()
	cc, ch, tw := make([]complex128, 512), make([]complex128, 512), make([]complex128, 512)
	for _, c := range []struct{ r, cnt, nb int }{
		{2, 4, 1}, {5, 4, 1}, {4, 2, 1}, {4, 6, 1}, {4, 0, 1}, {4, 4, 0},
	} {
		if StockhamStrided(c.r, cc, ch, tw, c.cnt, c.nb, 8, 32, 8, 8, 0, true, false, false, true) {
			t.Errorf("r=%d cnt=%d nb=%d: reported a kernel", c.r, c.cnt, c.nb)
		}
	}
	UseStockhamAVX2 = false
	if StockhamStrided(4, cc, ch, tw, 4, 1, 8, 32, 8, 8, 0, true, false, false, true) {
		t.Error("without AVX2: reported a kernel")
	}
}

// TestStockhamStridedKernel checks the selection for both widths, and flag.
func TestStockhamStridedKernel(t *testing.T) {
	same := func(a, b any) bool { return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer() }
	for _, r := range []int{4, 8} {
		if !same(stockhamStridedKernel(r, 512), skStrided512[r]) || !same(stockhamStridedKernel(r, 256), skStrided256[r]) {
			t.Errorf("r=%d: wrong kernel", r)
		}
	}
	if flag(true) != 1 || flag(false) != 0 {
		t.Error("flag")
	}
}
