package kernels

import (
	"math"
	"reflect"
	"testing"
)

// TestStockhamLastRunPlacesTheFinalPass: StockhamLastRun computes what the
// final-pass kernel of the same width computes, every output landing at
// q·gap + c + j·os, and writes nothing else.
func TestStockhamLastRunPlacesTheFinalPass(t *testing.T) {
	a2, a512 := UseStockhamAVX2, UseStockhamAVX512
	defer func() { UseStockhamAVX2, UseStockhamAVX512 = a2, a512 }()
	if !a2 {
		t.Skip("no AVX2")
	}
	for _, wide := range []bool{false, true} {
		for _, r := range []int{2, 4, 8} {
			for _, inverse := range []bool{false, true} {
				const runs, run, gap = 6, 8, 20
				l1 := runs * run
				os := runs*gap + 3
				cc := make([]complex128, r*l1)
				for i := range cc {
					cc[i] = complex(math.Sin(float64(i)*0.7), math.Cos(float64(i)*1.3))
				}
				want := make([]complex128, r*l1)
				if !StockhamPass(r, 1, l1, cc, want, nil, inverse, wide) {
					t.Fatal("no final-pass kernel")
				}
				sentinel := complex(-7, 7)
				ch := make([]complex128, (r-1)*os+(runs-1)*gap+run+5)
				for i := range ch {
					ch[i] = sentinel
				}
				if !StockhamLastRun(r, cc, ch, os, runs, run, gap, inverse, wide) {
					t.Fatal("StockhamLastRun reported no kernel")
				}
				seen := make([]bool, len(ch))
				for j := 0; j < r; j++ {
					for k := 0; k < l1; k++ {
						at := (k/run)*gap + k%run + j*os
						seen[at] = true
						if math.Float64bits(real(ch[at])) != math.Float64bits(real(want[j*l1+k])) ||
							math.Float64bits(imag(ch[at])) != math.Float64bits(imag(want[j*l1+k])) {
							t.Fatalf("wide=%v r=%d inverse=%v: output %d of block %d = %v, want %v", wide, r, inverse, j, k, ch[at], want[j*l1+k])
						}
					}
				}
				for i, v := range ch {
					if !seen[i] && v != sentinel {
						t.Fatalf("wide=%v r=%d: wrote [%d] = %v outside the runs", wide, r, i, v)
					}
				}
			}
		}
	}
}

// TestStockhamLastRunDeclines: the cases with no kernel report false.
func TestStockhamLastRunDeclines(t *testing.T) {
	a2 := UseStockhamAVX2
	defer func() { UseStockhamAVX2 = a2 }()
	cc, ch := make([]complex128, 64), make([]complex128, 256)
	for _, c := range []struct{ r, runs, run int }{
		{3, 1, 4}, {5, 1, 4}, {4, 0, 4}, {4, 1, 2}, {4, 1, 6}, {4, 1, 0},
	} {
		if StockhamLastRun(c.r, cc, ch, 16, c.runs, c.run, 16, false, true) {
			t.Errorf("r=%d runs=%d run=%d: reported a kernel", c.r, c.runs, c.run)
		}
	}
	UseStockhamAVX2 = false
	if StockhamLastRun(4, cc, ch, 16, 1, 4, 16, false, true) {
		t.Error("without AVX2: reported a kernel")
	}
}

// TestStockhamLastRunKernel checks the selection for both widths.
func TestStockhamLastRunKernel(t *testing.T) {
	same := func(a, b any) bool { return reflect.ValueOf(a).Pointer() == reflect.ValueOf(b).Pointer() }
	for _, r := range []int{2, 4, 8} {
		if !same(stockhamLastRunKernel(r, 512), skLastRun512[r]) || !same(stockhamLastRunKernel(r, 256), skLastRun256[r]) {
			t.Errorf("r=%d: wrong kernel", r)
		}
	}
}
