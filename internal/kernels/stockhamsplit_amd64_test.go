package kernels

import (
	"slices"
	"strings"
	"testing"
)

// TestSplitModes: every maximal run of passes a split kernel can run gets
// splitOut, splitBoth…, splitIn (splitNone alone); the others, and the final
// pass, stay interleaved; off, everything is interleaved.
func TestSplitModes(t *testing.T) {
	for _, c := range []struct {
		r, ido []int
		want   []uint8
	}{
		{[]int{4, 4}, []int{4, 1}, []uint8{splitNone, 0}},
		{[]int{4, 4, 4, 4}, []int{64, 16, 4, 1}, []uint8{splitOut, splitBoth, splitIn, 0}},
		{[]int{8, 8, 16}, []int{128, 16, 1}, []uint8{splitOut, splitIn, 0}},
		{[]int{2, 4, 4, 8}, []int{128, 32, 8, 1}, []uint8{0, splitOut, splitIn, 0}},
		{[]int{4, 3, 4, 8}, []int{96, 32, 8, 1}, []uint8{splitNone, 0, splitNone, 0}},
		{[]int{4, 3, 4, 4, 8}, []int{384, 128, 32, 8, 1}, []uint8{splitNone, 0, splitOut, splitIn, 0}},
		{[]int{4, 4, 2}, []int{8, 2, 1}, []uint8{splitNone, 0, 0}},
		{[]int{16, 4, 4}, []int{16, 4, 1}, []uint8{0, splitNone, 0}},
		{[]int{4, 4, 7}, []int{28, 7, 1}, []uint8{splitNone, 0, 0}},
		{[]int{5}, []int{1}, []uint8{0}},
		{nil, nil, []uint8{}},
	} {
		if got := splitModes(c.r, c.ido, true); !slices.Equal(got, c.want) {
			t.Errorf("splitModes(%v, %v) = %v, want %v", c.r, c.ido, got, c.want)
		}
		if got := splitModes(c.r, c.ido, false); slices.ContainsFunc(got, func(m uint8) bool { return m != 0 }) {
			t.Errorf("splitModes(%v, %v, off) = %v, want all zero", c.r, c.ido, got)
		}
	}
	old := UseStockhamSplit
	defer func() { UseStockhamSplit = old }()
	UseStockhamSplit = true
	if got := StockhamSplitModes([]int{4, 4}, []int{4, 1}, []int{1, 4}); got[0] != splitNone {
		t.Errorf("StockhamSplitModes on: %v", got)
	}
	UseStockhamSplit = false
	if got := StockhamSplitModes([]int{4, 4}, []int{4, 1}, []int{1, 4}); got[0] != 0 {
		t.Errorf("StockhamSplitModes off: %v", got)
	}
}

// TestSplitPassLayoutRefuses: a mode the pass cannot run, and slices shorter
// than the pass, are refused before any kernel runs, on every machine.
func TestSplitPassLayoutRefuses(t *testing.T) {
	buf := make([]complex128, 4096)
	expect := func(name, prefix string, f func()) {
		t.Helper()
		defer func() {
			msg := ""
			switch v := recover().(type) {
			case string:
				msg = v
			case error:
				msg = v.Error()
			}
			if !strings.Contains(msg, prefix) {
				t.Errorf("%s: recovered %q, want a %q panic", name, msg, prefix)
			}
		}()
		f()
	}
	const refuse = "kernels: StockhamPassLayout:"
	for _, c := range []struct {
		mode       uint8
		r, ido, l1 int
	}{
		{5, 4, 4, 1}, {splitBoth, 4, 6, 1}, {splitBoth, 4, 2, 1}, {splitBoth, 4, 0, 1},
		{splitBoth, 4, -4, 1}, {splitIn, 3, 4, 1}, {splitOut, 16, 4, 1}, {splitNone, 2, 4, 1},
		{splitOut, -1, 4, 1}, {splitOut, 9, 4, 1}, {splitBoth, 4, 4, 0}, {splitBoth, 8, 4, -1},
	} {
		expect("mode", refuse, func() {
			StockhamPassLayout(c.mode, c.r, c.ido, c.l1, buf, buf, buf, false, false)
		})
	}
	short := buf[:63]
	expect("cc", "index out of range", func() { StockhamPassLayout(splitBoth, 4, 4, 4, short, buf, buf, false, false) })
	expect("ch", "index out of range", func() { StockhamPassLayout(splitBoth, 4, 4, 4, buf, short, buf, false, false) })
	expect("tw", "index out of range", func() { StockhamPassLayout(splitBoth, 8, 8, 1, buf, buf, buf[:55], false, false) })
}

// TestSplitPassLayoutOff: without AVX2 a split mode reports false, so the Go
// pass runs (every pass of the transform does, interleaved), and mode 0 is
// StockhamPass.
func TestSplitPassLayoutOff(t *testing.T) {
	old := UseStockhamAVX2
	defer func() { UseStockhamAVX2 = old }()
	UseStockhamAVX2 = false
	buf := make([]complex128, 64)
	if StockhamPassLayout(splitBoth, 4, 4, 4, buf, buf, buf, false, false) {
		t.Error("split mode ran without AVX2")
	}
	if StockhamPassLayout(0, 4, 4, 4, buf, buf, buf, false, false) {
		t.Error("mode 0 ran without AVX2")
	}
}

// TestSplitTwiddles: the table holds, per four-point group and twiddle j, the
// real parts in the order 0, 2, 1, 3, then the imaginary parts, conjugated in
// the conjugate table.
func TestSplitTwiddles(t *testing.T) {
	const n = 128
	root := make([]complex128, n)
	for k := range root {
		root[k] = complex(float64(k), -float64(k)-0.5)
	}
	r, ido, l1 := 4, 8, 4
	fwd, conj := StockhamSplitTwiddles(r, ido, l1, root)
	if len(fwd) != (r-1)*ido || len(conj) != len(fwd) {
		t.Fatalf("lengths %d, %d", len(fwd), len(conj))
	}
	at := 0
	for i0 := 0; i0 < ido; i0 += 4 {
		for j := 1; j < r; j++ {
			w := func(d int) complex128 { return root[(j*l1*(i0+d))%n] }
			want := []complex128{
				complex(real(w(0)), real(w(2))), complex(real(w(1)), real(w(3))),
				complex(imag(w(0)), imag(w(2))), complex(imag(w(1)), imag(w(3))),
			}
			for q := range want {
				c := want[q]
				if q >= 2 {
					c = -c
				}
				if fwd[at+q] != want[q] || conj[at+q] != c {
					t.Fatalf("i0=%d j=%d q=%d: %v %v, want %v %v", i0, j, q, fwd[at+q], conj[at+q], want[q], c)
				}
			}
			at += 4
		}
	}
}
