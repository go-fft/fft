//go:build amd64 || arm64

package kernels

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestStockhamPass32RejectsInconsistentArguments: the float32 kernels trust
// ido, l1 and the slice lengths, so the entry point panics on any of them
// rather than letting the assembly overrun.
func TestStockhamPass32RejectsInconsistentArguments(t *testing.T) {
	defer func(v bool) { UseStockham32 = v }(UseStockham32)
	UseStockham32 = true
	a := make([]complex64, 64)
	tw := make([]complex64, 64)
	for _, c := range []struct {
		name string
		f    func()
	}{
		{"ido 0", func() { StockhamPass32(4, 0, 4, a, a, tw, false) }},
		{"l1 0", func() { StockhamPass32(4, 4, 0, a, a, tw, false) }},
		{"negative pair", func() { StockhamPass32(4, -2, -2, a, a, tw, false) }},
		{"cc short", func() { StockhamPass32(4, 4, 4, a[:63], a, tw, false) }},
		{"ch short", func() { StockhamPass32(4, 4, 4, a, a[:63], tw, false) }},
		{"tw short", func() { StockhamPass32(4, 4, 4, a, a, tw[:11], false) }},
		{"last cc short", func() { StockhamPass32(4, 1, 16, a[:63], a, nil, true) }},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", c.name)
				}
			}()
			c.f()
		}()
	}
}

// TestStockhamPass32Declines: no kernel for radix 7, and none when the
// kernels are switched off; the pass is left to Go and nothing is written.
func TestStockhamPass32Declines(t *testing.T) {
	defer func(v bool) { UseStockham32 = v }(UseStockham32)
	a := make([]complex64, 56)
	b := make([]complex64, 56)
	a[0] = 1
	UseStockham32 = true
	if StockhamPass32(7, 2, 4, a, b, make([]complex64, 12), false) || b[0] != 0 {
		t.Error("radix 7 ran on a kernel")
	}
	UseStockham32 = false
	if StockhamPass32(4, 2, 7, a, b, make([]complex64, 6), false) || b[0] != 0 {
		t.Error("a kernel ran while switched off")
	}
	if fwd, conj := StockhamTwiddles32(7, 8, 1, make([]complex128, 56)); fwd != nil || conj != nil {
		t.Error("radix 7 got a kernel twiddle table")
	}
	if fwd, conj := StockhamTwiddles32(4, 1, 4, make([]complex128, 16)); fwd != nil || conj != nil {
		t.Error("a final pass got a twiddle table")
	}
}

// TestStockhamTwiddles32Layout reads every entry back through the layout the
// kernels assume and compares it with the root it must hold, rounded once.
func TestStockhamTwiddles32Layout(t *testing.T) {
	for _, c := range []struct{ r, ido, l1 int }{{4, 2, 1}, {4, 3, 2}, {8, 13, 3}, {5, 7, 1}, {3, 16, 5}, {2, 6, 4}} {
		n := c.r * c.ido * c.l1
		root := make([]complex128, n)
		for k := range root {
			root[k] = cmplx.Exp(complex(0, -2*math.Pi*float64(k)/float64(n)))
		}
		fwd, conj := StockhamTwiddles32(c.r, c.ido, c.l1, root)
		if len(fwd) != (c.r-1)*c.ido || len(conj) != len(fwd) {
			t.Fatalf("%v: lengths %d, %d", c, len(fwd), len(conj))
		}
		for i := 0; i < c.ido; i++ {
			for j := 1; j < c.r; j++ {
				w := complex64(root[(j*c.l1*i)%n])
				f, g := twiddle32At(fwd, c.r, c.ido, i, j), twiddle32At(conj, c.r, c.ido, i, j)
				if f != w || g != complex(real(w), -imag(w)) {
					t.Fatalf("%v: i=%d j=%d: %v, %v; want %v and its conjugate", c, i, j, f, g, w)
				}
			}
		}
	}
}
