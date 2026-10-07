package kernels

import (
	"strings"
	"testing"
)

// TestUntangleBoundsNEON: the float64 untangle kernels trust their lengths,
// so the wrappers bound-check the last element each reads or writes before
// the kernel runs, and do nothing below one pair of bins (or for a negative
// m).
func TestUntangleBoundsNEON(t *testing.T) {
	const m = 64 // pairs = 15, the kernels touch tw[1..30]
	ok := func() (dst, z, tw []complex128) {
		return make([]complex128, m+1), make([]complex128, m), make([]complex128, 31)
	}
	for _, c := range []struct {
		name string
		cut  func(dst, z, tw []complex128) ([]complex128, []complex128, []complex128)
	}{
		{"short dst", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d[:m], z, w }},
		{"short z", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d, z[:m-1], w }},
		{"short tw", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d, z, w[:30] }},
	} {
		for _, inverse := range []bool{false, true} {
			func() {
				defer func() {
					err, _ := recover().(error)
					if err == nil || !strings.Contains(err.Error(), "index out of range") {
						t.Errorf("%s inverse=%v: recovered %v, want a bounds panic", c.name, inverse, err)
					}
				}()
				d, z, w := c.cut(ok())
				if inverse {
					// Retangle(z, x, tw): x plays dst's part (m+1), z z's.
					Retangle(z, d, w, m, 0.5)
				} else {
					Untangle(d, z, w, m)
				}
			}()
		}
	}
	d, z, w := ok()
	if got := Untangle(d, z, w, m); got != 30 {
		t.Errorf("Untangle(m=64) = %d, want 30", got)
	}
	if got := Retangle(z, d, w, m, 0.5); got != 30 {
		t.Errorf("Retangle(m=64) = %d, want 30", got)
	}
	for _, small := range []int{-9, -1, 0, 1, 4} {
		if Untangle(d, z, w, small) != 0 || Retangle(z, d, w, small, 0.5) != 0 {
			t.Errorf("m=%d: a kernel ran below one pair of bins", small)
		}
	}
	defer func(v bool) { UseUntangleNEON = v }(UseUntangleNEON)
	UseUntangleNEON = false
	if Untangle(d, z, w, m) != 0 || Retangle(z, d, w, m, 0.5) != 0 {
		t.Error("a kernel ran with UseUntangleNEON off")
	}
}
