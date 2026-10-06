//go:build amd64 || arm64

package kernels

import (
	"strings"
	"testing"
)

// TestUntangle32Bounds: the float32 untangle kernels trust their lengths, so
// the wrappers bound-check the last element each reads or writes before the
// kernel runs, and do nothing below one group of four bins (or for a
// negative m).
func TestUntangle32Bounds(t *testing.T) {
	if !UseUntangle32 {
		t.Skip("no float32 untangle kernel on this CPU")
	}
	const m = 64 // quads = 7, the kernels touch tw[1..28]
	ok := func() (dst, z, tw []complex64) {
		return make([]complex64, m+1), make([]complex64, m), make([]complex64, 29)
	}
	for _, c := range []struct {
		name string
		cut  func(dst, z, tw []complex64) ([]complex64, []complex64, []complex64)
	}{
		{"short dst", func(d, z, w []complex64) ([]complex64, []complex64, []complex64) { return d[:m], z, w }},
		{"short z", func(d, z, w []complex64) ([]complex64, []complex64, []complex64) { return d, z[:m-1], w }},
		{"short tw", func(d, z, w []complex64) ([]complex64, []complex64, []complex64) { return d, z, w[:28] }},
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
					// Retangle32(z, x, tw): x plays dst's part (m+1), z z's.
					Retangle32(z, d, w, m, 0.5)
				} else {
					Untangle32(d, z, w, m)
				}
			}()
		}
	}
	d, z, w := ok()
	if got := Untangle32(d, z, w, m); got != 28 {
		t.Errorf("Untangle32(m=64) = %d, want 28", got)
	}
	if got := Retangle32(z, d, w, m, 0.5); got != 28 {
		t.Errorf("Retangle32(m=64) = %d, want 28", got)
	}
	for _, small := range []int{-9, 0, 1, 8} {
		if Untangle32(d, z, w, small) != 0 || Retangle32(z, d, w, small, 0.5) != 0 {
			t.Errorf("m=%d: a kernel ran below one group of four bins", small)
		}
	}
	defer func(v bool) { UseUntangle32 = v }(UseUntangle32)
	UseUntangle32 = false
	if Untangle32(d, z, w, m) != 0 || Retangle32(z, d, w, m, 0.5) != 0 {
		t.Error("a kernel ran with UseUntangle32 off")
	}
}
