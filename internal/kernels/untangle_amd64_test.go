package kernels

import (
	"strings"
	"testing"
)

// TestUntangleBoundsAVX2: the float64 untangle kernels trust their lengths,
// so the wrappers bound-check the last element each reads or writes before
// the kernel runs (the twiddles up to tw[2·pairs+1], whose real part the last
// step's VMOVDDUP at tw+8 reads), and do nothing below one pair of bins (or
// for a negative m).
func TestUntangleBoundsAVX2(t *testing.T) {
	if !UseUntangleAVX2 {
		t.Skip("no AVX2 on this CPU: the wrappers return before any bound check")
	}
	const m = 64 // pairs = 15, the kernels touch tw[1..31]
	ok := func() (dst, z, tw []complex128) {
		return make([]complex128, m+1), make([]complex128, m), make([]complex128, 32)
	}
	for _, c := range []struct {
		name string
		cut  func(dst, z, tw []complex128) ([]complex128, []complex128, []complex128)
	}{
		{"short dst", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d[:m], z, w }},
		{"short z", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d, z[:m-1], w }},
		{"short tw", func(d, z, w []complex128) ([]complex128, []complex128, []complex128) { return d, z, w[:31] }},
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
	defer func(v bool) { UseUntangleAVX2 = v }(UseUntangleAVX2)
	UseUntangleAVX2 = false
	if Untangle(d, z, w, m) != 0 || Retangle(z, d, w, m, 0.5) != 0 {
		t.Error("a kernel ran with UseUntangleAVX2 off")
	}
}

// TestUntangle512Default: the 512-bit untangle runs with AVX2 and AVX-512 on
// an Intel CPU only.
func TestUntangle512Default(t *testing.T) {
	for _, c := range []struct{ avx2, avx512, intel, want bool }{
		{true, true, true, true},
		{true, true, false, false},
		{true, false, true, false},
		{false, true, true, false},
	} {
		if got := untangle512Default(c.avx2, c.avx512, c.intel); got != c.want {
			t.Errorf("untangle512Default(%v, %v, %v) = %v", c.avx2, c.avx512, c.intel, got)
		}
	}
}

// TestClUntangleWide: the 512-bit untangle takes the powers of two from 256.
func TestClUntangleWide(t *testing.T) {
	for m, want := range map[int]bool{128: false, 255: false, 256: true, 500: false, 1024: true, 1 << 14: true, 1 << 15: false, 3 << 8: false} {
		if got := clUntangleWide(m); got != want {
			t.Errorf("clUntangleWide(%d) = %v, want %v", m, got, want)
		}
	}
}

// TestClQuads: half the pairs, rounded down, go to the 512-bit kernel when it
// is on and the length takes it; none otherwise.
func TestClQuads(t *testing.T) {
	for _, c := range []struct {
		on          bool
		pairs, m, q int
	}{
		{true, 63, 256, 31}, {true, 64, 512, 32}, {false, 63, 256, 0}, {true, 31, 128, 0}, {true, 249, 1000, 0},
	} {
		if got := clQuads(c.on, c.pairs, c.m); got != c.q {
			t.Errorf("clQuads(%v, %d, %d) = %d, want %d", c.on, c.pairs, c.m, got, c.q)
		}
	}
}
