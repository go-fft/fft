package fft

import "testing"

// withWideStrips builds a plan with Round 39's wide strips forced on or off.
func withWideStrips(on bool, shape ...int) *PlanN {
	defer func(v bool) { cl3WideStrips = v }(cl3WideStrips)
	cl3WideStrips = on
	return NewPlanN(shape...)
}

// TestCl3StripWidth pins cl3StripWidth: with cl3WideStrips, 128 lines up to
// n = 256 and 64 above on an axis spanning at least 2^18 elements whose
// lines are at least 256 elements apart, stripWidth(n) on any other; and
// stripWidth(n) everywhere without it.
func TestCl3StripWidth(t *testing.T) {
	defer func(v bool) { cl3WideStrips = v }(cl3WideStrips)
	for _, c := range []struct {
		n, st, wide int
	}{
		{512, 512, 64},
		{1024, 1024, 64},
		{257, 1021, 64},
		{256, 1024, 128},
		{64, 4096, 128},
		{1024, 256, 64},            // exactly 2^18 elements, 256 apart
		{1023, 256, 32},            // one line short of the span
		{1025, 255, 32},            // the span, lines 255 apart
		{512, 256, 16},             // 2^17
		{4096, 64, 32},             // lines 64 apart
		{128, 128, 16},             // small
		{2, 1 << 17, 128},          // two lines far apart
		{1 << 18, 1, 32},           // contiguous (never a strip axis)
		{512, 1 << 20, 64},         // large
		{100, (1 << 18) / 100, 16}, // 2^18 - 44 elements
	} {
		cl3WideStrips = false
		if got := cl3StripWidth(c.n, c.st); got != stripWidth(c.n) {
			t.Errorf("off: cl3StripWidth(%d, %d) = %d, want stripWidth %d", c.n, c.st, got, stripWidth(c.n))
		}
		cl3WideStrips = true
		if got := cl3StripWidth(c.n, c.st); got != c.wide {
			t.Errorf("on: cl3StripWidth(%d, %d) = %d, want %d", c.n, c.st, got, c.wide)
		}
	}
}

// TestCl3WideStripsPlans: a plan built with the wide strips gives each strip
// axis cl3StripWidth's width and its strip count, and the scratch it
// allocates holds two strips of it.
func TestCl3WideStripsPlans(t *testing.T) {
	if !stripAxes {
		t.Skip("no batched kernels here")
	}
	p := withWideStrips(true, 512, 512)
	if p.sw[0] != 64 || p.stripCount(0) != 8 {
		t.Errorf("512²: width %d, %d strips; want 64, 8", p.sw[0], p.stripCount(0))
	}
	if b := p.bufs.Get().(*[]complex128); len(*b) < 2*512*64 {
		t.Errorf("512²: scratch %d", len(*b))
	}
	q := withWideStrips(false, 512, 512)
	if q.sw[0] != stripWidth(512) || q.stripCount(0) != 512/stripWidth(512) {
		t.Errorf("512² narrow: width %d, %d strips", q.sw[0], q.stripCount(0))
	}
	r := withWideStrips(true, 64, 64, 64)
	if r.sw[0] != 128 || r.sw[1] != 16 || r.stripCount(0) != 32 || r.stripCount(1) != 64*4 {
		t.Errorf("64³: widths %v, strips %d %d", r.sw, r.stripCount(0), r.stripCount(1))
	}
}

// TestCl3WideStripsMatch holds plans with the wide strips to the same plans
// with stripWidth's, bit for bit: forward and inverse, out of place and in
// place. Each line of a strip gets its own 1-D transform's arithmetic
// whatever the strip's width. The shapes take both wide widths, a last
// strip narrower than the width (300×1000: 1000 = 15·64 + 40), composites,
// three axes, and the real 2-D transform's columns.
func TestCl3WideStripsMatch(t *testing.T) {
	if !stripAxes {
		t.Skip("no batched kernels here")
	}
	shapes := [][]int{{512, 512}, {300, 1000}, {64, 4096}, {64, 64, 64}, {2, 256, 1024}}
	if testing.Short() {
		shapes = shapes[:2]
	}
	for _, shape := range shapes {
		wide, narrow := withWideStrips(true, shape...), withWideStrips(false, shape...)
		differ := false
		for ax := range shape {
			differ = differ || wide.sw[ax] != narrow.sw[ax]
		}
		if !differ {
			t.Fatalf("shape %v: widths %v and %v", shape, wide.sw, narrow.sw)
		}
		n := shapeProduct(shape...)
		x := benchComplex(n)
		for _, inverse := range []bool{false, true} {
			for _, inPlace := range []bool{false, true} {
				run := func(p *PlanN) []complex128 {
					out := make([]complex128, n)
					src := x
					if inPlace {
						copy(out, x)
						src = out
					}
					if inverse {
						return p.IFFT(out, src)
					}
					return p.FFT(out, src)
				}
				got, want := run(wide), run(narrow)
				for i := range want {
					if !sameBits(got[i], want[i]) {
						t.Fatalf("shape %v inverse=%v in place=%v index %d: wide %v, narrow %v", shape, inverse, inPlace, i, got[i], want[i])
					}
				}
			}
		}
	}
	// RealPlan2's columns are a PlanN axis: 512×1024 real is 512×513
	// complex columns, lines 513 apart.
	defer func(v bool) { cl3WideStrips = v }(cl3WideStrips)
	x := make([]float64, 512*1024)
	for i := range x {
		x[i] = float64(i%17) - 8.5
	}
	cl3WideStrips = true
	a := NewRealPlan2(512, 1024)
	cl3WideStrips = false
	b := NewRealPlan2(512, 1024)
	if a.col.sw[0] != 64 || b.col.sw[0] != stripWidth(512) {
		t.Fatalf("real 512×1024 columns: widths %d, %d", a.col.sw[0], b.col.sw[0])
	}
	ya := a.RFFT(make([]complex128, 512*513), x)
	yb := b.RFFT(make([]complex128, 512*513), x)
	for i := range ya {
		if !sameBits(ya[i], yb[i]) {
			t.Fatalf("real 512×1024 index %d: wide %v, narrow %v", i, ya[i], yb[i])
		}
	}
}
