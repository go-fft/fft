package fft

import (
	"math"
	"testing"
)

// armrealLines is the reference for the rotating passes: the 1-D plan of each
// axis run on every line of that axis, gathered and scattered, last axis
// first, unnormalized.
func armrealLines(p *PlanN, src []complex128, inverse bool) []complex128 {
	out := append([]complex128(nil), src...)
	for ax := len(p.shape) - 1; ax >= 0; ax-- {
		n, st, pl := p.shape[ax], p.stride[ax], p.axes[ax]
		if pl == nil {
			continue
		}
		line, res := make([]complex128, n), make([]complex128, n)
		for c := 0; c < p.size/n; c++ {
			base := lineBase(c, p.shape, p.stride, ax)
			for i := range line {
				line[i] = out[base+i*st]
			}
			pl.execute(res, line, inverse)
			for i, v := range res {
				out[base+i*st] = v
			}
		}
	}
	return out
}

// withRotate builds a PlanN with the rotating passes on or off.
func withRotate(on bool, shape ...int) *PlanN {
	defer func(v bool) { armrealRotateND = v }(armrealRotateND)
	armrealRotateND = on
	return NewPlanN(shape...)
}

// TestRotateMatchesLines holds the rotating passes to the lines, bit for bit
// (two NaNs equal), on every architecture: forward and inverse, out of place
// and in place, with even and odd pass counts, length-1 axes, radix 7 and
// three and four dimensions, on neonSignals (generic, signed zeros, ±0/−1, an
// infinity, a constant, ±0 and ±1 draws, subnormals).
func TestRotateMatchesLines(t *testing.T) {
	for _, shape := range [][]int{
		{2, 2}, {4, 8}, {8, 4}, {16, 16}, {64, 64}, {5, 7}, {7, 9}, {49, 6},
		{12, 10}, {11, 13}, {100, 3}, {3, 100}, {1, 64, 1, 32}, {3, 4, 5}, {8, 1, 8}, {16, 12, 10}, {2, 3, 4, 5},
	} {
		p := withRotate(true, shape...)
		if p.rot == nil {
			t.Fatalf("shape %v: no rotating passes", shape)
		}
		for s, x := range neonSignals(p.Len()) {
			for _, inverse := range []bool{false, true} {
				want := armrealLines(p, x, inverse)
				got := make([]complex128, p.Len())
				p.transform(got, x, inverse)
				in := append([]complex128(nil), x...)
				p.transform(in, in, inverse)
				for i := range want {
					if !neonSameBits(got[i], want[i]) || !neonSameBits(in[i], want[i]) {
						t.Fatalf("shape %v signal %d inverse=%v index %d: rotating %v, in place %v, lines %v", shape, s, inverse, i, got[i], in[i], want[i])
					}
				}
			}
		}
	}
}

// TestRotateAgainstOldPath compares the rotating passes with the rows and
// column strips they replace, to rounding (the axes run in the other order),
// through the public FFT and IFFT.
func TestRotateAgainstOldPath(t *testing.T) {
	for _, shape := range [][]int{{64, 64}, {128, 64}, {10, 24, 6}} {
		a, b := withRotate(true, shape...), withRotate(false, shape...)
		x := cmplxSignal(a.Len())
		for _, inverse := range []bool{false, true} {
			fa, fb := make([]complex128, a.Len()), make([]complex128, a.Len())
			if inverse {
				a.IFFT(fa, x)
				b.IFFT(fb, x)
			} else {
				a.FFT(fa, x)
				b.FFT(fb, x)
			}
			scale := maxPart(fb)
			for i := range fa {
				if d := math.Max(math.Abs(real(fa[i]-fb[i])), math.Abs(imag(fa[i]-fb[i]))); d > 1e-13*scale {
					t.Fatalf("shape %v inverse=%v index %d: %v vs %v", shape, inverse, i, fa[i], fb[i])
				}
			}
		}
	}
}

// TestRotateWhen: the rotating passes need every axis longer than 1
// transformed by a Stockham plan without the blocked schedule, at least two
// such axes, and a plan run on one goroutine; with the switch off no plan
// takes them.
func TestRotateWhen(t *testing.T) {
	for _, c := range []struct {
		shape []int
		want  bool
	}{
		{[]int{64, 64}, true},
		{[]int{1, 64, 1, 32}, true},
		{[]int{64}, false},       // one axis
		{[]int{1, 64}, false},    // one axis longer than 1
		{[]int{1, 1}, false},     // none
		{[]int{17, 64}, false},   // a prime axis: no Stockham plan
		{[]int{64, 1031}, false}, // the same, last
	} {
		if got := withRotate(true, c.shape...).rot != nil; got != c.want {
			t.Errorf("shape %v: rotating %v, want %v", c.shape, got, c.want)
		}
		if withRotate(false, c.shape...).rot != nil {
			t.Errorf("shape %v: rotating with the switch off", c.shape)
		}
	}
	// Fanned out across goroutines: not rotated.
	withWorkers(4, func() {
		if withRotate(true, 256, 256).rot != nil {
			t.Error("256×256 on four workers: rotating")
		}
	})
	// An axis left out of the transform keeps the rows-and-columns path.
	defer func(v bool) { armrealRotateND = v }(armrealRotateND)
	armrealRotateND = true
	if p := newPlanNAxes([]int{8, 16}, []int{0}); p.rot != nil {
		t.Error("an axis left out: rotating")
	}
}
