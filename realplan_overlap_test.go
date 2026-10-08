package fft

import (
	"math"
	"testing"
	"unsafe"
)

// TestRealPlanOverlappingSlices: dst and src of RFFT and IRFFT can share
// memory through an unsafe view. The direct paths build the half spectrum in
// dst, so an overlap must take the buffered path and give the same bits as
// separate slices.
func TestRealPlanOverlappingSlices(t *testing.T) {
	for _, n := range []int{8, 64, 256, 1000, 1024, 4096} {
		p := NewRealPlan(n)
		m := n / 2
		x := make([]float64, n)
		for i := range x {
			x[i] = math.Sin(0.37*float64(i)) + 0.25*math.Cos(1.3*float64(i))
		}
		want := p.RFFT(make([]complex128, m+1), x)

		back := p.IRFFT(make([]float64, n), want)
		// Exact aliasing happens to be harmless (the passes run in place);
		// a SHIFTED overlap is what would corrupt a transform that builds
		// its intermediate in dst. Offsets in complex128 elements.
		for _, off := range []int{0, 1, 3} {
			mem := make([]complex128, m+1+off)
			src := unsafe.Slice((*float64)(unsafe.Pointer(&mem[0])), n)
			copy(src, x)
			got := p.RFFT(mem[off:], src)
			for k := range want {
				if math.Float64bits(real(got[k])) != math.Float64bits(real(want[k])) || math.Float64bits(imag(got[k])) != math.Float64bits(imag(want[k])) {
					t.Fatalf("n=%d off=%d: overlapping RFFT bin %d = %v, want %v", n, off, k, got[k], want[k])
				}
			}
			mem2 := make([]complex128, m+1+off)
			copy(mem2[off:], want)
			dst := unsafe.Slice((*float64)(unsafe.Pointer(&mem2[0])), n)
			got2 := p.IRFFT(dst, mem2[off:])
			for i := range back {
				if math.Float64bits(got2[i]) != math.Float64bits(back[i]) {
					t.Fatalf("n=%d off=%d: overlapping IRFFT sample %d = %v, want %v", n, off, i, got2[i], back[i])
				}
			}
		}
	}
}

func TestRealOverlap(t *testing.T) {
	b := make([]byte, 64)
	p := func(i int) unsafe.Pointer { return unsafe.Pointer(&b[i]) }
	for _, c := range []struct {
		a, an, b, bn int
		want         bool
	}{{0, 16, 16, 16, false}, {0, 17, 16, 16, true}, {16, 16, 0, 16, false}, {8, 8, 0, 64, true}} {
		if got := realOverlap(p(c.a), c.an, p(c.b), c.bn); got != c.want {
			t.Errorf("realOverlap(%d,%d,%d,%d) = %v", c.a, c.an, c.b, c.bn, got)
		}
	}
}
