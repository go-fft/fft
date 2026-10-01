package fft

import (
	"math/cmplx"
	"sync"
	"testing"
)

var planNShapes = [][]int{
	{1}, {7}, {3, 5}, {8, 8}, {4, 9}, {9, 4}, {16, 12}, {2, 3, 5}, {5, 1, 7}, {1, 6, 1},
	{6, 10, 3}, {2, 2, 2, 2}, {30, 17}, {64, 64}, {3, 129}, {129, 3},
}

func maxAbsDiff(a, b []complex128) float64 {
	var m float64
	for i := range a {
		m = max(m, cmplx.Abs(a[i]-b[i]))
	}
	return m
}

// TestPlanNAgainstNaive checks PlanN.FFT against the naive separable DFT, out
// of place and in place, and that IFFT inverts it.
func TestPlanNAgainstNaive(t *testing.T) {
	for _, shape := range planNShapes {
		p := NewPlanN(shape...)
		x := complexGrid(shape)
		want := naiveDFTN(x, shape)
		got := p.FFT(make([]complex128, len(x)), x)
		tol := 1e-9 * float64(len(x))
		if d := maxAbsDiff(got, want); d > tol {
			t.Fatalf("shape %v: FFT differs from the naive DFT by %g", shape, d)
		}
		in := append([]complex128(nil), x...)
		p.FFT(in, in)
		if d := maxAbsDiff(in, want); d > tol {
			t.Fatalf("shape %v: in-place FFT differs by %g", shape, d)
		}
		back := p.IFFT(in, in)
		if d := maxAbsDiff(back, x); d > 1e-9 {
			t.Fatalf("shape %v: IFFT(FFT(x)) differs from x by %g", shape, d)
		}
		if p.Len() != len(x) || len(p.Shape()) != len(shape) {
			t.Fatalf("shape %v: Len %d Shape %v", shape, p.Len(), p.Shape())
		}
	}
}

// TestPlanNParallelMatchesSerial runs shapes large enough to fan out with the
// worker pool forced on and off; the lines are independent, so the results
// must be identical, including at block edges (a last axis that is not a
// multiple of the 8-line gather block).
func TestPlanNParallelMatchesSerial(t *testing.T) {
	for _, shape := range [][]int{{256, 256}, {300, 70}, {70, 300}, {40, 41, 43}, {512, 37}} {
		p := NewPlanN(shape...)
		x := complexGrid(shape)
		var serial, parallel []complex128
		withWorkers(1, func() { serial = p.FFT(make([]complex128, len(x)), x) })
		withWorkers(8, func() { parallel = p.FFT(make([]complex128, len(x)), x) })
		for i := range serial {
			if serial[i] != parallel[i] {
				t.Fatalf("shape %v index %d: serial %v parallel %v", shape, i, serial[i], parallel[i])
			}
		}
	}
}

// TestPlanNAllocatesNothing is the point of PlanN: in steady state a serial
// transform allocates nothing.
func TestPlanNAllocatesNothing(t *testing.T) {
	if raceEnabled {
		t.Skip("-race makes sync.Pool drop items on purpose; the arch CI jobs run this without it")
	}
	withWorkers(1, func() {
		for _, shape := range [][]int{{64, 64}, {16, 12, 10}, {1000, 3}} {
			p := NewPlanN(shape...)
			x := complexGrid(shape)
			dst := make([]complex128, len(x))
			p.FFT(dst, x) // warm the scratch pools
			if a := testing.AllocsPerRun(20, func() { p.FFT(dst, x) }); a != 0 {
				t.Errorf("shape %v: FFT allocates %v times per call", shape, a)
			}
			if a := testing.AllocsPerRun(20, func() { p.IFFT(dst, dst) }); a != 0 {
				t.Errorf("shape %v: IFFT allocates %v times per call", shape, a)
			}
		}
	})
}

// TestPlanNConcurrent shares one plan across goroutines (run under -race).
func TestPlanNConcurrent(t *testing.T) {
	shape := []int{48, 40}
	p := NewPlanN(shape...)
	x := complexGrid(shape)
	want := p.FFT(make([]complex128, len(x)), x)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := p.FFT(make([]complex128, len(x)), x)
			for i := range want {
				if got[i] != want[i] {
					t.Errorf("concurrent FFT differs at %d", i)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestPlanNPanics(t *testing.T) {
	expectPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s did not panic", name)
			}
		}()
		f()
	}
	expectPanic("zero length", func() { NewPlanN(4, 0) })
	expectPanic("negative length", func() { NewPlanN(-1) })
	p := NewPlanN(4, 4)
	expectPanic("short dst", func() { p.FFT(make([]complex128, 15), make([]complex128, 16)) })
	expectPanic("short src", func() { p.FFT(make([]complex128, 16), make([]complex128, 15)) })
	if NewPlanN().Len() != 1 {
		t.Error("an empty shape is the single-element array")
	}
}

func TestBlockWidth(t *testing.T) {
	for _, c := range [][2]int{{1, 8}, {64, 8}, {1024, 8}, {2048, 4}, {4096, 2}, {8192, 1}, {100000, 1}} {
		if got := blockWidth(c[0]); got != c[1] {
			t.Errorf("blockWidth(%d) = %d, want %d", c[0], got, c[1])
		}
	}
}

// TestInPlacePow2KernelAllocatesNothing pins the pow2 kernel's in-place path,
// which every architecture can reach (directly, or above the per-architecture
// Stockham limit): it used to copy its input with append on every call.
func TestInPlacePow2KernelAllocatesNothing(t *testing.T) {
	if raceEnabled {
		t.Skip("-race makes sync.Pool drop items on purpose")
	}
	for _, n := range []int{8, 64, 1024, 1 << 14} {
		p := newITPlan(n)
		x := complexGrid([]int{n})
		p.transform(x, x, false)
		if a := testing.AllocsPerRun(20, func() { p.transform(x, x, false) }); a != 0 {
			t.Errorf("n=%d: in-place transform allocates %v times per call", n, a)
		}
	}
}
