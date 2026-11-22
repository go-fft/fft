package kernels

import (
	"math"
	"testing"
	"unsafe"
)

func unsafeSlice(p *float64, n int) []float64 { return unsafe.Slice(p, n) }

// The run helpers hold the Go around the kernels (the copy of src when dst overlaps
// it). A pure-Go kernel stands in for the assembly so that this runs on any amd64
// CPU. It writes dst while it still reads src, like the assembly, so an overlapping
// call is right only because the helper copied src first.

func fakeRun2(n int) func(dst, src *float64) {
	return func(dst, src *float64) {
		d, s := unsafeSlice(dst, n), unsafeSlice(src, n)
		for k := 0; k < n; k++ {
			var sum float64
			for j := 0; j < n; j++ {
				sum += s[j] * math.Cos(math.Pi*float64((2*j+1)*k)/float64(2*n))
			}
			d[k] = 2 * sum
		}
	}
}

func fakeRun3(n int) func(dst, src *float64, a0 float64) {
	return func(dst, src *float64, a0 float64) {
		d, s := unsafeSlice(dst, n), unsafeSlice(src, n)
		for j := 0; j < n; j++ {
			sum := a0 * s[0]
			for k := 1; k < n; k++ {
				sum += 2 * s[k] * math.Cos(math.Pi*float64((2*j+1)*k)/float64(2*n))
			}
			d[j] = sum
		}
	}
}

func fake2x32(dst, src *float64)            { fakeRun2(32)(dst, src) }
func fake2x64(dst, src *float64)            { fakeRun2(64)(dst, src) }
func fake3x32(dst, src *float64, a float64) { fakeRun3(32)(dst, src, a) }
func fake3x64(dst, src *float64, a float64) { fakeRun3(64)(dst, src, a) }

func input(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(float64(i)*1.3) + 0.25*float64(i%5)
	}
	return x
}

// TestRunHelpersAlias: disjoint, identical and partly overlapping dst and src give
// the answer of a call on a separate copy of src.
func TestRunHelpersAlias(t *testing.T) {
	for _, off := range []int{0, 1, -1, 5, -5, 31, -31} {
		// 32 points
		x := input(32)
		var w2, w3 [32]float64
		fake2x32(&w2[0], &x[0])
		fake3x32(&w3[0], &x[0], 0.5)
		for _, kind := range []int{2, 3} {
			buf := make([]float64, 128)
			copy(buf[48:], x)
			src := (*[32]float64)(buf[48:])
			dst := (*[32]float64)(buf[48+off:])
			want := w2
			if kind == 2 {
				Run2x32(dst, src, fake2x32)
			} else {
				want = w3
				Run3x32(dst, src, 0.5, fake3x32)
			}
			for i := range want {
				if math.Abs(dst[i]-want[i]) > 1e-12 {
					t.Fatalf("32 dct%d offset %d: index %d got %g want %g", kind, off, i, dst[i], want[i])
				}
			}
		}
		// 64 points
		y := input(64)
		var v2, v3 [64]float64
		fake2x64(&v2[0], &y[0])
		fake3x64(&v3[0], &y[0], 0.5)
		for _, kind := range []int{2, 3} {
			buf := make([]float64, 256)
			copy(buf[96:], y)
			src := (*[64]float64)(buf[96:])
			dst := (*[64]float64)(buf[96+off:])
			want := v2
			if kind == 2 {
				Run2x64(dst, src, fake2x64)
			} else {
				want = v3
				Run3x64(dst, src, 0.5, fake3x64)
			}
			for i := range want {
				if math.Abs(dst[i]-want[i]) > 1e-12 {
					t.Fatalf("64 dct%d offset %d: index %d got %g want %g", kind, off, i, dst[i], want[i])
				}
			}
		}
	}
	// disjoint buffers: src stays untouched
	x := input(32)
	var d [32]float64
	s := [32]float64(x)
	Run2x32(&d, &s, fake2x32)
	if s != [32]float64(x) {
		t.Fatal("a disjoint call wrote src")
	}
}

// TestRunHelpersNoAllocs: the helpers with a constant kernel do not allocate, in
// place or not (the copy stays on the stack).
func TestRunHelpersNoAllocs(t *testing.T) {
	var a, b [32]float64
	var c [64]float64
	for name, f := range map[string]func(){
		"2x32 in place": func() { Run2x32(&a, &a, fake2x32) },
		"3x32 in place": func() { Run3x32(&a, &a, 1, fake3x32) },
		"2x64 in place": func() { Run2x64(&c, &c, fake2x64) },
		"3x64 in place": func() { Run3x64(&c, &c, 1, fake3x64) },
		"2x32 disjoint": func() { Run2x32(&a, &b, fake2x32) },
	} {
		if n := testing.AllocsPerRun(50, f); n != 0 {
			t.Errorf("%s: %v allocs, want 0", name, n)
		}
	}
}
