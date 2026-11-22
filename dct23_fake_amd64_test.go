//go:build amd64

package fft

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

// The routes in dct23_run_amd64.go must be covered on a CPU without AVX-512, where
// the assembly cannot run: these tests hand them pure-Go kernels. The aliasing copy
// is internal/kernels' (dct23_run_amd64_test.go there).

func fakeDCT2(n int) func(dst, src *float64) {
	return func(dst, src *float64) {
		d, s := unsafe.Slice(dst, n), unsafe.Slice(src, n)
		for k := 0; k < n; k++ {
			var sum float64
			for j := 0; j < n; j++ {
				sum += s[j] * math.Cos(math.Pi*float64((2*j+1)*k)/float64(2*n))
			}
			d[k] = 2 * sum
		}
	}
}

func fakeDCT3(n int) func(dst, src *float64, a0 float64) {
	return func(dst, src *float64, a0 float64) {
		d, s := unsafe.Slice(dst, n), unsafe.Slice(src, n)
		for j := 0; j < n; j++ {
			var sum float64
			for k := 1; k < n; k++ {
				sum += s[k] * math.Cos(math.Pi*float64((2*j+1)*k)/float64(2*n))
			}
			d[j] = a0*s[0] + 2*sum
		}
	}
}

func fake2x32(dst, src *float64)             { fakeDCT2(32)(dst, src) }
func fake2x64(dst, src *float64)             { fakeDCT2(64)(dst, src) }
func fake3x32(dst, src *float64, a0 float64) { fakeDCT3(32)(dst, src, a0) }
func fake3x64(dst, src *float64, a0 float64) { fakeDCT3(64)(dst, src, a0) }

// TestDCT23RouteFake: a route with ok set calls its kernel and gives the table
// route's values (to rounding); with ok clear it does nothing and says so, so that
// the caller takes the table route.
func TestDCT23RouteFake(t *testing.T) {
	isolatePlans(t)
	rng := rand.New(rand.NewSource(3))
	for _, n := range kernelLens() {
		d := NewDCTPlan(n, 2).fast
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		for _, typ := range []string{"dct2", "dct3"} {
			want, got := make([]float64, n), make([]float64, n)
			var ran bool
			switch {
			case typ == "dct2" && n == 32:
				d.forwardTable(want, x)
				ran = dct23Route2x32(got, x, true, fake2x32)
			case typ == "dct2":
				d.forwardTable(want, x)
				ran = dct23Route2x64(got, x, true, fake2x64)
			case n == 32:
				d.inverseTable(want, x, 0.5)
				ran = dct23Route3x32(got, x, 0.5, true, fake3x32)
			default:
				d.inverseTable(want, x, 0.5)
				ran = dct23Route3x64(got, x, 0.5, true, fake3x64)
			}
			if !ran {
				t.Fatalf("%s n=%d: the route refused", typ, n)
			}
			for i := range want {
				if math.Abs(got[i]-want[i]) > 1e-12*float64(n) {
					t.Fatalf("%s n=%d: index %d got %g want %g", typ, n, i, got[i], want[i])
				}
			}
		}
	}

	// not ok: no kernel runs, nothing is written
	a, b := make([]float64, 64), make([]float64, 64)
	for i := range b {
		b[i] = 1
	}
	if dct23Route2x32(a, b, false, fake2x32) || dct23Route2x64(a, b, false, fake2x64) ||
		dct23Route3x32(a, b, 1, false, fake3x32) || dct23Route3x64(a, b, 1, false, fake3x64) {
		t.Fatal("a route ran with ok clear")
	}
	for _, v := range a {
		if v != 0 {
			t.Fatal("a route wrote dst with ok clear")
		}
	}
}

// TestDCT23RunGate: the production entry points refuse a length the kernels do not
// serve and every length with the gate off (the caller then takes the table route).
func TestDCT23RunGate(t *testing.T) {
	isolatePlans(t)
	old := kernels.Dct23On
	t.Cleanup(func() { kernels.Dct23On = old })
	kernels.Dct23On = false
	c, e := make([]float64, 32), make([]float64, 32)
	if dct23Run2(c, e, 32) || dct23Run3(c, e, 32, 1) {
		t.Fatal("a call ran a kernel with the gate off")
	}
	kernels.Dct23On = true // a length refused here never reaches an assembly kernel
	a, b := make([]float64, 16), make([]float64, 16)
	if dct23Run2(a, b, 16) || dct23Run3(a, b, 16, 1) {
		t.Fatal("a 16-point call ran a kernel")
	}
}

// TestDCT23NoAllocs holds the escape analysis: an in-place call and a call on a
// caller's stack array must not allocate (a package-level kernel variable made the
// alias copy and both slices escape, one 256-byte allocation per in-place call).
// It runs on the route in force: the assembly kernels where AVX-512 is present,
// the table route elsewhere, and the routes with the pure-Go kernels everywhere.
func TestDCT23NoAllocs(t *testing.T) {
	isolatePlans(t)
	old := kernels.Dct23On
	t.Cleanup(func() { kernels.Dct23On = old })
	routes := []bool{false}
	if kernels.UseStockhamAVX512 {
		routes = append(routes, true)
	}
	for _, on := range routes {
		kernels.Dct23On = on
		for _, n := range kernelLens() {
			p := NewDCTPlan(n, 2)
			x := make([]float64, n)
			for i := range x {
				x[i] = float64(i%7) - 3
			}
			check := func(name string, f func()) {
				t.Helper()
				f() // warm the pools
				if a := testing.AllocsPerRun(50, f); a != 0 {
					t.Errorf("%s n=%d kernel=%v: %v allocs per call, want 0", name, n, on, a)
				}
			}
			check("DCT in place", func() { p.DCT(x, x, NormBackward) })
			check("IDCT in place", func() { p.IDCT(x, x, NormBackward) })
			if n == 32 {
				var s [32]float64
				for i := range s {
					s[i] = float64(i)
				}
				check("DCT on a stack array", func() { p.DCT(s[:], s[:], NormBackward) })
				check("IDCT on a stack array", func() { p.IDCT(s[:], s[:], NormBackward) })
			}
		}
	}

	// the routes with the pure-Go kernels (constants here, as in production),
	// in place and on the stack
	var s32 [32]float64
	var s64 [64]float64
	for name, f := range map[string]func(){
		"route2 32": func() { dct23Route2x32(s32[:], s32[:], true, fake2x32) },
		"route3 32": func() { dct23Route3x32(s32[:], s32[:], 1, true, fake3x32) },
		"route2 64": func() { dct23Route2x64(s64[:], s64[:], true, fake2x64) },
		"route3 64": func() { dct23Route3x64(s64[:], s64[:], 1, true, fake3x64) },
	} {
		if a := testing.AllocsPerRun(50, f); a != 0 {
			t.Errorf("%s: %v allocs per call, want 0", name, a)
		}
	}
}
