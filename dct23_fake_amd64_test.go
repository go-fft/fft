//go:build amd64

package fft

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/go-fft/fft/internal/kernels"
)

// The Go around the generated kernels (the overlap copy in internal/kernels and the
// routing in dct23_run_amd64.go) must be covered on a CPU without AVX-512, where the
// assembly cannot run: these tests put a pure-Go kernel in its place. It reads src
// while it writes dst, like the assembly would if the copy were missing, so an
// aliased call is right only because the wrapper detached src.

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

// useFakeKernels routes the 32- and 64-point calls to the pure-Go kernels on any CPU.
func useFakeKernels(t *testing.T) {
	t.Helper()
	isolatePlans(t)
	on, f2a, f3a, f2b, f3b := kernels.Dct23On, kernels.Dct2x32Fn, kernels.Dct3x32Fn, kernels.Dct2x64Fn, kernels.Dct3x64Fn
	kernels.Dct23On = true
	kernels.Dct2x32Fn, kernels.Dct3x32Fn = fakeDCT2(32), fakeDCT3(32)
	kernels.Dct2x64Fn, kernels.Dct3x64Fn = fakeDCT2(64), fakeDCT3(64)
	t.Cleanup(func() {
		kernels.Dct23On, kernels.Dct2x32Fn, kernels.Dct3x32Fn, kernels.Dct2x64Fn, kernels.Dct3x64Fn = on, f2a, f3a, f2b, f3b
	})
}

// TestDCT23RunOtherLength: with the kernels on, a length they do not serve is refused
// (the caller then takes the table route), and with them off every length is.
func TestDCT23RunOtherLength(t *testing.T) {
	useFakeKernels(t)
	a, b := make([]float64, 16), make([]float64, 16)
	if dct23Run2(a, b, 16) || dct23Run3(a, b, 16, 1) {
		t.Fatal("a 16-point call ran a kernel")
	}
	isolatePlans(t)
	kernels.Dct23On = false
	c, d := make([]float64, 32), make([]float64, 32)
	if dct23Run2(c, d, 32) || dct23Run3(c, d, 32, 1) {
		t.Fatal("a call ran a kernel with the gate off")
	}
}

// TestDCT23FakeKernelRouting: the routing and the aliasing copy, with the fake kernel,
// give the table route's values (to rounding) for disjoint, identical and partly
// overlapping dst and src, both transforms, both lengths.
func TestDCT23FakeKernelRouting(t *testing.T) {
	useFakeKernels(t)
	rng := rand.New(rand.NewSource(3))
	for _, n := range kernelLens() {
		d := NewDCTPlan(n, 2).fast
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64()
		}
		for _, typ := range []string{"dct2", "dct3"} {
			call := func(dst, src []float64) {
				if typ == "dct2" {
					d.forward(dst, src)
				} else {
					d.inverse(dst, src, 0.5)
				}
			}
			want := make([]float64, n)
			withKernel(t, false, func() { call(want, x) })
			for _, off := range []int{n, 0, 1, -1, n / 2, -(n - 1)} { // n: disjoint
				buf := make([]float64, 4*n)
				s0 := 2 * n
				copy(buf[s0:], x)
				src := buf[s0 : s0+n]
				dst := buf[s0+off : s0+off+n]
				if off == n {
					dst = buf[:n]
				}
				call(dst, src)
				for i := range want {
					if math.Abs(dst[i]-want[i]) > 1e-12*float64(n) {
						t.Fatalf("%s n=%d offset %d: index %d got %g want %g", typ, n, off, i, dst[i], want[i])
					}
				}
			}
		}
	}
}
