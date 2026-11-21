//go:build amd64

package fft

import (
	"math"
	"math/rand"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The generated 32- and 64-point kernels (internal/gendct23) against the table
// route they replace, which stays the fallback and the oracle's neighbour: the
// accuracy against the extended-precision oracle is TestDCT23Oracle's; these tests
// pin what only a kernel can get wrong (aliasing, stray writes, special values).

func kernelLens() []int { return []int{32, 64} }

func withKernel(on bool, f func()) {
	old := kernels.Dct23On
	kernels.Dct23On = on
	defer func() { kernels.Dct23On = old }()
	f()
}

// TestDCT23KernelMatchesRoute: on random data both routes agree to a few ulps of the
// largest output (they are different FMA groupings of the same sums).
func TestDCT23KernelMatchesRoute(t *testing.T) {
	if !kernels.Dct23On {
		t.Skip("the kernels are not routed on this CPU")
	}
	rng := rand.New(rand.NewSource(1))
	for _, n := range kernelLens() {
		d := NewDCTPlan(n, 2).fast
		for it := 0; it < 200; it++ {
			x := make([]float64, n)
			for i := range x {
				x[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(8)-4))
			}
			a0 := 1.0
			if it%2 == 1 {
				a0 = 0.5
			}
			for _, typ := range []string{"dct2", "dct3"} {
				got, want := make([]float64, n), make([]float64, n)
				run := func(dst []float64) {
					if typ == "dct2" {
						d.forward(dst, x)
					} else {
						d.inverse(dst, x, a0)
					}
				}
				withKernel(true, func() { run(got) })
				withKernel(false, func() { run(want) })
				var big float64
				for _, v := range want {
					big = math.Max(big, math.Abs(v))
				}
				for i := range got {
					if e := math.Abs(got[i] - want[i]); e > 64*eps64*big {
						t.Fatalf("%s n=%d it=%d index %d: kernel %g route %g (err %.3g of %.3g)", typ, n, it, i, got[i], want[i], e, big)
					}
				}
			}
		}
	}
}

// TestDCT23KernelAliasing: dst equal to src, or overlapping it partially, gives the
// bits of the disjoint call.
func TestDCT23KernelAliasing(t *testing.T) {
	if !kernels.Dct23On {
		t.Skip("the kernels are not routed on this CPU")
	}
	rng := rand.New(rand.NewSource(2))
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
					d.inverse(dst, src, 1)
				}
			}
			want := make([]float64, n)
			call(want, x)
			for _, off := range []int{0, 1, 3, n / 2, n - 1, -1, -3, -(n / 2), -(n - 1)} {
				buf := make([]float64, 3*n)
				s0 := n
				copy(buf[s0:], x)
				src := buf[s0 : s0+n]
				dst := buf[s0+off : s0+off+n]
				call(dst, src)
				for i := range want {
					if dst[i] != want[i] {
						t.Fatalf("%s n=%d offset %d: index %d got %g want %g", typ, n, off, i, dst[i], want[i])
					}
				}
			}
		}
	}
}

// TestDCT23KernelWritesOnlyDst: nothing outside dst[:n] changes, src is not modified.
func TestDCT23KernelWritesOnlyDst(t *testing.T) {
	if !kernels.Dct23On {
		t.Skip("the kernels are not routed on this CPU")
	}
	const canary = 7.25
	for _, n := range kernelLens() {
		d := NewDCTPlan(n, 2).fast
		for _, typ := range []string{"dct2", "dct3"} {
			buf := make([]float64, 3*n)
			for i := range buf {
				buf[i] = canary
			}
			src := make([]float64, n)
			for i := range src {
				src[i] = float64(i + 1)
			}
			keep := append([]float64(nil), src...)
			dst := buf[n : 2*n]
			if typ == "dct2" {
				d.forward(dst, src)
			} else {
				d.inverse(dst, src, 1)
			}
			for i, v := range buf {
				if (i < n || i >= 2*n) && v != canary {
					t.Fatalf("%s n=%d: buf[%d] = %g, outside dst", typ, n, i, v)
				}
			}
			for i := range src {
				if src[i] != keep[i] {
					t.Fatalf("%s n=%d: src[%d] modified", typ, n, i)
				}
			}
		}
	}
}

// TestDCT23KernelSpecial: infinities, NaNs, signed zeros and subnormals run without
// a fault, a NaN in the input reaches the output, and the finite-but-tiny inputs
// stay within the absolute floor of the route.
func TestDCT23KernelSpecial(t *testing.T) {
	if !kernels.Dct23On {
		t.Skip("the kernels are not routed on this CPU")
	}
	for _, n := range kernelLens() {
		d := NewDCTPlan(n, 2).fast
		for _, typ := range []string{"dct2", "dct3"} {
			call := func(dst, src []float64) {
				if typ == "dct2" {
					d.forward(dst, src)
				} else {
					d.inverse(dst, src, 1)
				}
			}
			for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
				for pos := 0; pos < n; pos += 5 {
					src := make([]float64, n)
					src[pos] = bad
					dst := make([]float64, n)
					call(dst, src)
					if math.IsNaN(bad) {
						any := false
						for _, v := range dst {
							any = any || math.IsNaN(v)
						}
						if !any {
							t.Errorf("%s n=%d: a NaN at %d vanished", typ, n, pos)
						}
					}
				}
			}
			tiny := make([]float64, n)
			for i := range tiny {
				tiny[i] = math.SmallestNonzeroFloat64 * float64(i+1)
				if i%2 == 1 {
					tiny[i] = math.Copysign(0, -1)
				}
			}
			got, want := make([]float64, n), make([]float64, n)
			withKernel(true, func() { call(got, tiny) })
			withKernel(false, func() { call(want, tiny) })
			for i := range got {
				if math.Abs(got[i]-want[i]) > 1e-300 {
					t.Errorf("%s n=%d subnormal index %d: kernel %g route %g", typ, n, i, got[i], want[i])
				}
			}
		}
	}
}

// FuzzDCT23Kernel: arbitrary finite inputs, kernel against the route.
func FuzzDCT23Kernel(f *testing.F) {
	f.Add(int64(1), uint8(0))
	f.Add(int64(77), uint8(1))
	f.Fuzz(func(t *testing.T, seed int64, which uint8) {
		if !kernels.Dct23On {
			t.Skip("the kernels are not routed on this CPU")
		}
		rng := rand.New(rand.NewSource(seed))
		n := kernelLens()[int(which)%2]
		d := NewDCTPlan(n, 2).fast
		x := make([]float64, n)
		for i := range x {
			x[i] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(40)-20))
		}
		for _, typ := range []string{"dct2", "dct3"} {
			got, want := make([]float64, n), make([]float64, n)
			run := func(dst []float64) {
				if typ == "dct2" {
					d.forward(dst, x)
				} else {
					d.inverse(dst, x, 1)
				}
			}
			withKernel(true, func() { run(got) })
			withKernel(false, func() { run(want) })
			var big float64
			for _, v := range want {
				big = math.Max(big, math.Abs(v))
			}
			for i := range got {
				if math.Abs(got[i]-want[i]) > 64*eps64*big {
					t.Fatalf("%s n=%d seed %d index %d: %g vs %g", typ, n, seed, i, got[i], want[i])
				}
			}
		}
	})
}
