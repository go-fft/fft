package kernels

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// zmSignals are the inputs CMulTo is held to (Round 36): generic values,
// random draws from {±0} and from {±0, ±1} (a sum of signed zeros keeps or
// loses its sign by the order of its operands), ±∞ among finite values, and
// odd multiples of the smallest subnormal (a product of two underflows, of
// one and a finite value rounds).
func zmSignals(n int, seed uint64) [][]complex128 {
	r := rand.New(rand.NewPCG(36, seed))
	neg := math.Copysign(0, -1)
	pick := func(vs ...float64) float64 { return vs[r.IntN(len(vs))] }
	var g, z, o, inf, sub []complex128
	for i := range n {
		g = append(g, complex(r.NormFloat64(), r.NormFloat64()))
		z = append(z, complex(pick(0, neg), pick(0, neg)))
		o = append(o, complex(pick(0, neg, 1, -1), pick(0, neg, 1, -1)))
		v := complex(r.NormFloat64(), r.NormFloat64())
		if i%3 == 1 {
			v = complex(pick(math.Inf(1), math.Inf(-1)), imag(v))
		}
		if i%5 == 2 {
			v = complex(real(v), pick(math.Inf(1), math.Inf(-1), 0))
		}
		inf = append(inf, v)
		sub = append(sub, complex(float64(2*(i%7)+1)*5e-324*pick(1, -1), pick(0.75, 3, 1e300)*pick(1, -1)))
	}
	return [][]complex128{g, z, o, inf, sub}
}

// zmSame is bit equality, any NaN equal to any NaN: which NaN payload
// propagates is not something the transforms promise.
func zmSame(x, y complex128) bool {
	f := func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
	}
	return f(real(x), real(y)) && f(imag(x), imag(y))
}

// TestCMulToMatchesScalar holds CMulTo, out of place and in place, to Go's
// complex product point by point, for every length to 70 and a few long
// ones, every pair of signals. Under GOAMD64=v3 the compiler fuses the
// oracle's products into FMAs and this test fails on amd64, as the other Go
// oracles of the package do. On amd64 TestZmCMulKernelMatchesScalar runs
// it with the kernel on whatever the vendor.
func TestCMulToMatchesScalar(t *testing.T) { zmCheckCMulTo(t) }

func zmCheckCMulTo(t *testing.T) {
	sizes := []int{1021, 4096, 20480}
	for n := 0; n <= 70; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		as, bs := zmSignals(n, uint64(n)), zmSignals(n, uint64(n)+1000)
		for i, a := range as {
			for j, b := range bs {
				want := make([]complex128, n)
				for k := range want {
					want[k] = a[k] * b[k]
				}
				got := make([]complex128, n)
				CMulTo(got, a, b)
				in := slices.Clone(a)
				CMulTo(in, in, b)
				for k := range want {
					if !zmSame(got[k], want[k]) || !zmSame(in[k], want[k]) {
						t.Fatalf("n=%d signals %d·%d point %d: %v, in place %v, want %v", n, i, j, k, got[k], in[k], want[k])
					}
				}
			}
		}
	}
}

// TestCMulToLengths: CMulTo writes len(dst) points and no more, reads
// longer a and b only that far, and panics before writing anything when a
// or b is shorter than dst.
func TestCMulToLengths(t *testing.T) {
	// Small integers: every product is exact, fused or not.
	var a, b []complex128
	for k := range 9 {
		a = append(a, complex(float64(k+1), float64(2-k)))
		b = append(b, complex(float64(3*k-7), float64(k+5)))
	}
	dst := make([]complex128, 9)
	CMulTo(dst[:6], a, b)
	for k := range 9 {
		if want := a[k] * b[k]; k < 6 && !zmSame(dst[k], want) || k >= 6 && dst[k] != 0 {
			t.Fatalf("point %d: %v", k, dst[k])
		}
	}
	for _, c := range []struct{ a, b []complex128 }{{a[:8], b}, {a, b[:8]}, {a[:3], b}, {a, b[:2]}} {
		out := make([]complex128, 9)
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("len(a)=%d len(b)=%d: no panic", len(c.a), len(c.b))
				}
			}()
			CMulTo(out, c.a, c.b)
		}()
		if slices.ContainsFunc(out, func(v complex128) bool { return v != 0 }) {
			t.Fatalf("len(a)=%d len(b)=%d: wrote before panicking", len(c.a), len(c.b))
		}
	}
}
