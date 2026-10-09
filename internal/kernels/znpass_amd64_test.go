package kernels

import (
	"math"
	"math/rand/v2"
	"testing"
)

// znSignals are the inputs a ZnPlan is held to its passes on: generic
// values, signed zeros, a ±0/−1 mix, an infinity, random draws from {±0}
// and {±0, ±1}, and odd multiples of the smallest subnormal.
func znSignals(n int) [][]complex128 {
	rng := rand.New(rand.NewPCG(32, uint64(n)))
	neg0 := math.Copysign(0, -1)
	pick := func(vals []float64) []complex128 {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(vals[rng.IntN(len(vals))], vals[rng.IntN(len(vals))])
		}
		return x
	}
	gen := make([]complex128, n)
	zeros := make([]complex128, n)
	mix := make([]complex128, n)
	inf := make([]complex128, n)
	sub := make([]complex128, n)
	for i := range gen {
		gen[i] = complex(math.Sin(float64(i)*0.7)+0.25, math.Cos(float64(i)*1.3)-0.5)
		zeros[i] = complex(neg0, 0)
		if i%2 == 1 {
			zeros[i] = complex(0, neg0)
		}
		mix[i] = complex(neg0, -1)
		if i%3 == 0 {
			mix[i] = complex(-1, 0)
		}
		inf[i] = gen[i]
		sub[i] = complex(float64(2*i+1)*math.SmallestNonzeroFloat64, -float64(2*i+3)*math.SmallestNonzeroFloat64)
	}
	inf[n/3] = complex(math.Inf(1), 0)
	return [][]complex128{gen, zeros, mix, inf, pick([]float64{0, neg0}), pick([]float64{0, neg0, 1, -1}), sub}
}

// znRoot is the n-point root table the fft package builds.
func znRoot(n int) []complex128 {
	root := make([]complex128, n)
	for k := range root {
		s, c := math.Sincos(-2 * math.Pi * float64(k) / float64(n))
		root[k] = complex(c, s)
	}
	return root
}

func znSame(a, b complex128) bool {
	return math.Float64bits(real(a)) == math.Float64bits(real(b)) && math.Float64bits(imag(a)) == math.Float64bits(imag(b))
}

// TestZnKernelAddrs: every kernel a ZnPlan can call has an entry point, and
// no slot without a kernel has one.
func TestZnKernelAddrs(t *testing.T) {
	for r := range 21 {
		has := stockhamSIMD(r)
		if (znAddrs[r] != 0) != has || (znAddrs[21+r] != 0) != has {
			t.Errorf("radix %d: entry points %#x %#x, kernels %v", r, znAddrs[r], znAddrs[21+r], has)
		}
	}
	for d := range 2 {
		for r := range 9 {
			for m := range 5 {
				has := m >= 1 && (r == 4 || r == 8)
				if got := znAddrs[42+(9*d+r)*5+m]; (got != 0) != has {
					t.Errorf("split d=%d r=%d m=%d: %#x", d, r, m, got)
				}
			}
		}
	}
	seen := map[uintptr]bool{}
	for _, a := range znAddrs {
		if a != 0 && seen[a] {
			t.Errorf("entry point %#x twice", a)
		}
		seen[a] = true
	}
}
