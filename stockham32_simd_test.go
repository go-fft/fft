//go:build amd64 || arm64

package fft

import (
	"math"
	"runtime"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// The float32 pass kernels (kernels.StockhamPass32) against the Go passes of
// stockham32_passes.go, bit for bit. On amd64 gc compiles the Go passes
// without FMA (GOAMD64=v1) and the AVX2 kernels round every operation
// separately; on arm64 gc fuses some products into FMADDS/FMSUBS and the NEON
// kernels fuse exactly those.

// f32simdFactorings returns the radix orders the tests run n through: the
// one Plan32 uses, the odd-first one, and, for powers of four, all radix 4
// (radix-4 passes at every ido).
func f32simdFactorings(n int) [][]int {
	fs := [][]int{skFactorizeOrder(n, false), skFactorizeOrder(n, true)}
	if n&(n-1) == 0 && n >= 4 {
		e := 0
		for m := n; m > 1; m >>= 1 {
			e++
		}
		if e%2 == 0 {
			fs = append(fs, pow2Radices(e, false))
		}
	}
	return fs
}

// f32simdSignals are the float64 SIMD tests' signals in float32: a generic
// signal, signed zeros and infinities (a multiply by one is not exact on
// them), constants and draws from {±0} and {±0, ±1}, whose zero sums tell
// -(a+b) from (-a)-b, and odd multiples of the smallest float32 subnormal,
// where even 0.5·t rounds, so a fused product differs from an unfused one.
func f32simdSignals(n int) [][]complex64 {
	neg := float32(math.Copysign(0, -1))
	gen := make([]complex64, n)
	zeros := make([]complex64, n)
	mixed := make([]complex64, n)
	inf := make([]complex64, n)
	flat := make([]complex64, n)
	zsigns := make([]complex64, n)
	signs := make([]complex64, n)
	tiny := make([]complex64, n)
	vals := [4]float32{0, neg, 1, -1}
	seed := uint32(1)
	draw := func(k uint32) float32 {
		seed = seed*1664525 + 1013904223
		return vals[seed>>(32-k)]
	}
	sub := float32(math.SmallestNonzeroFloat32)
	for i, v := range cmplxSignal(n) {
		gen[i] = complex64(v)
		inf[i] = gen[i]
		re, im := neg, float32(0)
		if i%3 == 0 {
			re = 0
		}
		if i%2 == 0 {
			im = neg
		}
		if i%5 == 0 {
			re = -1
		}
		zeros[i] = complex(neg, neg)
		mixed[i] = complex(re, im)
		flat[i] = complex(1.5, -0.5)
		zsigns[i] = complex(draw(1), draw(1))
		signs[i] = complex(draw(2), draw(2))
		tiny[i] = complex(float32(2*(i%7)+1)*sub, -float32(2*(i%5)+1)*sub)
	}
	inf[n/2] = complex(float32(math.Inf(1)), 0)
	return [][]complex64{gen, zeros, mixed, inf, flat, zsigns, signs, tiny}
}

// f32simdMatch compares bit patterns, two NaNs counting as equal whatever
// their sign and payload (IEEE 754 leaves an invalid operation's open).
// Under -race on arm64 gc compiles some Go passes with other fusions (as it
// does the float64 ones, Round 18), so there the kernel only has to agree to
// rounding: within 1e-6·log2(n) of tol, the largest magnitude of the Go
// result. The bit-for-bit comparison is the non-race build's (the
// arch-native CI job).
func f32simdMatch(a, b complex64, tol float64) bool {
	eq := func(x, y float32) bool {
		return math.Float32bits(x) == math.Float32bits(y) || (x != x && y != y)
	}
	if eq(real(a), real(b)) && eq(imag(a), imag(b)) {
		return true
	}
	near := func(x, y float32) bool { return x == y || math.Abs(float64(x-y)) <= tol }
	return raceEnabled && runtime.GOARCH == "arm64" && near(real(a), real(b)) && near(imag(a), imag(b))
}

// f32simdTol is f32simdMatch's tolerance for the Go result y.
func f32simdTol(y []complex64) float64 {
	m := 1.0
	for _, v := range y {
		m = max(m, math.Abs(float64(real(v))), math.Abs(float64(imag(v))))
	}
	return 1e-6 * m * math.Log2(float64(len(y))+1)
}

func TestStockham32PassMatchesScalar(t *testing.T) {
	if !kernels.UseStockham32 {
		t.Skip("no float32 pass kernels on this machine")
	}
	defer func(v bool) { kernels.UseStockham32 = v }(kernels.UseStockham32)
	sizes := []int{4096, 8192, 16384, 20160, 45000, 65536, 65536 * 3, 1 << 17}
	for n := 2; n <= 2100; n++ {
		sizes = append(sizes, n)
	}
	if testing.Short() {
		sizes = sizes[:3]
		for n := 2; n <= 300; n++ {
			sizes = append(sizes, n)
		}
	}
	for _, n := range sizes {
		if !factorsAreSmall(n) {
			continue
		}
		for _, factors := range f32simdFactorings(n) {
			p := newSKPlan32Factors(n, factors)
			for s, x := range f32simdSignals(n) {
				for _, inverse := range []bool{false, true} {
					kernels.UseStockham32 = false
					scalar := make([]complex64, n)
					p.transform(scalar, x, inverse)
					kernels.UseStockham32 = true
					simd := make([]complex64, n)
					p.transform(simd, x, inverse)
					alias := append([]complex64(nil), x...)
					p.transform(alias, alias, inverse)
					tol := f32simdTol(scalar)
					for i := range scalar {
						if !f32simdMatch(simd[i], scalar[i], tol) || !f32simdMatch(alias[i], scalar[i], tol) {
							t.Fatalf("n=%d factors %v signal %d inverse=%v index %d: %v (in place %v) vs Go %v",
								n, factors, s, inverse, i, simd[i], alias[i], scalar[i])
						}
					}
				}
			}
		}
	}
}

// TestStockham32EachPassMatchesScalar compares every pass with its Go pass on
// its own, output for output: through a whole transform a zero's sign that
// differs inside one pass is mostly absorbed by later sums.
func TestStockham32EachPassMatchesScalar(t *testing.T) {
	if !kernels.UseStockham32 {
		t.Skip("no float32 pass kernels on this machine")
	}
	defer func(v bool) { kernels.UseStockham32 = v }(kernels.UseStockham32)
	kernels.UseStockham32 = true
	for _, n := range []int{8, 12, 16, 20, 24, 32, 40, 48, 64, 72, 120, 128, 256, 512, 2048, 4096, 8192, 960, 1000, 1080, 4000, 20160} {
		for _, factors := range f32simdFactorings(n) {
			p := newSKPlan32Factors(n, factors)
			for k := range p.stages {
				st := &p.stages[k]
				for s, x := range f32simdSignals(n) {
					for _, inverse := range []bool{false, true} {
						scalar := make([]complex64, n)
						st.passScalar(scalar, x, inverse)
						simd := make([]complex64, n)
						st.pass(simd, x, inverse)
						tol := f32simdTol(scalar)
						for i := range scalar {
							if !f32simdMatch(simd[i], scalar[i], tol) {
								t.Fatalf("n=%d factors %v pass %d (r=%d ido=%d l1=%d) signal %d inverse=%v index %d: %v vs Go %v",
									n, factors, k, st.r, st.ido, st.l1, s, inverse, i, simd[i], scalar[i])
							}
						}
					}
				}
			}
		}
	}
}
