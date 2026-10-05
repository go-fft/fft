package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestStockhamPassMatchesScalarNEON runs every smooth length up to 2100, and a
// few larger ones, through the Stockham engine twice — NEON pass kernels on,
// then off — and requires the two results to be bit-identical, forward and
// inverse, in place and not. gc compiles the Go passes with fused multiply-adds
// on arm64, so this holds the kernels to the very products gc fuses. The range
// covers radix-4 passes at ido == 2, at long ido and as the final pass, with
// many and few blocks, and the lengths whose radix-4 passes the kernels leave
// to Go (odd ido, odd l1).
func TestStockhamPassMatchesScalarNEON(t *testing.T) {
	defer func(v bool) { kernels.UseStockhamNEON = v }(kernels.UseStockhamNEON)
	sizes := []int{4096, 8192, 16384, 20160, 45000, 65536, 65536 * 3, 1 << 17}
	for n := 2; n <= 2100; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		if !factorsAreSmall(n) {
			continue
		}
		for _, factors := range [][]int{skFactorize(n), pow2Radices4(n)} {
			if factors == nil {
				continue
			}
			p := newSKPlanFactors(n, factors)
			for s, x := range neonSignals(n) {
				for _, inverse := range []bool{false, true} {
					kernels.UseStockhamNEON = false
					scalar := make([]complex128, n)
					p.transform(scalar, x, inverse)
					kernels.UseStockhamNEON = true
					simd := make([]complex128, n)
					p.transform(simd, x, inverse)
					alias := append([]complex128(nil), x...)
					p.transform(alias, alias, inverse)
					for i := range scalar {
						if !neonSameBits(simd[i], scalar[i]) || !neonSameBits(alias[i], scalar[i]) {
							t.Fatalf("n=%d factors %v signal %d inverse=%v index %d: %v (in place %v) vs scalar %v",
								n, factors, s, inverse, i, simd[i], alias[i], scalar[i])
						}
					}
				}
			}
		}
	}
}

// pow2Radices4 returns the all-radix-4 factorization of a power of four, which
// puts radix-4 passes at every ido, or nil for any other n.
func pow2Radices4(n int) []int {
	if n&(n-1) != 0 || n < 4 {
		return nil
	}
	e := 0
	for m := n; m > 1; m >>= 1 {
		e++
	}
	if e%2 != 0 {
		return nil
	}
	return pow2Radices(e, false)
}

// neonSignals is the amd64 test's simdSignals: a generic signal, then signed
// zeros and infinities, which tell a multiply by one from no multiply.
func neonSignals(n int) [][]complex128 {
	neg := math.Copysign(0, -1)
	zeros := make([]complex128, n)
	mixed := make([]complex128, n)
	inf := cmplxSignal(n)
	for i := range zeros {
		zeros[i] = complex(neg, neg)
		re, im := neg, 0.0
		if i%3 == 0 {
			re = 0
		}
		if i%2 == 0 {
			im = neg
		}
		if i%5 == 0 {
			re = -1
		}
		mixed[i] = complex(re, im)
	}
	inf[n/2] = complex(math.Inf(1), 0)
	// Constant inputs make many butterfly sums exactly zero,
	// which is where -(a+b) and (-a)-b, or a fused and an unfused product of
	// a zero, would part: their sign survives into the output.
	// Pseudo-random draws from {+0, -0} and from {+0, -0, +1, -1} per
	// component reach those zero sums with every combination of signs.
	flat := make([]complex128, n)
	zsigns := make([]complex128, n)
	signs := make([]complex128, n)
	vals := [4]float64{0, neg, 1, -1}
	seed := uint32(1)
	draw := func(k uint32) float64 {
		seed = seed*1664525 + 1013904223
		return vals[seed>>(32-k)]
	}
	// Subnormals make even the exact-looking products round (0.5·t of an odd
	// multiple of the smallest subnormal), which tells a fused 0.5 product
	// from an unfused one.
	tiny := make([]complex128, n)
	for i := range flat {
		flat[i] = complex(1.5, -0.5)
		zsigns[i] = complex(draw(1), draw(1))
		signs[i] = complex(draw(2), draw(2))
		tiny[i] = complex(float64(2*(i%7)+1)*5e-324, -float64(2*(i%5)+1)*5e-324)
	}
	return [][]complex128{cmplxSignal(n), zeros, mixed, inf, flat, zsigns, signs, tiny}
}

// neonSameBits compares bit patterns; two NaNs count as equal whatever their
// payload, since IEEE 754 leaves the payload of an invalid operation open.
func neonSameBits(a, b complex128) bool {
	eq := func(x, y float64) bool {
		return math.Float64bits(x) == math.Float64bits(y) || (math.IsNaN(x) && math.IsNaN(y))
	}
	return eq(real(a), real(b)) && eq(imag(a), imag(b))
}

// TestStockhamEachPassMatchesScalarNEON compares every pass a kernel runs with
// its Go pass on its own, output against output: through a whole transform a
// zero's sign that differs inside one pass is mostly absorbed by later sums,
// so only a pass-level comparison holds the kernels to it.
func TestStockhamEachPassMatchesScalarNEON(t *testing.T) {
	defer func(v bool) { kernels.UseStockhamNEON = v }(kernels.UseStockhamNEON)
	kernels.UseStockhamNEON = true
	for _, n := range []int{8, 16, 32, 64, 128, 256, 512, 2048, 4096, 8192, 960, 1000, 1080, 4000, 20160} {
		for _, factors := range [][]int{skFactorize(n), pow2Radices4(n)} {
			if factors == nil {
				continue
			}
			p := newSKPlanFactors(n, factors)
			for k := range p.stages {
				st := &p.stages[k]
				for s, x := range neonSignals(n) {
					for _, inverse := range []bool{false, true} {
						scalar := make([]complex128, n)
						st.passScalar(scalar, x, inverse)
						simd := make([]complex128, n)
						st.pass(simd, x, inverse)
						for i := range scalar {
							if !neonSameBits(simd[i], scalar[i]) {
								t.Fatalf("n=%d factors %v pass %d (r=%d ido=%d) signal %d inverse=%v index %d: %v vs scalar %v",
									n, factors, k, st.r, st.ido, s, inverse, i, simd[i], scalar[i])
							}
						}
					}
				}
			}
		}
	}
}
