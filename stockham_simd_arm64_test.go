package fft

import (
	"math"
	"slices"
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
		for _, factors := range [][]int{skFactorize(n), pow2Radices4(n), compOtherOrder(n)} {
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
						if !neonMatch(simd[i], scalar[i]) || !neonMatch(alias[i], scalar[i]) {
							t.Fatalf("n=%d factors %v signal %d inverse=%v index %d: %v (in place %v) vs scalar %v",
								n, factors, s, inverse, i, simd[i], alias[i], scalar[i])
						}
					}
				}
			}
		}
	}
}

// compOtherOrder is n's factorization in the radix order this architecture
// does not take (skFactorizeOrder), so both orders reach the kernels: the
// odd-first order ends on a radix-2, 4 or 8 final pass over an odd number of
// blocks. It is nil when the two orders agree.
func compOtherOrder(n int) []int {
	f, g := skFactorize(n), skFactorizeOrder(n, !compOddFirst)
	if slices.Equal(f, g) {
		return nil
	}
	return g
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
	for _, n := range []int{8, 16, 32, 64, 128, 256, 512, 2048, 4096, 8192, 960, 1000, 1080, 4000, 20160, 15, 45, 75, 120, 135, 375, 1296, 1920, 2000, 6000} {
		for _, factors := range [][]int{skFactorize(n), pow2Radices4(n), compOtherOrder(n)} {
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
						in := x
						if st.split >= 2 { // the kernel reads block-split
							in = toSplit(x)
						}
						st.pass(simd, in, inverse)
						if st.split == 1 || st.split == 2 { // and writes block-split
							simd = fromSplit(simd)
						}
						for i := range scalar {
							if !neonMatch(simd[i], scalar[i]) {
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

// neonMatch is neonSameBits, except under -race: a race build compiles the Go
// passes with other fusions (pass5last rounds the other product of a pair,
// Go 1.27.1), so they are not the code the kernels copy, and there the two
// only have to agree to rounding, component by component. The arch-native CI job runs these tests
// without -race, bit for bit.
func neonMatch(a, b complex128) bool {
	if neonSameBits(a, b) {
		return true
	}
	near := func(x, y float64) bool { return x == y || math.Abs(x-y) <= 1e-12*(1+math.Abs(y)) }
	return raceEnabled && near(real(a), real(b)) && near(imag(a), imag(b))
}

// TestStockhamBatchMatchesLinesNEON holds the batched pass kernels to the 1-D
// passes, line by line and bit for bit: every batched radix, l1 and ido (1,
// the final pass, through odd and even), batches of 1 to 9 lines (an odd one
// ends with a lone lane), and point strides wider than the batch on either
// side, forward and inverse, on every neonSignals signal. Each line of the
// batch is gathered, run through the stage's 1-D pass (st.pass: the NEON
// kernel where there is one, the Go pass otherwise) and compared with the
// batch's output for that line. Outputs start as a sentinel, so a kernel that
// wrote between the rows of the strip fails too.
func TestStockhamBatchMatchesLinesNEON(t *testing.T) {
	defer func(v bool) { kernels.UseStockhamBatchNEON = v }(kernels.UseStockhamBatchNEON)
	kernels.UseStockhamBatchNEON = true
	sentinel := complex(-7.25, 3.5)
	for _, r := range []int{2, 3, 4, 5, 8} {
		for _, l1 := range []int{1, 2, 3, 5} {
			for _, ido := range []int{1, 2, 3, 4, 5, 7, 8, 13} {
				var factors []int
				stage := 0
				if l1 > 1 {
					factors, stage = append(factors, l1), 1
				}
				factors = append(factors, r)
				if ido > 1 {
					factors = append(factors, ido)
				}
				n := r * ido * l1
				st := newSKPlanFactors(n, factors).stages[stage]
				if st.r != r || st.l1 != l1 || st.ido != ido {
					t.Fatalf("stage %+v, want r=%d l1=%d ido=%d", st, r, l1, ido)
				}
				st.split = 0 // the 1-D pass on interleaved lines
				fwd, conj := st.batchTwiddles()
				for _, w := range []int{1, 2, 3, 4, 5, 8, 9} {
					for _, gap := range [][2]int{{0, 0}, {3, 0}, {0, 1}, {2, 5}} {
						sIn, sOut := w+gap[0], w+gap[1]
						size := (n-1)*sIn + w
						for s, x := range neonSignals(size) {
							for _, inverse := range []bool{false, true} {
								tw := fwd
								if inverse {
									tw = conj
								}
								out := make([]complex128, (n-1)*sOut+w)
								for i := range out {
									out[i] = sentinel
								}
								if !kernels.StockhamBatchPass(r, ido, l1, x, out, tw, w, sIn, sOut, inverse) {
									t.Fatalf("r=%d: no batched kernel", r)
								}
								line, want := make([]complex128, n), make([]complex128, n)
								for c := 0; c < w; c++ {
									for p := range line {
										line[p] = x[p*sIn+c]
									}
									st.pass(want, line, inverse)
									for p := range want {
										if got := out[p*sOut+c]; !neonMatch(got, want[p]) {
											t.Fatalf("r=%d l1=%d ido=%d w=%d sIn=%d sOut=%d signal %d inverse=%v line %d point %d: batch %v vs 1-D %v",
												r, l1, ido, w, sIn, sOut, s, inverse, c, p, got, want[p])
										}
									}
								}
								for p := 0; p < n-1; p++ {
									for c := w; c < sOut; c++ {
										if out[p*sOut+c] != sentinel {
											t.Fatalf("r=%d l1=%d ido=%d w=%d sOut=%d: wrote between rows at point %d, column %d", r, l1, ido, w, sOut, p, c)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	// A radix without a batched kernel is refused, and the Go pass runs.
	if kernels.StockhamBatchPass(7, 1, 1, make([]complex128, 7), make([]complex128, 7), nil, 1, 1, 1, false) {
		t.Fatal("radix 7 has no batched kernel")
	}
	kernels.UseStockhamBatchNEON = false
	if kernels.StockhamBatchPass(4, 1, 1, make([]complex128, 4), make([]complex128, 4), nil, 1, 1, 1, false) {
		t.Fatal("UseStockhamBatchNEON off: the kernel ran")
	}
}

// toSplit returns x in the block-split layout of the split kernels: the pair
// of points 2q, 2q+1 as re(2q), re(2q+1), im(2q), im(2q+1). len(x) is even.
func toSplit(x []complex128) []complex128 {
	y := make([]complex128, len(x))
	for q := 0; q < len(x); q += 2 {
		y[q] = complex(real(x[q]), real(x[q+1]))
		y[q+1] = complex(imag(x[q]), imag(x[q+1]))
	}
	return y
}

// fromSplit is toSplit's inverse.
func fromSplit(y []complex128) []complex128 {
	x := make([]complex128, len(y))
	for q := 0; q < len(y); q += 2 {
		x[q] = complex(real(y[q]), real(y[q+1]))
		x[q+1] = complex(imag(y[q]), imag(y[q+1]))
	}
	return x
}

// TestStockhamBatchGoPassNEON runs the Go batched pass, which the kernels
// replace here, against them: gc fuses its complex products differently from
// the 1-D passes (scalarFuses), so the two agree to rounding, not to the bit,
// and exactly where a value is not finite.
func TestStockhamBatchGoPassNEON(t *testing.T) {
	defer func(v bool) { kernels.UseStockhamBatchNEON = v }(kernels.UseStockhamBatchNEON)
	for _, r := range []int{2, 3, 4, 5, 8} {
		for _, f := range [][]int{{r, 3}, {2, r}} {
			n := f[0] * f[1]
			p := newSKPlanFactors(n, f)
			for k := range p.stages {
				st := p.stages[k]
				fwd, conj := st.batchTwiddles()
				const w, sIn, sOut = 5, 7, 6
				x := cmplxSignal((n-1)*sIn + w)
				for _, inverse := range []bool{false, true} {
					tw := fwd
					if inverse {
						tw = conj
					}
					var out [2][]complex128
					for i, on := range []bool{true, false} {
						kernels.UseStockhamBatchNEON = on
						out[i] = make([]complex128, (n-1)*sOut+w)
						st.passBatch(out[i], x, tw, w, sIn, sOut, inverse)
					}
					scale := maxPart(out[0])
					for i := range out[0] {
						near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-12*scale }
						if !near(real(out[0][i]), real(out[1][i])) || !near(imag(out[0][i]), imag(out[1][i])) {
							t.Fatalf("radices %v pass %d inverse=%v index %d: kernel %v, Go %v", f, k, inverse, i, out[0][i], out[1][i])
						}
					}
				}
			}
		}
	}
}
