package fft

import (
	"math"
	"math/big"
	"math/rand/v2"
	"sync"
	"testing"
)

// This file is the extended-precision DFT oracle the accuracy contract
// (docs/accuracy.md, accuracy_test.go) is measured against. It lives in test
// code only.
//
// The oracle is a double-double FFT (about 106 bits, 1e-32 relative):
//
//   - a mixed-radix decimation-in-time recursion, O(n·Σp) for n = Πp, so a
//     power of two, a smooth length or a prime of a few thousand points costs
//     little (a prime p is a direct O(p²) DFT, 1009 points take 0.1 s);
//   - every twiddle is computed once per length by math/big at 200 bits
//     (a Taylor series on [0, π/4], the other angles by exact symmetry) and
//     rounded to a double-double, never by a recurrence, so it is exact to
//     the oracle's own precision;
//   - the real, N-dimensional and r2r (DCT/DST) oracles are built from the
//     same complex transform, along each axis, or as the direct O(n²) sum
//     whose angles are indices into the same root table.
//
// It is validated against independently known values in TestOracle*.

// dd is a double-double: the unevaluated sum hi+lo with |lo| <= ulp(hi)/2.
type dd struct{ hi, lo float64 }

// cdd is a complex number of double-doubles.
type cdd struct{ re, im dd }

func twoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	e = (a - (s - bb)) + (b - bb)
	return
}

func quickTwoSum(a, b float64) (s, e float64) {
	s = a + b
	e = b - (s - a)
	return
}

func ddAdd(a, b dd) dd {
	s, e := twoSum(a.hi, b.hi)
	t, f := twoSum(a.lo, b.lo)
	e += t
	s, e = quickTwoSum(s, e)
	e += f
	s, e = quickTwoSum(s, e)
	return dd{s, e}
}

func ddNeg(a dd) dd { return dd{-a.hi, -a.lo} }

func ddSub(a, b dd) dd { return ddAdd(a, ddNeg(b)) }

func ddMul(a, b dd) dd {
	// The float64 conversions are load-bearing: without them gc on arm64
	// fuses p into the sum that follows (FMADDD) and the error-free product
	// is no longer one.
	p := float64(a.hi * b.hi)
	e := math.FMA(a.hi, b.hi, -p) // the exact error of the product
	e += float64(a.hi*b.lo) + float64(a.lo*b.hi)
	s, e := quickTwoSum(p, e)
	return dd{s, e}
}

// ddDivF is a/d for a float64 d.
func ddDivF(a dd, d float64) dd {
	q1 := a.hi / d
	p := float64(q1 * d)
	pe := math.FMA(q1, d, -p)
	r := ddSub(a, dd{p, pe})
	q2 := r.hi / d
	s, e := quickTwoSum(q1, q2)
	return dd{s, e}
}

// ddSqrtF is the square root of a positive float64, to double-double.
func ddSqrtF(a float64) dd {
	x := math.Sqrt(a)
	p := float64(x * x)
	pe := math.FMA(x, x, -p)
	r := ddSub(dd{a, 0}, dd{p, pe})
	s, e := quickTwoSum(x, r.hi/(2*x))
	return dd{s, e}
}

func cAdd(a, b cdd) cdd { return cdd{ddAdd(a.re, b.re), ddAdd(a.im, b.im)} }

func cMul(a, b cdd) cdd {
	return cdd{
		ddSub(ddMul(a.re, b.re), ddMul(a.im, b.im)),
		ddAdd(ddMul(a.re, b.im), ddMul(a.im, b.re)),
	}
}

func cScale(a cdd, s dd) cdd { return cdd{ddMul(a.re, s), ddMul(a.im, s)} }

func cConj(a cdd) cdd { return cdd{a.re, ddNeg(a.im)} }

// bigToDD rounds f to a double-double.
func bigToDD(f *big.Float) dd {
	hi, _ := f.Float64()
	r := new(big.Float).SetPrec(f.Prec()).Sub(f, new(big.Float).SetFloat64(hi))
	lo, _ := r.Float64()
	return dd{hi, lo}
}

const oraclePrec = 200

var bigPi = func() *big.Float {
	f, _, err := big.ParseFloat("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899", 10, 256, big.ToNearestEven)
	if err != nil {
		panic(err)
	}
	return f
}()

// bigCosSin returns cos x and sin x, 0 <= x <= π/4, by Taylor series.
func bigCosSin(x *big.Float) (c, s *big.Float) {
	nf := func(v int64) *big.Float { return new(big.Float).SetPrec(oraclePrec).SetInt64(v) }
	x2 := new(big.Float).SetPrec(oraclePrec).Mul(x, x)
	c, s = nf(1), new(big.Float).SetPrec(oraclePrec).Set(x)
	tc, ts := nf(1), new(big.Float).SetPrec(oraclePrec).Set(x)
	for i := int64(1); i <= 28; i++ {
		tc.Mul(tc, x2).Quo(tc, nf((2*i-1)*(2*i))).Neg(tc)
		ts.Mul(ts, x2).Quo(ts, nf((2*i)*(2*i+1))).Neg(ts)
		c.Add(c, tc)
		s.Add(s, ts)
	}
	return c, s
}

// ddRootTable holds exp(-2πi·k/N) for k = 0..N-1 as double-doubles.
type ddRootTable []cdd

var (
	rootMu    sync.Mutex
	rootCache = map[int]ddRootTable{}
)

// ddRoots returns the table of N-th roots of unity exp(-2πi·k/N).
func ddRoots(N int) ddRootTable {
	rootMu.Lock()
	defer rootMu.Unlock()
	if t, ok := rootCache[N]; ok {
		return t
	}
	// base[r] = cos, sin of π·r/(2N), r = 0..N/2 (angles up to π/4); the rest
	// of r < N by the complement π/2 - θ.
	base := make([][2]dd, N/2+1)
	for r := range base {
		x := new(big.Float).SetPrec(oraclePrec).Mul(bigPi, new(big.Float).SetPrec(oraclePrec).SetInt64(int64(r)))
		x.Quo(x, new(big.Float).SetPrec(oraclePrec).SetInt64(int64(2*N)))
		c, s := bigCosSin(x)
		base[r] = [2]dd{bigToDD(c), bigToDD(s)}
	}
	t := make(ddRootTable, N)
	for k := range t {
		m := 4 * k
		q, r := m/N, m%N
		var c, s dd
		if r <= N/2 {
			c, s = base[r][0], base[r][1]
		} else {
			c, s = base[N-r][1], base[N-r][0]
		}
		var C, S dd // cos, sin of 2πk/N
		switch q {
		case 0:
			C, S = c, s
		case 1:
			C, S = ddNeg(s), c
		case 2:
			C, S = ddNeg(c), ddNeg(s)
		default:
			C, S = s, ddNeg(c)
		}
		t[k] = cdd{C, ddNeg(S)}
	}
	rootCache[N] = t
	return t
}

func smallestPrimeFactor(n int) int {
	for p := 2; p*p <= n; p++ {
		if n%p == 0 {
			return p
		}
	}
	return n
}

// ddDFT is the unnormalized DFT of x (the forward exp(-2πi·jk/n), or its
// conjugate if inverse), in double-double.
func ddDFT(x []cdd, inverse bool) []cdd {
	return ddDFTTable(x, inverse, ddRoots(len(x)))
}

// ddDFTTable is ddDFT with the root table given: the mutation check of
// accuracy_test.go runs it with one twiddle perturbed.
func ddDFTTable(x []cdd, inverse bool, tw ddRootTable) []cdd {
	n := len(x)
	if n == 0 {
		return nil
	}
	return ddRec(x, 0, 1, n, tw, n, inverse)
}

func ddRec(in []cdd, off, step, n int, tw ddRootTable, N int, inverse bool) []cdd {
	if n == 1 {
		return []cdd{in[off]}
	}
	p := smallestPrimeFactor(n)
	m := n / p
	subs := make([][]cdd, p)
	for r := range subs {
		subs[r] = ddRec(in, off+r*step, step*p, m, tw, N, inverse)
	}
	out := make([]cdd, n)
	unit := N / n
	for k := 0; k < m; k++ {
		for q := 0; q < p; q++ {
			idx := k + m*q
			acc := subs[0][k]
			for r := 1; r < p; r++ {
				w := tw[unit*((r*idx)%n)%N]
				if inverse {
					w = cConj(w)
				}
				acc = cAdd(acc, cMul(subs[r][k], w))
			}
			out[idx] = acc
		}
	}
	return out
}

// ddInvN is 1/n as a double-double.
func ddInvN(n int) dd { return ddDivF(dd{1, 0}, float64(n)) }

func scaleAll(a []cdd, s dd) {
	for i := range a {
		a[i] = cScale(a[i], s)
	}
}

func prod(shape []int) int {
	p := 1
	for _, s := range shape {
		p *= s
	}
	return p
}

// applyAxis replaces every line of the row-major array a along axis by f(line).
func applyAxis(a []cdd, shape []int, axis int, f func([]cdd) []cdd) []cdd {
	out := make([]cdd, len(a))
	n := shape[axis]
	stride := prod(shape[axis+1:])
	outer := prod(shape[:axis])
	line := make([]cdd, n)
	for o := 0; o < outer; o++ {
		for s := 0; s < stride; s++ {
			base := o*n*stride + s
			for j := 0; j < n; j++ {
				line[j] = a[base+j*stride]
			}
			res := f(line)
			for j := 0; j < n; j++ {
				out[base+j*stride] = res[j]
			}
		}
	}
	return out
}

// oFFTN is the N-dimensional DFT over axes, scaled 1/Π(shape[axes]) if inverse.
func oFFTN(x []cdd, shape, axes []int, inverse bool) []cdd {
	out := x
	size := 1
	for _, ax := range axes {
		out = applyAxis(out, shape, ax, func(l []cdd) []cdd { return ddDFT(l, inverse) })
		size *= shape[ax]
	}
	if inverse {
		out = append([]cdd(nil), out...)
		scaleAll(out, ddInvN(size))
	}
	return out
}

func allAxes(nd int) []int {
	a := make([]int, nd)
	for i := range a {
		a[i] = i
	}
	return a
}

// hermitianFull extends the half spectrum h (n/2+1 bins) of a real signal of
// n points to all n bins; the imaginary parts of bin 0 and, for even n, of the
// Nyquist bin are ignored, as in IRFFT.
func hermitianFull(h []cdd, n int) []cdd {
	full := make([]cdd, n)
	for j := range full {
		switch {
		case j <= n/2:
			full[j] = h[j]
		default:
			full[j] = cConj(h[n-j])
		}
	}
	full[0].im = dd{}
	if n%2 == 0 {
		full[n/2].im = dd{}
	}
	return full
}

// oRFFT is the first n/2+1 bins of the DFT of the real x.
func oRFFT(x []cdd) []cdd { return ddDFT(x, false)[:len(x)/2+1] }

// oIRFFT is the real inverse DFT (1/n) of the half spectrum h for n points.
func oIRFFT(h []cdd, n int) []cdd {
	out := ddDFT(hermitianFull(h, n), true)
	scaleAll(out, ddInvN(n))
	return realOnly(out)
}

func realOnly(a []cdd) []cdd {
	out := make([]cdd, len(a))
	for i := range a {
		out[i].re = a[i].re
	}
	return out
}

// oRFFTN is RFFTN: the DFT of the real array, the last axis cut to n/2+1.
func oRFFTN(x []cdd, shape []int) []cdd {
	full := oFFTN(x, shape, allAxes(len(shape)), false)
	n := shape[len(shape)-1]
	rows := len(full) / n
	out := make([]cdd, 0, rows*(n/2+1))
	for r := 0; r < rows; r++ {
		out = append(out, full[r*n:r*n+n/2+1]...)
	}
	return out
}

// oIRFFTN is IRFFTN: h has shape with the last length n/2+1.
func oIRFFTN(h []cdd, shape []int) []cdd {
	nd := len(shape)
	n := shape[nd-1]
	hs := append(append([]int(nil), shape[:nd-1]...), n/2+1)
	// The complex inverse along every axis but the last, then the real inverse
	// along the last (a Hermitian extension of each row).
	cur := h
	size := n
	for ax := 0; ax < nd-1; ax++ {
		cur = applyAxis(cur, hs, ax, func(l []cdd) []cdd { return ddDFT(l, true) })
		size *= shape[ax]
	}
	rows := len(cur) / (n/2 + 1)
	out := make([]cdd, 0, rows*n)
	for r := 0; r < rows; r++ {
		out = append(out, ddDFT(hermitianFull(cur[r*(n/2+1):(r+1)*(n/2+1)], n), true)...)
	}
	out = realOnly(out)
	scaleAll(out, ddInvN(size))
	return out
}

// oHFFT is HFFT: the forward DFT of the Hermitian signal whose half is h, n
// real values, unscaled.
func oHFFT(h []cdd, n int) []cdd { return realOnly(ddDFT(hermitianFull(h, n), false)) }

// oIHFFT is IHFFT: conj(rfft(x))/n.
func oIHFFT(x []cdd) []cdd {
	out := oRFFT(x)
	for i := range out {
		out[i] = cConj(out[i])
	}
	scaleAll(out, ddInvN(len(x)))
	return out
}

// normScale is Norm.scale(m, inverse) as a double-double.
func normScale(norm Norm, m int, inverse bool) dd {
	if norm == NormOrtho {
		return ddDivDD1(ddSqrtF(float64(m)))
	}
	if (norm == NormBackward) == inverse {
		return ddInvN(m)
	}
	return dd{1, 0}
}

// ddDivDD1 is 1/s for a double-double s (one Newton step on the float64
// reciprocal, to double-double accuracy).
func ddDivDD1(s dd) dd {
	x := dd{1 / s.hi, 0}
	// x <- x + x(1 - s·x), twice
	for i := 0; i < 2; i++ {
		r := ddSub(dd{1, 0}, ddMul(s, x))
		x = ddAdd(x, ddMul(x, r))
	}
	return x
}

// oR2R is the DCT (cosine) or DST of type typ of the real line x (the
// transform of the inverse type, with the inverse scaling, if inverse), as the
// direct sum of the definitions in dct.go and dst.go, with the "ortho"
// corrections of scipy. Angles are indices into the root table of order D.
func oR2R(x []cdd, cosine bool, typ int, norm Norm, inverse bool) []cdd {
	n := len(x)
	t := typ
	if inverse {
		t = inverseType(typ)
	}
	in := append([]cdd(nil), x...)
	ortho := norm == NormOrtho
	r2 := ddSqrtF(2)
	if ortho {
		switch {
		case cosine && t == 1:
			in[0] = cScale(in[0], r2)
			in[n-1] = cScale(in[n-1], r2)
		case cosine && t == 3:
			in[0] = cScale(in[0], r2)
		case !cosine && t == 3:
			in[n-1] = cScale(in[n-1], r2)
		}
	}
	var D int
	switch {
	case t == 1 && cosine:
		D = 2 * (n - 1)
	case t == 1:
		D = 2 * (n + 1)
	case t == 4:
		D = 8 * n
	default:
		D = 4 * n
	}
	tw := ddRoots(D)
	trig := func(m int) dd { // cos or sin of 2π·m/D
		w := tw[((m%D)+D)%D]
		if cosine {
			return w.re
		}
		return ddNeg(w.im)
	}
	two := dd{2, 0}
	out := make([]cdd, n)
	for k := 0; k < n; k++ {
		var acc dd
		for j := 0; j < n; j++ {
			var m int
			switch {
			case t == 1 && cosine:
				m = j * k
			case t == 1:
				m = (j + 1) * (k + 1)
			case t == 2 && cosine:
				m = k * (2*j + 1)
			case t == 2:
				m = (k + 1) * (2*j + 1)
			case t == 3 && cosine:
				m = (2*k + 1) * j
			case t == 3:
				m = (2*k + 1) * (j + 1)
			default:
				m = (2*k + 1) * (2*j + 1)
			}
			v := ddMul(in[j].re, trig(m))
			// Every term has weight 2 but the endpoint terms the definitions
			// write without it: DCT-I x[0] and (-1)^k x[N-1] (cos(πk) is trig
			// at j = N-1), DCT-III x[0], DST-III (-1)^k x[N-1].
			if !((cosine && t == 1 && (j == 0 || j == n-1)) || (cosine && t == 3 && j == 0) || (!cosine && t == 3 && j == n-1)) {
				v = ddMul(v, two)
			}
			acc = ddAdd(acc, v)
		}
		out[k].re = acc
	}
	if ortho {
		switch {
		case cosine && t == 1:
			out[0].re = ddDivDDr(out[0].re, r2)
			out[n-1].re = ddDivDDr(out[n-1].re, r2)
		case cosine && t == 2:
			out[0].re = ddDivDDr(out[0].re, r2)
		case !cosine && t == 2:
			out[n-1].re = ddDivDDr(out[n-1].re, r2)
		}
	}
	M := logicalSize(cosine, t, n)
	s := normScale(norm, M, inverse)
	for k := range out {
		out[k].re = ddMul(out[k].re, s)
	}
	return out
}

// ddDivDDr is a/s for double-doubles.
func ddDivDDr(a, s dd) dd { return ddMul(a, ddDivDD1(s)) }

// --- conversions and errors ---

func cddFromC[T complex64 | complex128](x []T) []cdd {
	out := make([]cdd, len(x))
	for i, v := range x {
		c := complex128(v)
		out[i] = cdd{dd{real(c), 0}, dd{imag(c), 0}}
	}
	return out
}

func cddFromR[T float32 | float64](x []T) []cdd {
	out := make([]cdd, len(x))
	for i, v := range x {
		out[i].re = dd{float64(v), 0}
	}
	return out
}

// roundC rounds an oracle spectrum to the input type of a transform.
func roundC[T complex64 | complex128](a []cdd) []T {
	out := make([]T, len(a))
	for i, v := range a {
		out[i] = T(complex(v.re.hi, v.im.hi))
	}
	return out
}

// errC is ‖got − want‖₂ / ‖want‖₂ for complex results (the difference is
// formed against the double-double, so the oracle's own error is invisible).
func errC[T complex64 | complex128](got []T, want []cdd) float64 {
	var num, den float64
	for i, w := range want {
		g := complex128(got[i])
		dr := (real(g) - w.re.hi) - w.re.lo
		di := (imag(g) - w.im.hi) - w.im.lo
		num += dr*dr + di*di
		den += w.re.hi*w.re.hi + w.im.hi*w.im.hi
	}
	return math.Sqrt(num / den)
}

// errR is errC for real results (the oracle holds them in .re).
func errR[T float32 | float64](got []T, want []cdd) float64 {
	var num, den float64
	for i, w := range want {
		d := (float64(got[i]) - w.re.hi) - w.re.lo
		num += d * d
		den += w.re.hi * w.re.hi
	}
	return math.Sqrt(num / den)
}

// errDD is the relative RMS distance between two oracle results.
func errDD(a, b []cdd) float64 {
	var num, den float64
	for i := range a {
		dr := ddSub(a[i].re, b[i].re).hi
		di := ddSub(a[i].im, b[i].im).hi
		num += dr*dr + di*di
		den += b[i].re.hi*b[i].re.hi + b[i].im.hi*b[i].im.hi
	}
	return math.Sqrt(num / den)
}

func randSeq(n int, seed uint64) func() float64 {
	r := rand.New(rand.NewPCG(seed, uint64(n)*2+1))
	return func() float64 { return 2*r.Float64() - 1 }
}

func accRealF64(n int, seed uint64) []float64 {
	f := randSeq(n, seed)
	x := make([]float64, n)
	for i := range x {
		x[i] = f()
	}
	return x
}

func accComplexF64(n int, seed uint64) []complex128 {
	f := randSeq(n, seed)
	x := make([]complex128, n)
	for i := range x {
		x[i] = complex(f(), f())
	}
	return x
}

func accRealF32(n int, seed uint64) []float32 {
	f := randSeq(n, seed)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(f())
	}
	return x
}

func accComplexF32(n int, seed uint64) []complex64 {
	f := randSeq(n, seed)
	x := make([]complex64, n)
	for i := range x {
		x[i] = complex(float32(f()), float32(f()))
	}
	return x
}

// --- the oracle's own validation ---

// TestOracleRoots: the root table against values known in closed form and
// against its own identities (w[2k] = w[k]², |w| = 1), for lengths that
// exercise every quadrant and both halves of the octant symmetry.
func TestOracleRoots(t *testing.T) {
	const tol = 1e-30
	near := func(a, b dd) bool { return math.Abs(ddSub(a, b).hi) < tol }
	// N = 8: exp(-iπ/4) = (√2/2, -√2/2); N = 12: exp(-iπ/3) = (1/2, -√3/2).
	r2 := ddDivF(ddSqrtF(2), 2)
	w8 := ddRoots(8)
	if !near(w8[1].re, r2) || !near(w8[1].im, ddNeg(r2)) || !near(w8[3].re, ddNeg(r2)) || !near(w8[3].im, ddNeg(r2)) ||
		!near(w8[5].re, ddNeg(r2)) || !near(w8[5].im, r2) || !near(w8[7].re, r2) || !near(w8[7].im, r2) {
		t.Errorf("N=8 roots wrong: %v", w8)
	}
	w12 := ddRoots(12)
	if !near(w12[2].re, dd{0.5, 0}) || !near(w12[2].im, ddNeg(ddDivF(ddSqrtF(3), 2))) || !near(w12[4].re, dd{-0.5, 0}) {
		t.Errorf("N=12 roots wrong: %v", w12)
	}
	for _, N := range []int{1, 2, 3, 5, 7, 9, 16, 30, 97, 1000, 1009} {
		w := ddRoots(N)
		for k := 0; k < N; k++ {
			if m := ddAdd(ddMul(w[k].re, w[k].re), ddMul(w[k].im, w[k].im)); !near(m, dd{1, 0}) {
				t.Fatalf("N=%d k=%d: |w|² = %v", N, k, m)
			}
			c, s := math.Cos(2*math.Pi*float64(k)/float64(N)), -math.Sin(2*math.Pi*float64(k)/float64(N))
			if math.Abs(w[k].re.hi-c) > 1e-14 || math.Abs(w[k].im.hi-s) > 1e-14 {
				t.Fatalf("N=%d k=%d: (%v, %v) vs math (%v, %v)", N, k, w[k].re.hi, w[k].im.hi, c, s)
			}
		}
		if N%2 == 0 {
			w2 := ddRoots(N / 2)
			for k := 0; k < N/2; k++ {
				d := ddSub(w[2*k].re, w2[k].re)
				if !near(d, dd{}) {
					t.Fatalf("N=%d: w[2k] != w_{N/2}[k] at %d", N, k)
				}
			}
		}
	}
}

// TestOracleDFT: the recursion against a direct O(n²) double-double sum, a
// delta, a pure tone, Parseval and the inverse round trip.
func TestOracleDFT(t *testing.T) {
	const tol = 1e-29
	for _, n := range []int{1, 2, 3, 4, 6, 7, 12, 30, 49, 97, 128, 360, 1000} {
		x := cddFromC(accComplexF64(n, 7))
		got := ddDFT(x, false)
		w := ddRoots(n)
		want := make([]cdd, n)
		for k := range want {
			var acc cdd
			for j := range x {
				acc = cAdd(acc, cMul(x[j], w[(j*k)%n]))
			}
			want[k] = acc
		}
		if e := errDD(got, want); e > tol {
			t.Errorf("n=%d: recursion vs direct sum: %.3g", n, e)
		}
		back := ddDFT(got, true)
		scaleAll(back, ddInvN(n))
		if e := errDD(back, x); e > tol {
			t.Errorf("n=%d: round trip: %.3g", n, e)
		}
		var ex, eg float64
		for i := range x {
			ex += x[i].re.hi*x[i].re.hi + x[i].im.hi*x[i].im.hi
			eg += got[i].re.hi*got[i].re.hi + got[i].im.hi*got[i].im.hi
		}
		if math.Abs(eg/float64(n)-ex) > 1e-13*ex {
			t.Errorf("n=%d: Parseval %v vs %v", n, eg/float64(n), ex)
		}
	}
	// A tone exp(2πi·3j/n) concentrates in bin 3 with magnitude n.
	n := 60
	tw := ddRoots(n)
	tone := make([]cdd, n)
	for j := range tone {
		tone[j] = cConj(tw[(3*j)%n])
	}
	spec := ddDFT(tone, false)
	for k, v := range spec {
		want := 0.0
		if k == 3 {
			want = float64(n)
		}
		if math.Abs(v.re.hi-want) > 1e-28 || math.Abs(v.im.hi) > 1e-28 {
			t.Fatalf("tone: bin %d = %v", k, v)
		}
	}
}

// TestOracleR2R: the DCT/DST oracle against known small values and against the
// orthogonality of the "ortho" transforms (DCT-III is the inverse of DCT-II,
// types I and IV are involutions, whatever the length).
func TestOracleR2R(t *testing.T) {
	// DCT-II of (1, 2, 3) by hand: y[k] = 2 Σ x[j] cos(πk(2j+1)/6).
	got := oR2R(cddFromR([]float64{1, 2, 3}), true, 2, NormBackward, false)
	want := []float64{12, 2 * (math.Cos(math.Pi/6) + 2*math.Cos(math.Pi/2) + 3*math.Cos(5*math.Pi/6)), 2 * (math.Cos(2*math.Pi/6) + 2*math.Cos(2*math.Pi/2) + 3*math.Cos(10*math.Pi/6))}
	for k := range want {
		if math.Abs(got[k].re.hi-want[k]) > 1e-13 {
			t.Errorf("DCT-II[%d] = %v, want %v", k, got[k].re.hi, want[k])
		}
	}
	for _, cosine := range []bool{true, false} {
		for typ := 1; typ <= 4; typ++ {
			for _, norm := range []Norm{NormBackward, NormOrtho, NormForward} {
				n := 9
				x := cddFromR(accRealF64(n, 3))
				y := oR2R(x, cosine, typ, norm, false)
				z := oR2R(y, cosine, typ, norm, true)
				if e := errDD(z, x); e > 1e-29 {
					t.Errorf("cosine=%v type %d %v: inverse(forward) off by %.3g", cosine, typ, norm, e)
				}
			}
		}
	}
	// Ortho DCT-II is orthogonal: it preserves the 2-norm.
	x := cddFromR(accRealF64(16, 5))
	y := oR2R(x, true, 2, NormOrtho, false)
	var a, b float64
	for i := range x {
		a += x[i].re.hi * x[i].re.hi
		b += y[i].re.hi * y[i].re.hi
	}
	if math.Abs(a-b) > 1e-13*a {
		t.Errorf("ortho DCT-II does not preserve the norm: %v vs %v", a, b)
	}
}
