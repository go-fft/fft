package fft

import (
	"flag"
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
	"testing"
)

// The accuracy contract (docs/accuracy.md): for every public transform, in
// both precisions, with the default arithmetic and with Options.FMA, the
// relative RMS error against the double-double oracle of oracle_test.go is at
// most
//
//	c · ε · log2(n)        (n = the number of points, at least 2)
//
// with ε = 2^-52 (float64) or 2^-23 (float32) and c a constant per function
// family and precision, fixed from the measurements below and listed in
// accBound. Run
//
//	go test -run TestAccuracyBound -v -accuracy.table
//
// to print the worst measured ratio err/(ε·log2 n) of every function and of
// every family, the numbers the constants were chosen from.

var accTable = flag.Bool("accuracy.table", false, "print the measured accuracy table (TestAccuracyBound)")

const (
	eps64 = 0x1p-52 // DBL_EPSILON
	// eps32 is FLT_EPSILON, 2^-23 (plan32_test.go).
)

// accBound is c per family: err <= c·ε·log2(n). The families are the groups of
// functions whose algorithms share an error behaviour: a complex 1-D
// transform, a real one, the N-dimensional complex and real ones, and the
// DCT/DST. The values come from the table printed by -accuracy.table: each is
// the first multiple of 0.25 at or above twice the worst ratio measured over
// every function of the family, every length, both seeds, the default and the
// FMA arithmetic.
//
// Worst measured ratios (arm64, Go 1.27, 2026-11-11; default and FMA arithmetic
// alike, the FMA ones within 0.05 of the default: c2c 0.64 against 0.60) and
// the constant chosen:
//
//	family    what                                   worst   c
//	c2c       FFT, IFFT, Plan, FFTReal                0.64   1.5
//	r2c       RFFT, IRFFT, HFFT, IHFFT, RealPlan      0.54   1.25
//	c2cN      FFTN, IFFTN, FFT2, IFFT2, PlanN         0.30   0.75
//	r2cN      RFFTN, IRFFTN, RFFT2, IRFFT2, RealPlan2 0.28   0.75
//	r2r       DCT/DST I-IV, inverses, plans, N-D      0.95   2.0
//	c2c/32    the float32 counterparts                0.34   0.75
//	r2c/32                                            0.43   1.0
//	c2cN/32                                           0.15   0.5
//	r2cN/32                                           0.14   0.5
//	r2r/32                                            0.86   1.75
//
// The worst ratios sit at the shortest lengths (19, 3, 2), where log2(n) is
// smallest and the error does not shrink with it; the long lengths, Rader and
// Bluestein included, are at 0.2 to 0.4. docs/accuracy.md has the rest.
var accBound = map[string]float64{
	"c2c": 1.5, "r2c": 1.25, "c2cN": 0.75, "r2cN": 0.75, "r2r": 2,
	"c2c/32": 0.75, "r2c/32": 1, "c2cN/32": 0.5, "r2cN/32": 0.5, "r2r/32": 1.75,
}

// accFn is one public function (or method) under test.
type accFn struct {
	family string
	name   string
	shapes [][]int
	// run transforms a seeded random input of the given shape and returns the
	// relative RMS error against the oracle. fma selects Options.FMA; the
	// float32 and r2r functions ignore it (hasFMA false).
	run    func(shape []int, seed uint64, fma bool) float64
	hasFMA bool
}

func (f accFn) points(shape []int) int { return prod(shape) }

// ratio is err/(ε·log2 n).
func accRatio(err float64, n int, eps float64) float64 {
	return err / (eps * math.Max(1, math.Log2(float64(n))))
}

var (
	memoMu sync.Mutex
	memo   = map[string][]cdd{}
)

// oracleMemo computes an oracle result once per key: several functions and
// both arithmetics compare against the same one.
func oracleMemo(key string, f func() []cdd) []cdd {
	memoMu.Lock()
	defer memoMu.Unlock()
	if v, ok := memo[key]; ok {
		return v
	}
	v := f()
	memo[key] = v
	return v
}

// accLarge reports whether the long lengths run: not under -race or -short,
// and not on the emulated architectures (loong64, ppc64le, riscv64, s390x
// under qemu), where the whole suite already takes most of the CI timeout.
func accLarge() bool {
	if raceEnabled || testing.Short() {
		return false
	}
	return runtime.GOARCH == "amd64" || runtime.GOARCH == "arm64"
}

func accLengths() []int {
	pow2 := []int{2, 4, 8, 16, 64, 256, 1024, 4096}
	pow3 := []int{3, 9, 27, 243, 2187}
	pow5 := []int{5, 25, 125, 625, 3125}
	pow7 := []int{7, 49, 343, 2401}
	mixed := []int{6, 12, 15, 30, 60, 360, 1000, 1001, 2310}
	rader := []int{17, 31, 127, 257, 1009}        // primes whose p-1 is 7-smooth
	bluestein := []int{19, 323, 1013, 2018, 2039} // everything else with a large prime factor
	all := append(append(append(append(append(append([]int{}, pow2...), pow3...), pow5...), pow7...), mixed...), rader...)
	all = append(all, bluestein...)
	if !accLarge() {
		var small []int
		for _, n := range all {
			if n <= 1100 {
				small = append(small, n)
			}
		}
		return small
	}
	return all
}

// accRealLengths are the real transforms' lengths: the complex ones (the
// even ones run an n/2-point complex transform) and the odd lengths.
func accRealLengths() []int {
	odd := []int{9, 15, 105, 343, 1001, 1009, 1013, 3125}
	var out []int
	for _, n := range append(accLengths(), odd...) {
		if n <= 1100 || accLarge() {
			out = append(out, n)
		}
	}
	return out
}

func accShapesND() [][]int {
	s := [][]int{{4, 4}, {32, 32}, {30, 31}, {64, 48}, {127, 33}, {15, 21}, {8, 9, 10}, {16, 16, 16}, {7, 5, 16}, {17, 3, 5, 4}}
	if accLarge() {
		s = append(s, []int{128, 128}, []int{1009, 3})
	}
	return s
}

func accShapes2D() [][]int {
	var out [][]int
	for _, s := range accShapesND() {
		if len(s) == 2 {
			out = append(out, s)
		}
	}
	return out
}

func accLines(ns []int) [][]int {
	out := make([][]int, len(ns))
	for i, n := range ns {
		out[i] = []int{n}
	}
	return out
}

func accR2RLengths() []int {
	if !accLarge() {
		return []int{2, 3, 8, 17, 100, 257}
	}
	return []int{2, 3, 8, 17, 100, 257, 1000, 1009}
}

// --- float64 ---

func accFns64() []accFn {
	cplx := func(inverse bool, f func(x []complex128, fma bool) []complex128) func([]int, uint64, bool) float64 {
		return func(shape []int, seed uint64, fma bool) float64 {
			n := shape[0]
			x := accComplexF64(n, seed)
			want := oracleMemo(fmt.Sprintf("c2c/%v/%d/%d", inverse, n, seed), func() []cdd {
				return oFFTN(cddFromC(x), shape, []int{0}, inverse)
			})
			return errC(f(x, fma), want)
		}
	}
	plan := func(n int, fma bool) *Plan {
		if fma {
			return NewPlanWith(n, Options{FMA: true})
		}
		return NewPlan(n)
	}
	cfns := []accFn{
		{family: "c2c", name: "FFT", hasFMA: true, run: cplx(false, func(x []complex128, fma bool) []complex128 {
			if fma {
				return FFTWith(x, Options{FMA: true})
			}
			return FFT(x)
		})},
		{family: "c2c", name: "IFFT", hasFMA: true, run: cplx(true, func(x []complex128, fma bool) []complex128 {
			if fma {
				return IFFTWith(x, Options{FMA: true})
			}
			return IFFT(x)
		})},
		{family: "c2c", name: "Plan.FFT", hasFMA: true, run: cplx(false, func(x []complex128, fma bool) []complex128 {
			return plan(len(x), fma).FFT(make([]complex128, len(x)), x)
		})},
		{family: "c2c", name: "Plan.IFFT", hasFMA: true, run: cplx(true, func(x []complex128, fma bool) []complex128 {
			return plan(len(x), fma).IFFT(make([]complex128, len(x)), x)
		})},
	}
	for i := range cfns {
		cfns[i].shapes = accLines(accLengths())
	}
	fftReal := accFn{family: "c2c", name: "FFTReal", shapes: accLines(accLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
		n := shape[0]
		x := accRealF64(n, seed)
		want := oracleMemo(fmt.Sprintf("fftreal/%d/%d", n, seed), func() []cdd { return ddDFT(cddFromR(x), false) })
		return errC(FFTReal(x), want)
	}}
	cfns = append(cfns, fftReal)

	// Real 1-D.
	realFwd := func(name string, f func(x []float64, fma bool) []complex128) accFn {
		return accFn{family: "r2c", name: name, hasFMA: true, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
			n := shape[0]
			x := accRealF64(n, seed)
			want := oracleMemo(fmt.Sprintf("rfft/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(x)) })
			return errC(f(x, fma), want)
		}}
	}
	spectrum := func(n int, seed uint64) []complex128 {
		return accSpectrum[complex128](oracleMemo(fmt.Sprintf("rfft/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF64(n, seed))) }), n)
	}
	realInv := func(name string, f func(s []complex128, n int, fma bool) []float64) accFn {
		return accFn{family: "r2c", name: name, hasFMA: true, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
			n := shape[0]
			s := spectrum(n, seed)
			want := oracleMemo(fmt.Sprintf("irfft/%d/%d", n, seed), func() []cdd { return oIRFFT(cddFromC(s), n) })
			return errR(f(s, n, fma), want)
		}}
	}
	rfns := []accFn{
		realFwd("RFFT", func(x []float64, fma bool) []complex128 {
			if fma {
				return RFFTWith(x, Options{FMA: true})
			}
			return RFFT(x)
		}),
		realFwd("RealPlan.RFFT", func(x []float64, fma bool) []complex128 {
			p := NewRealPlan(len(x))
			if fma {
				p = NewRealPlanWith(len(x), Options{FMA: true})
			}
			return p.RFFT(make([]complex128, len(x)/2+1), x)
		}),
		realInv("IRFFT", func(s []complex128, n int, fma bool) []float64 {
			if fma {
				return IRFFTWith(s, Options{N: n, FMA: true})
			}
			return IRFFT(s, n)
		}),
		realInv("RealPlan.IRFFT", func(s []complex128, n int, fma bool) []float64 {
			p := NewRealPlan(n)
			if fma {
				p = NewRealPlanWith(n, Options{FMA: true})
			}
			return p.IRFFT(make([]float64, n), s)
		}),
	}
	rfns = append(rfns,
		accFn{family: "r2c", name: "HFFT", hasFMA: true, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
			n := shape[0]
			s := spectrum(n, seed)
			want := oracleMemo(fmt.Sprintf("hfft/%d/%d", n, seed), func() []cdd { return oHFFT(cddFromC(s), n) })
			var got []float64
			if fma {
				got = HFFTWith(s, Options{N: n, FMA: true})
			} else {
				got = HFFT(s, n)
			}
			return errR(got, want)
		}},
		accFn{family: "r2c", name: "IHFFT", hasFMA: true, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
			n := shape[0]
			x := accRealF64(n, seed)
			want := oracleMemo(fmt.Sprintf("ihfft/%d/%d", n, seed), func() []cdd { return oIHFFT(cddFromR(x)) })
			var got []complex128
			if fma {
				got = IHFFTWith(x, Options{FMA: true})
			} else {
				got = IHFFT(x)
			}
			return errC(got, want)
		}},
	)

	// N-dimensional complex.
	nd := func(name string, shapes [][]int, inverse bool, f func(x []complex128, shape []int, fma bool) []complex128) accFn {
		return accFn{family: "c2cN", name: name, hasFMA: true, shapes: shapes, run: func(shape []int, seed uint64, fma bool) float64 {
			n := prod(shape)
			x := accComplexF64(n, seed)
			want := oracleMemo(fmt.Sprintf("fftn/%v/%v/%d", inverse, shape, seed), func() []cdd {
				return oFFTN(cddFromC(x), shape, allAxes(len(shape)), inverse)
			})
			return errC(f(x, shape, fma), want)
		}}
	}
	planN := func(shape []int, fma bool) *PlanN {
		if fma {
			return NewPlanNWith(shape, Options{FMA: true})
		}
		return NewPlanN(shape...)
	}
	ndfns := []accFn{
		nd("FFTN", accShapesND(), false, func(x []complex128, s []int, fma bool) []complex128 {
			if fma {
				return FFTNWith(x, s, Options{FMA: true})
			}
			return FFTN(x, s)
		}),
		nd("IFFTN", accShapesND(), true, func(x []complex128, s []int, fma bool) []complex128 {
			if fma {
				return IFFTNWith(x, s, Options{FMA: true})
			}
			return IFFTN(x, s)
		}),
		nd("FFT2", accShapes2D(), false, func(x []complex128, s []int, fma bool) []complex128 {
			if fma {
				return FFT2With(x, [2]int{s[0], s[1]}, Options{FMA: true})
			}
			return FFT2(x, [2]int{s[0], s[1]})
		}),
		nd("IFFT2", accShapes2D(), true, func(x []complex128, s []int, fma bool) []complex128 {
			if fma {
				return IFFT2With(x, [2]int{s[0], s[1]}, Options{FMA: true})
			}
			return IFFT2(x, [2]int{s[0], s[1]})
		}),
		nd("PlanN.FFT", accShapesND(), false, func(x []complex128, s []int, fma bool) []complex128 {
			return planN(s, fma).FFT(make([]complex128, len(x)), x)
		}),
		nd("PlanN.IFFT", accShapesND(), true, func(x []complex128, s []int, fma bool) []complex128 {
			return planN(s, fma).IFFT(make([]complex128, len(x)), x)
		}),
	}

	// N-dimensional real.
	rndFwd := func(name string, shapes [][]int, f func(x []float64, shape []int, fma bool) []complex128) accFn {
		return accFn{family: "r2cN", name: name, hasFMA: true, shapes: shapes, run: func(shape []int, seed uint64, fma bool) float64 {
			x := accRealF64(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("rfftn/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			return errC(f(x, shape, fma), want)
		}}
	}
	rndInv := func(name string, shapes [][]int, f func(s []complex128, shape []int, fma bool) []float64) accFn {
		return accFn{family: "r2cN", name: name, hasFMA: true, shapes: shapes, run: func(shape []int, seed uint64, fma bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF64(prod(shape), seed)), shape) })
			s := roundC[complex128](h)
			want := oracleMemo(fmt.Sprintf("irfftn/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			return errR(f(s, shape, fma), want)
		}}
	}
	rndfns := []accFn{
		rndFwd("RFFTN", accShapesND(), func(x []float64, s []int, fma bool) []complex128 {
			if fma {
				return RFFTNWith(x, s, Options{FMA: true})
			}
			return RFFTN(x, s)
		}),
		rndInv("IRFFTN", accShapesND(), func(c []complex128, s []int, fma bool) []float64 {
			if fma {
				return IRFFTNWith(c, s, Options{FMA: true})
			}
			return IRFFTN(c, s)
		}),
		rndFwd("RFFT2", accShapes2D(), func(x []float64, s []int, fma bool) []complex128 {
			if fma {
				return RFFT2With(x, [2]int{s[0], s[1]}, Options{FMA: true})
			}
			return RFFT2(x, [2]int{s[0], s[1]})
		}),
		rndInv("IRFFT2", accShapes2D(), func(c []complex128, s []int, fma bool) []float64 {
			if fma {
				return IRFFT2With(c, [2]int{s[0], s[1]}, Options{FMA: true})
			}
			return IRFFT2(c, [2]int{s[0], s[1]})
		}),
		rndFwd("RealPlan2.RFFT", accShapes2D(), func(x []float64, s []int, fma bool) []complex128 {
			p := NewRealPlan2(s[0], s[1])
			if fma {
				p = NewRealPlan2With(s[0], s[1], Options{FMA: true})
			}
			return p.RFFT(make([]complex128, p.SpectrumLen()), x)
		}),
		rndInv("RealPlan2.IRFFT", accShapes2D(), func(c []complex128, s []int, fma bool) []float64 {
			p := NewRealPlan2(s[0], s[1])
			if fma {
				p = NewRealPlan2With(s[0], s[1], Options{FMA: true})
			}
			return p.IRFFT(make([]float64, s[0]*s[1]), c)
		}),
	}

	out := append(cfns, rfns...)
	out = append(out, ndfns...)
	out = append(out, rndfns...)
	return append(out, accR2RFns64()...)
}

// accSpectrum rounds the oracle's half spectrum of a real signal of n points
// to T, with the imaginary parts of bin 0 and (n even) the Nyquist bin zero,
// so that it is exactly the spectrum of a real signal's bins as far as
// IRFFT/HFFT read it.
func accSpectrum[T complex64 | complex128](h []cdd, n int) []T {
	s := roundC[T](h)
	s[0] = T(complex(real(complex128(s[0])), 0))
	if n%2 == 0 {
		s[n/2] = T(complex(real(complex128(s[n/2])), 0))
	}
	return s
}

func accR2RFns64() []accFn {
	var out []accFn
	for _, cosine := range []bool{true, false} {
		for _, inverse := range []bool{false, true} {
			kind := "DST"
			if cosine {
				kind = "DCT"
			}
			if inverse {
				kind = "I" + kind
			}
			cosine, inverse := cosine, inverse
			mk := func(name string, shapes [][]int, f func(x []float64, shape []int, typ int, norm Norm) []float64) accFn {
				return accFn{family: "r2r", name: name, shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
					worst := 0.0
					x := accRealF64(prod(shape), seed)
					for typ := 1; typ <= 4; typ++ {
						if cosine && typ == 1 && shape[len(shape)-1] < 2 {
							continue
						}
						for _, norm := range []Norm{NormBackward, NormOrtho} {
							want := oracleMemo(fmt.Sprintf("r2r/%v/%v/%v/%d/%v/%d", cosine, inverse, shape, typ, norm, seed), func() []cdd {
								return oR2RN(cddFromR(x), shape, cosine, typ, norm, inverse)
							})
							worst = math.Max(worst, errR(f(x, shape, typ, norm), want))
						}
					}
					return worst
				}}
			}
			one := func(x []float64, shape []int, typ int, norm Norm) []float64 {
				switch {
				case cosine && !inverse:
					return DCT(x, typ, norm)
				case cosine:
					return IDCT(x, typ, norm)
				case !inverse:
					return DST(x, typ, norm)
				}
				return IDST(x, typ, norm)
			}
			plan := func(x []float64, shape []int, typ int, norm Norm) []float64 {
				dst := make([]float64, len(x))
				switch {
				case cosine && !inverse:
					return NewDCTPlan(len(x), typ).DCT(dst, x, norm)
				case cosine:
					return NewDCTPlan(len(x), typ).IDCT(dst, x, norm)
				case !inverse:
					return NewDSTPlan(len(x), typ).DST(dst, x, norm)
				}
				return NewDSTPlan(len(x), typ).IDST(dst, x, norm)
			}
			ndf := func(x []float64, shape []int, typ int, norm Norm) []float64 {
				switch {
				case cosine && !inverse:
					return DCTN(x, shape, typ, norm)
				case cosine:
					return IDCTN(x, shape, typ, norm)
				case !inverse:
					return DSTN(x, shape, typ, norm)
				}
				return IDSTN(x, shape, typ, norm)
			}
			out = append(out, mk(kind, accLines(accR2RLengths()), one), mk("Plan."+kind, accLines(accR2RLengths()), plan),
				mk(kind+"N", [][]int{{8, 9}, {16, 17}, {5, 6, 7}, {33, 8}}, ndf))
		}
	}
	return out
}

// oR2RN is the N-dimensional r2r oracle: the 1-D one along every axis.
func oR2RN(x []cdd, shape []int, cosine bool, typ int, norm Norm, inverse bool) []cdd {
	out := x
	for ax := range shape {
		out = applyAxis(out, shape, ax, func(l []cdd) []cdd { return oR2R(l, cosine, typ, norm, inverse) })
	}
	return out
}

// --- float32 ---

func accFns32() []accFn {
	cplx := func(name string, inverse bool, f func(x []complex64) []complex64) accFn {
		return accFn{family: "c2c/32", name: name, shapes: accLines(accLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accComplexF32(n, seed)
			want := oracleMemo(fmt.Sprintf("c2c32/%v/%d/%d", inverse, n, seed), func() []cdd {
				return oFFTN(cddFromC(x), shape, []int{0}, inverse)
			})
			return errC(f(x), want)
		}}
	}
	spectrum := func(n int, seed uint64) []complex64 {
		return accSpectrum[complex64](oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF32(n, seed))) }), n)
	}
	out := []accFn{
		cplx("FFT32", false, FFT32),
		cplx("IFFT32", true, IFFT32),
		cplx("Plan32.FFT", false, func(x []complex64) []complex64 { return NewPlan32(len(x)).FFT(make([]complex64, len(x)), x) }),
		cplx("Plan32.IFFT", true, func(x []complex64) []complex64 { return NewPlan32(len(x)).IFFT(make([]complex64, len(x)), x) }),
		cplx("FFT32With", false, func(x []complex64) []complex64 { return FFT32With(x, Options{FMA: true}) }),
		cplx("IFFT32With", true, func(x []complex64) []complex64 { return IFFT32With(x, Options{FMA: true}) }),
	}
	realFwd := func(name string, f func(x []float32) []complex64) accFn {
		return accFn{family: "r2c/32", name: name, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF32(n, seed)
			want := oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF32(n, seed))) })
			return errC(f(x), want)
		}}
	}
	realInv := func(name string, f func(s []complex64, n int) []float32) accFn {
		return accFn{family: "r2c/32", name: name, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := spectrum(n, seed)
			want := oracleMemo(fmt.Sprintf("irfft32/%d/%d", n, seed), func() []cdd { return oIRFFT(cddFromC(s), n) })
			return errR(f(s, n), want)
		}}
	}
	out = append(out,
		realFwd("RFFT32", RFFT32),
		realFwd("RealPlan32.RFFT", func(x []float32) []complex64 { return NewRealPlan32(len(x)).RFFT(make([]complex64, len(x)/2+1), x) }),
		realInv("IRFFT32", IRFFT32),
		realInv("RealPlan32.IRFFT", func(s []complex64, n int) []float32 { return NewRealPlan32(n).IRFFT(make([]float32, n), s) }),
		accFn{family: "r2c/32", name: "HFFT32", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := spectrum(n, seed)
			want := oracleMemo(fmt.Sprintf("hfft32/%d/%d", n, seed), func() []cdd { return oHFFT(cddFromC(s), n) })
			return errR(HFFT32(s, n), want)
		}},
		accFn{family: "r2c/32", name: "IHFFT32", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF32(n, seed)
			want := oracleMemo(fmt.Sprintf("ihfft32/%d/%d", n, seed), func() []cdd { return oIHFFT(cddFromR(x)) })
			return errC(IHFFT32(x), want)
		}},
	)

	nd := func(name string, shapes [][]int, inverse bool, f func(x []complex64, shape []int) []complex64) accFn {
		return accFn{family: "c2cN/32", name: name, shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
			x := accComplexF32(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("fftn32/%v/%v/%d", inverse, shape, seed), func() []cdd {
				return oFFTN(cddFromC(x), shape, allAxes(len(shape)), inverse)
			})
			return errC(f(x, shape), want)
		}}
	}
	two := func(s []int) [2]int { return [2]int{s[0], s[1]} }
	out = append(out,
		nd("FFTN32", accShapesND(), false, FFTN32),
		nd("IFFTN32", accShapesND(), true, IFFTN32),
		nd("FFT2_32", accShapes2D(), false, func(x []complex64, s []int) []complex64 { return FFT2_32(x, two(s)) }),
		nd("IFFT2_32", accShapes2D(), true, func(x []complex64, s []int) []complex64 { return IFFT2_32(x, two(s)) }),
		nd("PlanN32.FFT", accShapesND(), false, func(x []complex64, s []int) []complex64 {
			return NewPlanN32(s...).FFT(make([]complex64, len(x)), x)
		}),
		nd("PlanN32.IFFT", accShapesND(), true, func(x []complex64, s []int) []complex64 {
			return NewPlanN32(s...).IFFT(make([]complex64, len(x)), x)
		}),
		nd("FFTN32With", accShapesND(), false, func(x []complex64, s []int) []complex64 { return FFTN32With(x, s, Options{FMA: true}) }),
	)

	rndFwd := func(name string, shapes [][]int, f func(x []float32, shape []int) []complex64) accFn {
		return accFn{family: "r2cN/32", name: name, shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
			x := accRealF32(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			return errC(f(x, shape), want)
		}}
	}
	rndInv := func(name string, shapes [][]int, f func(s []complex64, shape []int) []float32) accFn {
		return accFn{family: "r2cN/32", name: name, shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF32(prod(shape), seed)), shape) })
			s := roundC[complex64](h)
			want := oracleMemo(fmt.Sprintf("irfftn32/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			return errR(f(s, shape), want)
		}}
	}
	out = append(out,
		rndFwd("RFFTN32", accShapesND(), RFFTN32),
		rndInv("IRFFTN32", accShapesND(), IRFFTN32),
		rndFwd("RFFT2_32", accShapes2D(), func(x []float32, s []int) []complex64 { return RFFT2_32(x, two(s)) }),
		rndInv("IRFFT2_32", accShapes2D(), func(c []complex64, s []int) []float32 { return IRFFT2_32(c, two(s)) }),
		rndFwd("RealPlan2_32.RFFT", accShapes2D(), func(x []float32, s []int) []complex64 {
			p := NewRealPlan2_32(s[0], s[1])
			return p.RFFT(make([]complex64, p.SpectrumLen()), x)
		}),
		rndInv("RealPlan2_32.IRFFT", accShapes2D(), func(c []complex64, s []int) []float32 {
			return NewRealPlan2_32(s[0], s[1]).IRFFT(make([]float32, s[0]*s[1]), c)
		}),
	)
	return append(out, accR2RFns32()...)
}

func accR2RFns32() []accFn {
	var out []accFn
	for _, cosine := range []bool{true, false} {
		for _, inverse := range []bool{false, true} {
			kind := "DST"
			if cosine {
				kind = "DCT"
			}
			if inverse {
				kind = "I" + kind
			}
			cosine, inverse := cosine, inverse
			mk := func(name string, shapes [][]int, f func(x []float32, shape []int, typ int, norm Norm) []float32) accFn {
				return accFn{family: "r2r/32", name: name + "32", shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
					worst := 0.0
					x := accRealF32(prod(shape), seed)
					for typ := 1; typ <= 4; typ++ {
						if cosine && typ == 1 && shape[len(shape)-1] < 2 {
							continue
						}
						for _, norm := range []Norm{NormBackward, NormOrtho} {
							want := oracleMemo(fmt.Sprintf("r2r32/%v/%v/%v/%d/%v/%d", cosine, inverse, shape, typ, norm, seed), func() []cdd {
								return oR2RN(cddFromR(x), shape, cosine, typ, norm, inverse)
							})
							worst = math.Max(worst, errR(f(x, shape, typ, norm), want))
						}
					}
					return worst
				}}
			}
			one := func(x []float32, shape []int, typ int, norm Norm) []float32 {
				switch {
				case cosine && !inverse:
					return DCT32(x, typ, norm)
				case cosine:
					return IDCT32(x, typ, norm)
				case !inverse:
					return DST32(x, typ, norm)
				}
				return IDST32(x, typ, norm)
			}
			plan := func(x []float32, shape []int, typ int, norm Norm) []float32 {
				dst := make([]float32, len(x))
				switch {
				case cosine && !inverse:
					return NewDCTPlan32(len(x), typ).DCT(dst, x, norm)
				case cosine:
					return NewDCTPlan32(len(x), typ).IDCT(dst, x, norm)
				case !inverse:
					return NewDSTPlan32(len(x), typ).DST(dst, x, norm)
				}
				return NewDSTPlan32(len(x), typ).IDST(dst, x, norm)
			}
			ndf := func(x []float32, shape []int, typ int, norm Norm) []float32 {
				switch {
				case cosine && !inverse:
					return DCTN32(x, shape, typ, norm)
				case cosine:
					return IDCTN32(x, shape, typ, norm)
				case !inverse:
					return DSTN32(x, shape, typ, norm)
				}
				return IDSTN32(x, shape, typ, norm)
			}
			out = append(out, mk(kind, accLines(accR2RLengths()), one), mk("Plan."+kind, accLines(accR2RLengths()), plan),
				mk(kind+"N", [][]int{{8, 9}, {16, 17}, {5, 6, 7}, {33, 8}}, ndf))
		}
	}
	return out
}

// --- the test ---

type accStat struct {
	worst float64
	where string
}

func accEps(family string) float64 {
	if len(family) > 3 && family[len(family)-3:] == "/32" {
		return eps32
	}
	return eps64
}

// accClass names the route a 1-D length takes, for the table.
func accClass(shape []int) string {
	if len(shape) != 1 {
		return fmt.Sprintf("%dD", len(shape))
	}
	n := shape[0]
	if n <= 1 {
		return "trivial"
	}
	p := NewPlan(n)
	switch {
	case p.it != nil:
		return "pow2"
	case p.sk != nil:
		return "smooth"
	case p.rader != nil:
		return "rader"
	}
	return "bluestein"
}

// TestAccuracyBound measures every public transform against the oracle, with
// the default arithmetic and with Options.FMA, and asserts the stated bound.
func TestAccuracyBound(t *testing.T) {
	fns := append(accFns64(), accFns32()...)
	for _, mode := range []string{"default", "fma"} {
		t.Run(mode, func(t *testing.T) {
			fma := mode == "fma"
			if fma {
				forceFMA(t)
			}
			perFn := map[string]accStat{}
			perFamily := map[string]accStat{}
			perClass := map[string]accStat{}
			for _, f := range fns {
				if fma && !f.hasFMA {
					continue // nothing to switch: the same code as the default run
				}
				eps := accEps(f.family)
				for _, shape := range f.shapes {
					n := f.points(shape)
					for _, seed := range []uint64{1, 2} {
						if n > 1100 && seed == 2 {
							continue
						}
						err := f.run(shape, seed, fma)
						r := accRatio(err, n, eps)
						where := fmt.Sprintf("%v seed %d", shape, seed)
						for _, m := range []struct {
							mp  map[string]accStat
							key string
						}{{perFn, f.family + " " + f.name}, {perFamily, f.family}, {perClass, f.family + " " + accClass(shape)}} {
							if s := m.mp[m.key]; r > s.worst {
								m.mp[m.key] = accStat{r, f.name + " " + where}
							}
						}
						if c, ok := accBound[f.family]; ok && r > c {
							t.Errorf("%s %v: relative RMS error %.3g = %.3f·ε·log2(n), above c = %v", f.name, where, err, r, c)
						}
					}
				}
			}
			if *accTable {
				accPrint(t, mode, perFn, perFamily, perClass)
			}
			for fam := range perFamily {
				if _, ok := accBound[fam]; !ok && !*accTable {
					t.Errorf("family %q has no bound in accBound", fam)
				}
			}
		})
	}
}

func accPrint(t *testing.T, mode string, perFn, perFamily, perClass map[string]accStat) {
	for _, m := range []struct {
		title string
		mp    map[string]accStat
	}{{"function", perFn}, {"route", perClass}, {"family", perFamily}} {
		keys := make([]string, 0, len(m.mp))
		for k := range m.mp {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			t.Logf("ACC %-7s %-8s %-34s worst err/(eps*log2n) = %6.3f  at %s", mode, m.title, k, m.mp[k].worst, m.mp[k].where)
		}
	}
}

// TestAccuracyMutation shows what the bound can see. One twiddle of the
// oracle's own root table is perturbed, by 1 ulp and by 1e-10, and the
// transform that results is held to the bound against the exact one: the 1e-10
// perturbation must exceed it at every length (it is a wrong twiddle, not
// rounding), the 1 ulp one does not (it is below what rounding does anyway,
// and the bound is not meant to see it). The same two perturbations were made
// by hand in the library's own twiddle tables (docs/accuracy.md): the 1e-10 one
// turned TestAccuracyBound red, the 1 ulp one did not.
func TestAccuracyMutation(t *testing.T) {
	for _, n := range []int{8, 16, 64, 360, 1000, 1009, 1024} {
		x := cddFromC(accComplexF64(n, 9))
		want := ddDFT(x, false)
		bound := accBound["c2c"] * eps64 * math.Log2(float64(n))
		for _, k0 := range []int{1, n / 8, n/2 - 1} {
			if k0 < 1 {
				continue
			}
			for _, m := range []struct {
				name   string
				delta  func(v float64) float64
				caught bool
			}{
				{"1 ulp", func(v float64) float64 { return math.Nextafter(v, math.Inf(1)) - v }, false},
				{"1e-10", func(float64) float64 { return 1e-10 }, true},
			} {
				tw := append(ddRootTable(nil), ddRoots(n)...)
				tw[k0].re.hi += m.delta(tw[k0].re.hi)
				err := errDD(ddDFTTable(x, false, tw), want)
				t.Logf("n=%-5d twiddle %-5d perturbed by %-5s: relative RMS error %.3g = %.3g·bound", n, k0, m.name, err, err/bound)
				if (err > bound) != m.caught {
					t.Errorf("n=%d twiddle %d %s: error %.3g against bound %.3g, caught = %v, want %v", n, k0, m.name, err, bound, err > bound, m.caught)
				}
			}
		}
	}
}
