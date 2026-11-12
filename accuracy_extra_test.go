package fft

import (
	"fmt"
	"math"
	"testing"
)

// The rest of the public entry points of the accuracy contract: the *Norm plan
// methods, Options.N/Norm/Axes, the float32 ...With functions, PSD and
// Spectrogram (accuracy_test.go has the rest and the bounds).

var accNorms = []Norm{NormBackward, NormOrtho, NormForward}

// oScaled returns a copy of the oracle result a scaled by f.
func oScaled(a []cdd, f dd) []cdd {
	out := append([]cdd(nil), a...)
	scaleAll(out, f)
	return out
}

// invFactor turns a result already divided by size (the oracles' inverses)
// into the one scaled as norm's inverse is.
func invFactor(norm Norm, size int) dd {
	return ddMul(normScale(norm, size, true), dd{float64(size), 0})
}

func maxOver(norms []Norm, f func(Norm) float64) float64 {
	w := 0.0
	for _, m := range norms {
		w = math.Max(w, f(m))
	}
	return w
}

func accExtraFns() []accFn {
	var out []accFn
	two := func(s []int) [2]int { return [2]int{s[0], s[1]} }

	// --- *Norm plan methods, float64 ---
	cn := func(name string, inverse bool, f func(x []complex128, norm Norm) []complex128) accFn {
		return accFn{family: "c2c", name: name, shapes: accLines(accLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accComplexF64(n, seed)
			base := oracleMemo(fmt.Sprintf("c2c/%v/%d/%d", inverse, n, seed), func() []cdd { return oFFTN(cddFromC(x), shape, []int{0}, inverse) })
			return maxOver(accNorms, func(m Norm) float64 {
				f64 := normScale(m, n, false)
				if inverse {
					f64 = invFactor(m, n)
				}
				return errC(f(x, m), oScaled(base, f64))
			})
		}}
	}
	out = append(out,
		cn("Plan.FFTNorm", false, func(x []complex128, m Norm) []complex128 {
			return NewPlan(len(x)).FFTNorm(make([]complex128, len(x)), x, m)
		}),
		cn("Plan.IFFTNorm", true, func(x []complex128, m Norm) []complex128 {
			return NewPlan(len(x)).IFFTNorm(make([]complex128, len(x)), x, m)
		}),
	)
	rn := func(name string, f func(x []float64, norm Norm) []complex128) accFn {
		return accFn{family: "r2c", name: name, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF64(n, seed)
			base := oracleMemo(fmt.Sprintf("rfft/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(x)) })
			return maxOver(accNorms, func(m Norm) float64 { return errC(f(x, m), oScaled(base, normScale(m, n, false))) })
		}}
	}
	ri := func(name string, f func(s []complex128, n int, norm Norm) []float64) accFn {
		return accFn{family: "r2c", name: name, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := accSpectrum[complex128](oracleMemo(fmt.Sprintf("rfft/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF64(n, seed))) }), n)
			base := oracleMemo(fmt.Sprintf("irfft/%d/%d", n, seed), func() []cdd { return oIRFFT(cddFromC(s), n) })
			return maxOver(accNorms, func(m Norm) float64 { return errR(f(s, n, m), oScaled(base, invFactor(m, n))) })
		}}
	}
	out = append(out,
		rn("RealPlan.RFFTNorm", func(x []float64, m Norm) []complex128 {
			return NewRealPlan(len(x)).RFFTNorm(make([]complex128, len(x)/2+1), x, m)
		}),
		ri("RealPlan.IRFFTNorm", func(s []complex128, n int, m Norm) []float64 {
			return NewRealPlan(n).IRFFTNorm(make([]float64, n), s, m)
		}),
	)
	nd := func(name string, inverse bool, f func(x []complex128, shape []int, norm Norm) []complex128) accFn {
		return accFn{family: "c2cN", name: name, shapes: accShapesND(), run: func(shape []int, seed uint64, _ bool) float64 {
			size := prod(shape)
			x := accComplexF64(size, seed)
			base := oracleMemo(fmt.Sprintf("fftn/%v/%v/%d", inverse, shape, seed), func() []cdd { return oFFTN(cddFromC(x), shape, allAxes(len(shape)), inverse) })
			return maxOver(accNorms, func(m Norm) float64 {
				f64 := normScale(m, size, false)
				if inverse {
					f64 = invFactor(m, size)
				}
				return errC(f(x, shape, m), oScaled(base, f64))
			})
		}}
	}
	out = append(out,
		nd("PlanN.FFTNorm", false, func(x []complex128, s []int, m Norm) []complex128 {
			return NewPlanN(s...).FFTNorm(make([]complex128, len(x)), x, m)
		}),
		nd("PlanN.IFFTNorm", true, func(x []complex128, s []int, m Norm) []complex128 {
			return NewPlanN(s...).IFFTNorm(make([]complex128, len(x)), x, m)
		}),
	)
	rnd := func(name string, f func(x []float64, shape []int, norm Norm) []complex128) accFn {
		return accFn{family: "r2cN", name: name, shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			x := accRealF64(prod(shape), seed)
			base := oracleMemo(fmt.Sprintf("rfftn/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			return maxOver(accNorms, func(m Norm) float64 { return errC(f(x, shape, m), oScaled(base, normScale(m, prod(shape), false))) })
		}}
	}
	rndi := func(name string, f func(s []complex128, shape []int, norm Norm) []float64) accFn {
		return accFn{family: "r2cN", name: name, shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF64(prod(shape), seed)), shape) })
			s := roundC[complex128](h)
			base := oracleMemo(fmt.Sprintf("irfftn/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			return maxOver(accNorms, func(m Norm) float64 { return errR(f(s, shape, m), oScaled(base, invFactor(m, prod(shape)))) })
		}}
	}
	out = append(out,
		rnd("RealPlan2.RFFTNorm", func(x []float64, s []int, m Norm) []complex128 {
			p := NewRealPlan2(s[0], s[1])
			return p.RFFTNorm(make([]complex128, p.SpectrumLen()), x, m)
		}),
		rndi("RealPlan2.IRFFTNorm", func(c []complex128, s []int, m Norm) []float64 {
			return NewRealPlan2(s[0], s[1]).IRFFTNorm(make([]float64, s[0]*s[1]), c, m)
		}),
	)

	// --- Options N, Norm, Axes, float64 ---
	for _, nn := range []struct {
		name string
		n    func(n int) int
	}{{"pad", func(n int) int { return n + n/2 + 1 }}, {"truncate", func(n int) int { return n/2 + 1 }}} {
		nn := nn
		out = append(out,
			accFn{family: "c2c", name: "FFTWith N " + nn.name + " Norm", hasFMA: true, shapes: accLines(accLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
				n := shape[0]
				N := nn.n(n)
				x := accComplexF64(n, seed)
				p := make([]complex128, N)
				copy(p, x)
				base := oracleMemo(fmt.Sprintf("fftwithN/%d/%d/%d", n, N, seed), func() []cdd { return ddDFT(cddFromC(p), false) })
				return maxOver(accNorms, func(m Norm) float64 {
					return errC(FFTWith(x, Options{N: N, Norm: m, FMA: fma}), oScaled(base, normScale(m, N, false)))
				})
			}},
			accFn{family: "c2c", name: "IFFTWith N " + nn.name + " Norm", hasFMA: true, shapes: accLines(accLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
				n := shape[0]
				N := nn.n(n)
				x := accComplexF64(n, seed)
				p := make([]complex128, N)
				copy(p, x)
				base := oracleMemo(fmt.Sprintf("ifftwithN/%d/%d/%d", n, N, seed), func() []cdd {
					return oScaled(ddDFT(cddFromC(p), true), dd{1, 0})
				})
				return maxOver(accNorms, func(m Norm) float64 {
					return errC(IFFTWith(x, Options{N: N, Norm: m, FMA: fma}), oScaled(base, normScale(m, N, true)))
				})
			}},
			accFn{family: "r2c", name: "RFFTWith N " + nn.name + " Norm", hasFMA: true, shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, fma bool) float64 {
				n := shape[0]
				N := nn.n(n)
				x := accRealF64(n, seed)
				p := make([]float64, N)
				copy(p, x)
				base := oracleMemo(fmt.Sprintf("rfftwithN/%d/%d/%d", n, N, seed), func() []cdd { return oRFFT(cddFromR(p)) })
				return maxOver(accNorms, func(m Norm) float64 {
					return errC(RFFTWith(x, Options{N: N, Norm: m, FMA: fma}), oScaled(base, normScale(m, N, false)))
				})
			}},
		)
	}
	for _, axes := range [][]int{{0}, {-1}, {1, 0}} {
		axes := axes
		out = append(out, accFn{family: "c2cN", name: fmt.Sprintf("FFTNWith Axes %v Norm", axes), hasFMA: true, shapes: accShapes2D(), run: func(shape []int, seed uint64, fma bool) float64 {
			size := prod(shape)
			x := accComplexF64(size, seed)
			ax := make([]int, len(axes))
			count := 1
			for i, a := range axes {
				if a < 0 {
					a += len(shape)
				}
				ax[i] = a
				count *= shape[a]
			}
			base := oracleMemo(fmt.Sprintf("fftnaxes/%v/%v/%d", shape, ax, seed), func() []cdd { return oFFTN(cddFromC(x), shape, ax, false) })
			return maxOver(accNorms, func(m Norm) float64 {
				return errC(FFTNWith(x, shape, Options{Axes: axes, Norm: m, FMA: fma}), oScaled(base, normScale(m, count, false)))
			})
		}})
	}

	// --- PSD and Spectrogram ---
	out = append(out,
		accFn{family: "psd", name: "PSD", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF64(n, seed)
			const d = 0.125
			return errR(PSD(x, d), oPSD(cddFromR(x), d))
		}},
		accFn{family: "psd", name: "Spectrogram", shapes: accLines([]int{16, 64, 100, 257, 300}), run: func(shape []int, seed uint64, _ bool) float64 {
			seg := shape[0]
			x := accRealF64(5*seg+seg/3, seed)
			w := Hann(seg)
			const d = 0.5
			frames := Spectrogram(x, seg, seg/2, w, d)
			var got []float64
			var want []cdd
			step := seg - seg/2
			for f := range frames {
				s := make([]float64, seg)
				for i := range s {
					s[i] = float64(x[f*step+i] * w[i])
				}
				got = append(got, frames[f]...)
				want = append(want, oPSD(cddFromR(s), d)...)
			}
			return errR(got, want)
		}},
	)

	// --- float32: *Norm plan methods and the ...With functions ---
	cn32 := func(name string, inverse bool, f func(x []complex64, norm Norm) []complex64) accFn {
		return accFn{family: "c2c/32", name: name, shapes: accLines(accLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accComplexF32(n, seed)
			base := oracleMemo(fmt.Sprintf("c2c32/%v/%d/%d", inverse, n, seed), func() []cdd { return oFFTN(cddFromC(x), shape, []int{0}, inverse) })
			return maxOver(accNorms, func(m Norm) float64 {
				f64 := normScale(m, n, false)
				if inverse {
					f64 = invFactor(m, n)
				}
				return errC(f(x, m), oScaled(base, f64))
			})
		}}
	}
	out = append(out,
		cn32("Plan32.FFTNorm", false, func(x []complex64, m Norm) []complex64 {
			return NewPlan32(len(x)).FFTNorm(make([]complex64, len(x)), x, m)
		}),
		cn32("Plan32.IFFTNorm", true, func(x []complex64, m Norm) []complex64 {
			return NewPlan32(len(x)).IFFTNorm(make([]complex64, len(x)), x, m)
		}),
		accFn{family: "r2c/32", name: "RealPlan32.RFFTNorm", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF32(n, seed)
			base := oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(x)) })
			return maxOver(accNorms, func(m Norm) float64 {
				return errC(NewRealPlan32(n).RFFTNorm(make([]complex64, n/2+1), x, m), oScaled(base, normScale(m, n, false)))
			})
		}},
		accFn{family: "r2c/32", name: "RealPlan32.IRFFTNorm", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := accSpectrum[complex64](oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF32(n, seed))) }), n)
			base := oracleMemo(fmt.Sprintf("irfft32/%d/%d", n, seed), func() []cdd { return oIRFFT(cddFromC(s), n) })
			return maxOver(accNorms, func(m Norm) float64 {
				return errR(NewRealPlan32(n).IRFFTNorm(make([]float32, n), s, m), oScaled(base, invFactor(m, n)))
			})
		}},
	)
	nd32 := func(name string, inverse bool, f func(x []complex64, shape []int, norm Norm) []complex64) accFn {
		return accFn{family: "c2cN/32", name: name, shapes: accShapesND(), run: func(shape []int, seed uint64, _ bool) float64 {
			size := prod(shape)
			x := accComplexF32(size, seed)
			base := oracleMemo(fmt.Sprintf("fftn32/%v/%v/%d", inverse, shape, seed), func() []cdd { return oFFTN(cddFromC(x), shape, allAxes(len(shape)), inverse) })
			return maxOver(accNorms, func(m Norm) float64 {
				f64 := normScale(m, size, false)
				if inverse {
					f64 = invFactor(m, size)
				}
				return errC(f(x, shape, m), oScaled(base, f64))
			})
		}}
	}
	out = append(out,
		nd32("PlanN32.FFTNorm", false, func(x []complex64, s []int, m Norm) []complex64 {
			return NewPlanN32(s...).FFTNorm(make([]complex64, len(x)), x, m)
		}),
		nd32("PlanN32.IFFTNorm", true, func(x []complex64, s []int, m Norm) []complex64 {
			return NewPlanN32(s...).IFFTNorm(make([]complex64, len(x)), x, m)
		}),
		accFn{family: "r2cN/32", name: "RealPlan2_32.RFFTNorm", shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			x := accRealF32(prod(shape), seed)
			base := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			p := NewRealPlan2_32(shape[0], shape[1])
			return maxOver(accNorms, func(m Norm) float64 {
				return errC(p.RFFTNorm(make([]complex64, p.SpectrumLen()), x, m), oScaled(base, normScale(m, prod(shape), false)))
			})
		}},
		accFn{family: "r2cN/32", name: "RealPlan2_32.IRFFTNorm", shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF32(prod(shape), seed)), shape) })
			s := roundC[complex64](h)
			base := oracleMemo(fmt.Sprintf("irfftn32/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			p := NewRealPlan2_32(shape[0], shape[1])
			return maxOver(accNorms, func(m Norm) float64 {
				return errR(p.IRFFTNorm(make([]float32, prod(shape)), s, m), oScaled(base, invFactor(m, prod(shape))))
			})
		}},
	)

	// The float32 ...With functions, with the FMA request they ignore.
	o := Options{FMA: true}
	w32 := func(name string, inverse bool, f func(x []complex64, shape []int) []complex64, shapes [][]int) accFn {
		return accFn{family: "c2cN/32", name: name, shapes: shapes, run: func(shape []int, seed uint64, _ bool) float64 {
			x := accComplexF32(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("fftn32/%v/%v/%d", inverse, shape, seed), func() []cdd { return oFFTN(cddFromC(x), shape, allAxes(len(shape)), inverse) })
			return errC(f(x, shape), want)
		}}
	}
	out = append(out,
		w32("IFFTN32With", true, func(x []complex64, s []int) []complex64 { return IFFTN32With(x, s, o) }, accShapesND()),
		w32("FFT2_32With", false, func(x []complex64, s []int) []complex64 { return FFT2_32With(x, two(s), o) }, accShapes2D()),
		w32("IFFT2_32With", true, func(x []complex64, s []int) []complex64 { return IFFT2_32With(x, two(s), o) }, accShapes2D()),
		accFn{family: "r2c/32", name: "RFFT32With", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF32(n, seed)
			want := oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(x)) })
			return errC(RFFT32With(x, o), want)
		}},
		accFn{family: "r2c/32", name: "IRFFT32With", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := accSpectrum[complex64](oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF32(n, seed))) }), n)
			want := oracleMemo(fmt.Sprintf("irfft32/%d/%d", n, seed), func() []cdd { return oIRFFT(cddFromC(s), n) })
			return errR(IRFFT32With(s, Options{N: n, FMA: true}), want)
		}},
		accFn{family: "r2c/32", name: "HFFT32With", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			s := accSpectrum[complex64](oracleMemo(fmt.Sprintf("rfft32/%d/%d", n, seed), func() []cdd { return oRFFT(cddFromR(accRealF32(n, seed))) }), n)
			want := oracleMemo(fmt.Sprintf("hfft32/%d/%d", n, seed), func() []cdd { return oHFFT(cddFromC(s), n) })
			return errR(HFFT32With(s, Options{N: n, FMA: true}), want)
		}},
		accFn{family: "r2c/32", name: "IHFFT32With", shapes: accLines(accRealLengths()), run: func(shape []int, seed uint64, _ bool) float64 {
			n := shape[0]
			x := accRealF32(n, seed)
			want := oracleMemo(fmt.Sprintf("ihfft32/%d/%d", n, seed), func() []cdd { return oIHFFT(cddFromR(x)) })
			return errC(IHFFT32With(x, o), want)
		}},
		accFn{family: "r2cN/32", name: "RFFTN32With", shapes: accShapesND(), run: func(shape []int, seed uint64, _ bool) float64 {
			x := accRealF32(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			return errC(RFFTN32With(x, shape, o), want)
		}},
		accFn{family: "r2cN/32", name: "IRFFTN32With", shapes: accShapesND(), run: func(shape []int, seed uint64, _ bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF32(prod(shape), seed)), shape) })
			s := roundC[complex64](h)
			want := oracleMemo(fmt.Sprintf("irfftn32/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			return errR(IRFFTN32With(s, shape, o), want)
		}},
		accFn{family: "r2cN/32", name: "RFFT2_32With", shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			x := accRealF32(prod(shape), seed)
			want := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(x), shape) })
			return errC(RFFT2_32With(x, two(shape), o), want)
		}},
		accFn{family: "r2cN/32", name: "IRFFT2_32With", shapes: accShapes2D(), run: func(shape []int, seed uint64, _ bool) float64 {
			h := oracleMemo(fmt.Sprintf("rfftn32/%v/%d", shape, seed), func() []cdd { return oRFFTN(cddFromR(accRealF32(prod(shape), seed)), shape) })
			s := roundC[complex64](h)
			want := oracleMemo(fmt.Sprintf("irfftn32/%v/%d", shape, seed), func() []cdd { return oIRFFTN(cddFromC(s), shape) })
			return errR(IRFFT2_32With(s, two(shape), o), want)
		}},
	)
	return out
}

// oPSD is PSD's definition in double-double for a real signal x at spacing d:
// |X[k]|²·d/n, doubled but for DC and (n even) Nyquist; the values are in .re.
func oPSD(x []cdd, d float64) []cdd {
	n := len(x)
	spec := oRFFT(x)
	scale := ddDivF(dd{d, 0}, float64(n))
	out := make([]cdd, len(spec))
	for k, v := range spec {
		p := ddMul(ddAdd(ddMul(v.re, v.re), ddMul(v.im, v.im)), scale)
		if k != 0 && !(n%2 == 0 && k == len(spec)-1) {
			p = ddMul(p, dd{2, 0})
		}
		out[k].re = p
	}
	return out
}

// TestAccuracyCoversEveryRoute: whatever the architecture and however short
// the run (-short, -race, qemu), the sweep reaches Rader, Bluestein, smooth
// and power-of-two lengths, an odd real length, an N-D shape and the r2r
// transforms; a cap that dropped one would leave a lane asserting nothing
// about it.
func TestAccuracyCoversEveryRoute(t *testing.T) {
	seen := map[string]bool{}
	for _, n := range accLengths() {
		seen[accClass([]int{n})] = true
	}
	for _, c := range []string{"smooth", "rader", "bluestein"} {
		if !seen[c] {
			t.Errorf("no %s length in the sweep", c)
		}
	}
	odd := false
	for _, n := range accRealLengths() {
		odd = odd || n%2 == 1 && n > 1
	}
	if !odd {
		t.Error("no odd real length")
	}
	if len(accShapesND()) == 0 || len(accShapes2D()) == 0 || len(accR2RLengths()) == 0 {
		t.Error("no N-D shape or r2r length")
	}
	fam := map[string]bool{}
	for _, f := range append(append(accFns64(), accFns32()...), accExtraFns()...) {
		fam[f.family] = true
		if len(f.shapes) == 0 {
			t.Errorf("%s has no shape", f.name)
		}
	}
	for k := range accBound {
		if !fam[k] {
			t.Errorf("family %s has no function", k)
		}
	}
}
