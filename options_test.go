package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// The oracles in this file come from the DEFINITIONS in numpy.fft's
// documentation, written independently of the package: O(n²) sums, numpy's
// scale factors spelled out per mode, and the documented identities (hfft is
// n·irfft(conj(x)), ifftshift inverts fftshift, ...). None of them calls the
// code under test.

var allNorms = []Norm{NormBackward, NormOrtho, NormForward}

// normFactor is numpy's scale factor for a transform of n points, from its
// documentation: backward = 1 forward, 1/n inverse; ortho = 1/sqrt(n) both
// ways; forward = 1/n forward, 1 inverse.
func normFactor(m Norm, n int, inverse bool) float64 {
	if n == 0 {
		return 1
	}
	switch m {
	case NormOrtho:
		return 1 / math.Sqrt(float64(n))
	case NormForward:
		if inverse {
			return 1
		}
		return 1 / float64(n)
	}
	if inverse {
		return 1 / float64(n)
	}
	return 1
}

// padC truncates or zero-pads x to n points (numpy's n argument).
func padC(x []complex128, n int) []complex128 {
	out := make([]complex128, n)
	copy(out, x)
	return out
}

func padR(x []float64, n int) []float64 {
	out := make([]float64, n)
	copy(out, x)
	return out
}

func scaledC(x []complex128, f float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = v * complex(f, 0)
	}
	return out
}

func toC(x []float64) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = complex(v, 0)
	}
	return out
}

// near reports |a-b| within a tolerance relative to the magnitude involved.
func near(a, b complex128, scale float64) bool {
	return cmplx.Abs(a-b) <= 1e-9*math.Max(1, scale)
}

func checkC(t *testing.T, ctx string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", ctx, len(got), len(want))
	}
	for i := range got {
		if !near(got[i], want[i], float64(len(want))) {
			t.Fatalf("%s: index %d: got %v, want %v", ctx, i, got[i], want[i])
		}
	}
}

func checkR(t *testing.T, ctx string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", ctx, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-9*math.Max(1, float64(len(want))) {
			t.Fatalf("%s: index %d: got %v, want %v", ctx, i, got[i], want[i])
		}
	}
}

// c2rOracle is the real inverse DFT of a half spectrum h of a length-n real
// signal, unnormalized, written from its definition: the Hermitian extension
// X[n-k] = conj(X[k]) summed, which leaves only the real parts of the DC bin
// and (n even) of the Nyquist bin — numpy's irfft discards their imaginary
// parts. Bins past len(h) are zero.
func c2rOracle(h []complex128, n int) []float64 {
	bin := func(k int) complex128 {
		if k < len(h) {
			return h[k]
		}
		return 0
	}
	out := make([]float64, n)
	for j := range out {
		s := real(bin(0))
		for k := 1; 2*k < n; k++ {
			s += 2 * real(bin(k)*cmplx.Rect(1, 2*math.Pi*float64(k*j)/float64(n)))
		}
		if n%2 == 0 && n > 0 {
			s += real(bin(n/2)) * math.Cos(math.Pi*float64(j))
		}
		out[j] = s
	}
	return out
}

// testSpectrum is a half spectrum with non-zero imaginary parts everywhere,
// DC and Nyquist included, so the tests see that those are discarded.
func testSpectrum(m int) []complex128 {
	x := make([]complex128, m)
	for i := range x {
		x[i] = complex(math.Cos(float64(i)*1.3)+float64(i%3), math.Sin(float64(i)*0.9)-0.5)
	}
	return x
}

func lengthsAround(l int) []int {
	ns := []int{0, 1, 2, l + 3, 2*l + 1}
	if l > 1 {
		ns = append(ns, l-1)
	}
	return ns
}

func TestFFTWithAgainstDefinition(t *testing.T) {
	for _, l := range []int{0, 1, 2, 3, 5, 8, 12, 17} {
		x := cmplxSignal(l)
		for _, n := range lengthsAround(l) {
			for _, m := range allNorms {
				eff := n
				if n == 0 {
					eff = l
				}
				ctx := fmt.Sprintf("len %d N %d %v", l, n, m)
				o := Options{N: n, Norm: m}
				checkC(t, "FFTWith "+ctx, FFTWith(x, o), scaledC(naiveDFT(padC(x, eff)), normFactor(m, eff, false)))
				checkC(t, "IFFTWith "+ctx, IFFTWith(x, o), scaledC(naiveIDFTUnnormalized(padC(x, eff)), normFactor(m, eff, true)))
			}
		}
		checkC(t, "FFTWith default", FFTWith(x, Options{}), FFT(x))
		checkC(t, "IFFTWith default", IFFTWith(x, Options{}), IFFT(x))
	}
	// The input is not modified, even when padded or truncated.
	x := cmplxSignal(6)
	keep := slices.Clone(x)
	FFTWith(x, Options{N: 4})
	IFFTWith(x, Options{N: 9, Norm: NormOrtho})
	if !slices.Equal(x, keep) {
		t.Fatal("FFTWith/IFFTWith modified the input")
	}
}

func TestRFFTWithAgainstDefinition(t *testing.T) {
	for _, l := range []int{0, 1, 2, 3, 6, 7, 16, 21} {
		x := realSignal(l)
		for _, n := range lengthsAround(l) {
			eff := n
			if n == 0 {
				eff = l
			}
			for _, m := range allNorms {
				ctx := fmt.Sprintf("len %d N %d %v", l, n, m)
				want := scaledC(naiveDFT(toC(padR(x, eff))), normFactor(m, eff, false))
				if eff > 0 {
					want = want[:eff/2+1]
				}
				checkC(t, "RFFTWith "+ctx, RFFTWith(x, Options{N: n, Norm: m}), want)
			}
		}
		checkC(t, "RFFTWith default", RFFTWith(x, Options{}), RFFT(x))
	}
}

func TestIRFFTWithAgainstDefinition(t *testing.T) {
	for _, l := range []int{0, 1, 2, 3, 4, 5, 9, 12} {
		h := testSpectrum(l)
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 2 * l, 2*l - 1, 2*l + 5} {
			if n < 0 {
				continue
			}
			eff := n
			if n == 0 {
				eff = max(0, 2*(l-1))
			}
			for _, m := range allNorms {
				ctx := fmt.Sprintf("len %d N %d %v", l, n, m)
				want := c2rOracle(h, eff)
				for i := range want {
					want[i] *= normFactor(m, eff, true)
				}
				checkR(t, "IRFFTWith "+ctx, IRFFTWith(h, Options{N: n, Norm: m}), want)
			}
		}
	}
	// The round trip holds under every mode, for odd and even lengths.
	for _, n := range []int{1, 2, 7, 10, 33} {
		x := realSignal(n)
		for _, m := range allNorms {
			o := Options{N: n, Norm: m}
			checkR(t, fmt.Sprintf("round trip n=%d %v", n, m), IRFFTWith(RFFTWith(x, o), o), x)
		}
	}
}

// hermitianOracle is numpy.fft.hfft's definition, unnormalized: the forward
// DFT of the length-n signal y whose first half is x and whose second half is
// its conjugate mirror, y[n-j] = conj(x[j]). Only the real parts of x[0] and
// (n even) x[n/2] take part, as in irfft.
func hermitianOracle(x []complex128, n int) []float64 {
	out := c2rOracle(conjAll(x), n) // sum over y with exp(+...) of conj(x) is real, = the exp(-...) sum over x
	return out
}

func conjAll(x []complex128) []complex128 {
	out := make([]complex128, len(x))
	for i, v := range x {
		out[i] = cmplx.Conj(v)
	}
	return out
}

func TestHFFTAgainstDefinition(t *testing.T) {
	for _, l := range []int{0, 1, 2, 3, 4, 6, 9} {
		x := testSpectrum(l)
		for _, n := range []int{0, 1, 2, 3, 5, 6, 2 * l, 2*l - 1, 2*l + 3} {
			if n < 0 {
				continue
			}
			eff := n
			if n == 0 {
				eff = max(0, 2*(l-1))
			}
			// The direct definition: y the Hermitian extension, X = DFT(y).
			y := make([]complex128, eff)
			for j := 0; j < eff; j++ {
				switch {
				case j < len(x) && 2*j <= eff:
					y[j] = x[j]
				case eff-j < len(x) && 2*(eff-j) < eff:
					y[j] = cmplx.Conj(x[eff-j])
				}
			}
			if eff > 0 {
				y[0] = complex(real(y[0]), 0)
				if eff%2 == 0 {
					y[eff/2] = complex(real(y[eff/2]), 0)
				}
			}
			direct := naiveDFT(y)
			for _, m := range allNorms {
				ctx := fmt.Sprintf("len %d N %d %v", l, n, m)
				want := make([]float64, eff)
				for k := range want {
					if math.Abs(imag(direct[k])) > 1e-9*math.Max(1, float64(eff)) {
						t.Fatalf("%s: the oracle's DFT of a Hermitian signal is not real: %v", ctx, direct[k])
					}
					want[k] = real(direct[k]) * normFactor(m, eff, false)
				}
				got := HFFTWith(x, Options{N: n, Norm: m})
				checkR(t, "HFFTWith "+ctx, got, want)
				checkR(t, "hermitianOracle "+ctx, scaledR(hermitianOracle(x, eff), normFactor(m, eff, false)), want)
				// numpy's own identity: hfft(x, n) = n·irfft(conj(x), n) (backward).
				if m == NormBackward && n > 0 {
					checkR(t, "HFFT "+ctx, HFFT(x, n), scaledR(IRFFT(conjAll(x), n), float64(n)))
				}
			}
		}
	}
	if got := HFFT(testSpectrum(4), 0); got == nil || len(got) != 0 {
		t.Fatalf("HFFT(x, 0) = %v, want an empty non-nil slice", got)
	}
	if got := HFFT(testSpectrum(4), -3); got == nil || len(got) != 0 {
		t.Fatalf("HFFT(x, -3) = %v, want an empty non-nil slice", got)
	}
}

func scaledR(x []float64, f float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * f
	}
	return out
}

func TestIHFFTAgainstDefinition(t *testing.T) {
	for _, l := range []int{0, 1, 2, 5, 8, 11} {
		x := realSignal(l)
		for _, n := range lengthsAround(l) {
			eff := n
			if n == 0 {
				eff = l
			}
			for _, m := range allNorms {
				ctx := fmt.Sprintf("len %d N %d %v", l, n, m)
				// numpy: ihfft(x) = conj(rfft(x))/n, i.e. the inverse DFT's half.
				full := naiveIDFTUnnormalized(toC(padR(x, eff)))
				want := scaledC(full, normFactor(m, eff, true))
				if eff > 0 {
					want = want[:eff/2+1]
				}
				checkC(t, "IHFFTWith "+ctx, IHFFTWith(x, Options{N: n, Norm: m}), want)
			}
		}
		checkC(t, "IHFFT default", IHFFT(x), IHFFTWith(x, Options{}))
		// hfft inverts ihfft under every mode (numpy documents the pair).
		for _, m := range allNorms {
			if l == 0 {
				continue
			}
			checkR(t, fmt.Sprintf("hfft(ihfft) len %d %v", l, m),
				HFFTWith(IHFFTWith(x, Options{Norm: m}), Options{N: l, Norm: m}), x)
		}
	}
}

// --- N-D ----------------------------------------------------------------------

// naiveAlongAxes applies the naive 1-D DFT (or the unnormalized inverse) along
// each listed axis of a row-major array.
func naiveAlongAxes(data []complex128, shape, axes []int, inverse bool) []complex128 {
	out := slices.Clone(data)
	stride := make([]int, len(shape))
	acc := 1
	for ax := len(shape) - 1; ax >= 0; ax-- {
		stride[ax] = acc
		acc *= shape[ax]
	}
	for _, ax := range axes {
		n := shape[ax]
		buf := make([]complex128, n)
		for c := 0; c < len(out)/n; c++ {
			// Locate line c by decoding it over the other axes.
			base, r := 0, c
			for a := len(shape) - 1; a >= 0; a-- {
				if a == ax {
					continue
				}
				base += (r % shape[a]) * stride[a]
				r /= shape[a]
			}
			for i := range buf {
				buf[i] = out[base+i*stride[ax]]
			}
			var res []complex128
			if inverse {
				res = naiveIDFTUnnormalized(buf)
			} else {
				res = naiveDFT(buf)
			}
			for i := range res {
				out[base+i*stride[ax]] = res[i]
			}
		}
	}
	return out
}

// cropAxis keeps the first h entries of axis ax of a row-major array.
func cropAxis[T any](data []T, shape []int, ax, h int) ([]T, []int) {
	outShape := slices.Clone(shape)
	outShape[ax] = h
	var out []T
	idx := make([]int, len(shape))
	for i := range data {
		// Decode i into idx.
		r := i
		for a := len(shape) - 1; a >= 0; a-- {
			idx[a] = r % shape[a]
			r /= shape[a]
		}
		if idx[ax] < h {
			out = append(out, data[i])
		}
	}
	return out, outShape
}

func normalizedAxes(axes []int, ndim int) []int {
	if axes == nil {
		axes = make([]int, ndim)
		for i := range axes {
			axes[i] = i
		}
		return axes
	}
	out := make([]int, len(axes))
	for i, a := range axes {
		if a < 0 {
			a += ndim
		}
		out[i] = a
	}
	return out
}

var ndCases = []struct {
	shape []int
	axes  []int
}{
	{[]int{3, 4, 5}, nil},
	{[]int{3, 4, 5}, []int{0}},
	{[]int{3, 4, 5}, []int{1}},
	{[]int{3, 4, 5}, []int{-1}},
	{[]int{3, 4, 5}, []int{2, 0}},
	{[]int{3, 4, 5}, []int{-1, -3, 1}},
	{[]int{2, 1, 6}, []int{1, 2}},
	{[]int{2, 1, 6}, []int{2, 1}},
	{[]int{7}, nil},
	{[]int{6, 4}, []int{1, 0}},
	{[]int{6, 4}, []int{}},
	{[]int{}, nil},
}

func TestFFTNWithAgainstDefinition(t *testing.T) {
	for _, c := range ndCases {
		x := complexGrid(c.shape)
		axes := normalizedAxes(c.axes, len(c.shape))
		cover := 1
		for _, a := range axes {
			cover *= c.shape[a]
		}
		for _, m := range allNorms {
			ctx := fmt.Sprintf("shape %v axes %v %v", c.shape, c.axes, m)
			o := Options{Axes: c.axes, Norm: m}
			checkC(t, "FFTNWith "+ctx, FFTNWith(x, c.shape, o), scaledC(naiveAlongAxes(x, c.shape, axes, false), normFactor(m, cover, false)))
			checkC(t, "IFFTNWith "+ctx, IFFTNWith(x, c.shape, o), scaledC(naiveAlongAxes(x, c.shape, axes, true), normFactor(m, cover, true)))
		}
	}
	x := complexGrid([]int{4, 6})
	checkC(t, "FFTNWith default", FFTNWith(x, []int{4, 6}, Options{}), FFTN(x, []int{4, 6}))
	checkC(t, "IFFTNWith default", IFFTNWith(x, []int{4, 6}, Options{}), IFFTN(x, []int{4, 6}))
	for _, m := range allNorms {
		o := Options{Norm: m}
		checkC(t, "FFT2With", FFT2With(x, [2]int{4, 6}, o), FFTNWith(x, []int{4, 6}, o))
		checkC(t, "IFFT2With", IFFT2With(x, [2]int{4, 6}, o), IFFTNWith(x, []int{4, 6}, o))
		o.Axes = []int{0}
		checkC(t, "FFT2With axes", FFT2With(x, [2]int{4, 6}, o), FFTNWith(x, []int{4, 6}, o))
	}
}

func TestRFFTNWithAgainstDefinition(t *testing.T) {
	for _, c := range ndCases {
		axes := normalizedAxes(c.axes, len(c.shape))
		if len(axes) == 0 {
			continue // refused, see TestOptionsValidation
		}
		r := realGridN(c.shape)
		ra := axes[len(axes)-1]
		cover := 1
		for _, a := range axes {
			cover *= c.shape[a]
		}
		full := naiveAlongAxes(toC(r), c.shape, axes, false)
		want, half := cropAxis(full, c.shape, ra, c.shape[ra]/2+1)
		for _, m := range allNorms {
			ctx := fmt.Sprintf("shape %v axes %v %v", c.shape, c.axes, m)
			o := Options{Axes: c.axes, Norm: m}
			got := RFFTNWith(r, c.shape, o)
			checkC(t, "RFFTNWith "+ctx, got, scaledC(want, normFactor(m, cover, false)))
			// irfftn's definition: the complex inverse along every axis but
			// the real one, then the real inverse along it.
			spec := testSpectrumN(len(got))
			mid := naiveAlongAxes(spec, half, axes[:len(axes)-1], true)
			wantR := c2rAlongAxis(mid, half, c.shape, ra)
			checkR(t, "IRFFTNWith "+ctx, IRFFTNWith(spec, c.shape, o), scaledR(wantR, normFactor(m, cover, true)))
			checkR(t, "IRFFTNWith round trip "+ctx, IRFFTNWith(got, c.shape, o), r)
		}
	}
	r := realGridN([]int{5, 6})
	checkC(t, "RFFTN default", RFFTN(r, []int{5, 6}), RFFT2(r, [2]int{5, 6}))
	checkR(t, "IRFFTN default", IRFFTN(RFFTN(r, []int{5, 6}), []int{5, 6}), r)
	for _, m := range allNorms {
		o := Options{Norm: m}
		for _, shape := range [][2]int{{5, 6}, {4, 7}, {1, 1}} {
			r := realGridN(shape[:])
			s := RFFT2With(r, shape, o)
			checkC(t, "RFFT2With", s, RFFTNWith(r, shape[:], o))
			checkR(t, "IRFFT2With", IRFFT2With(s, shape, o), IRFFTNWith(s, shape[:], o))
			oa := Options{Norm: m, Axes: []int{1, 0}}
			sa := RFFT2With(r, shape, oa)
			checkC(t, "RFFT2With axes", sa, RFFTNWith(r, shape[:], oa))
			checkR(t, "IRFFT2With axes", IRFFT2With(sa, shape, oa), r)
		}
	}
	// The spectrum is read up to its size: missing bins are zero, extra ones
	// ignored, as for IRFFT2.
	s := RFFTN(r, []int{5, 6})
	short := IRFFTN(s[:len(s)-4], []int{5, 6})
	padded := IRFFTN(append(slices.Clone(s[:len(s)-4]), 0, 0, 0, 0), []int{5, 6})
	checkR(t, "IRFFTN short spectrum", short, padded)
	checkR(t, "IRFFTN long spectrum", IRFFTN(append(slices.Clone(s), 9, 9), []int{5, 6}), r)
}

// c2rAlongAxis applies c2rOracle (unnormalized) to every line of axis ax of a
// spectrum of shape half, writing a real array of shape shape.
func c2rAlongAxis(spec []complex128, half, shape []int, ax int) []float64 {
	total := 1
	for _, s := range shape {
		total *= s
	}
	out := make([]float64, total)
	n, h := shape[ax], half[ax]
	inSt, outSt := rowMajorStrides(half), rowMajorStrides(shape)
	for c := 0; c < total/n; c++ {
		ib, ob := 0, 0
		r := c
		for a := len(shape) - 1; a >= 0; a-- {
			if a == ax {
				continue
			}
			ib += (r % shape[a]) * inSt[a]
			ob += (r % shape[a]) * outSt[a]
			r /= shape[a]
		}
		line := make([]complex128, h)
		for k := range line {
			line[k] = spec[ib+k*inSt[ax]]
		}
		for i, v := range c2rOracle(line, n) {
			out[ob+i*outSt[ax]] = v
		}
	}
	return out
}

func realGridN(shape []int) []float64 {
	total := 1
	for _, s := range shape {
		total *= s
	}
	return realSignal(total)
}

func testSpectrumN(n int) []complex128 { return testSpectrum(n) }

// TestBatchedTransforms: transforming the last axis alone is numpy's
// fft(x, axis=-1) on a matrix of signals — every row is FFT(row). Large enough
// to take the parallel path, and checked on the serial one too.
func TestBatchedTransforms(t *testing.T) {
	const rows, n = 300, 96 // 28800 elements, above parThreshold
	x := complexGrid([]int{rows, n})
	r := realGridN([]int{rows, n})
	for _, w := range []int{1, 4} {
		withWorkers(w, func() {
			got := FFTNWith(x, []int{rows, n}, Options{Axes: []int{-1}})
			rs := RFFTNWith(r, []int{rows, n}, Options{Axes: []int{-1}, Norm: NormOrtho})
			for i := 0; i < rows; i++ {
				checkC(t, fmt.Sprintf("w=%d row %d", w, i), got[i*n:(i+1)*n], FFT(x[i*n:(i+1)*n]))
				checkC(t, fmt.Sprintf("w=%d real row %d", w, i), rs[i*(n/2+1):(i+1)*(n/2+1)], RFFTWith(r[i*n:(i+1)*n], Options{Norm: NormOrtho}))
			}
			checkR(t, fmt.Sprintf("w=%d batched real round trip", w), IRFFTNWith(rs, []int{rows, n}, Options{Axes: []int{-1}, Norm: NormOrtho}), r)
			// The real transform along a strided axis (the columns).
			cs := RFFTNWith(r, []int{rows, n}, Options{Axes: []int{0}})
			checkR(t, fmt.Sprintf("w=%d column real round trip", w), IRFFTNWith(cs, []int{rows, n}, Options{Axes: []int{0}}), r)
			col := make([]float64, rows)
			for i := range col {
				col[i] = r[i*n+5]
			}
			wantCol := RFFT(col)
			for k := range wantCol {
				if !near(cs[k*n+5], wantCol[k], rows) {
					t.Fatalf("w=%d column 5 bin %d: %v want %v", w, k, cs[k*n+5], wantCol[k])
				}
			}
		})
	}
}

// --- shifts ---------------------------------------------------------------------

func TestFFTShiftAgainstDefinition(t *testing.T) {
	for n := 0; n <= 9; n++ {
		x := make([]int, n)
		for i := range x {
			x[i] = i
		}
		// numpy: fftshift rolls by n//2, out[(j + n//2) % n] = x[j].
		want := make([]int, n)
		for j := range x {
			want[(j+n/2)%n] = x[j]
		}
		got := FFTShift(x)
		if !slices.Equal(got, want) || got == nil {
			t.Fatalf("FFTShift(%v) = %v, want %v", x, got, want)
		}
		// ifftshift rolls by -(n//2): out[j] = x[(j + n//2) % n].
		for j := range want {
			want[j] = x[(j+n/2)%n]
		}
		if got := IFFTShift(x); !slices.Equal(got, want) || got == nil {
			t.Fatalf("IFFTShift(%v) = %v, want %v", x, got, want)
		}
		if back := IFFTShift(FFTShift(x)); !slices.Equal(back, x) {
			t.Fatalf("n=%d: IFFTShift(FFTShift(x)) = %v", n, back)
		}
		if back := FFTShift(IFFTShift(x)); !slices.Equal(back, x) {
			t.Fatalf("n=%d: FFTShift(IFFTShift(x)) = %v", n, back)
		}
		if n%2 == 0 && !slices.Equal(FFTShift(x), IFFTShift(x)) {
			t.Fatalf("n=%d even: fftshift and ifftshift differ", n)
		}
		// fftshift(fftfreq(n)) is increasing (numpy's documented use).
		if n > 0 {
			if f := FFTShift(FFTFreq(n, 1)); !slices.IsSorted(f) {
				t.Fatalf("n=%d: FFTShift(FFTFreq) not sorted: %v", n, f)
			}
		}
	}
	// It is generic: a complex spectrum works the same.
	c := []complex128{1, 2i, 3, 4i, 5}
	if got := FFTShift(c); !slices.Equal(got, []complex128{4i, 5, 1, 2i, 3}) {
		t.Fatalf("FFTShift(complex) = %v", got)
	}
}

func TestFFTShiftNAgainstDefinition(t *testing.T) {
	cases := []struct {
		shape []int
		axes  []int
	}{
		{[]int{3, 4}, nil}, {[]int{3, 4}, []int{0}}, {[]int{3, 4}, []int{-1}},
		{[]int{5, 2, 3}, nil}, {[]int{5, 2, 3}, []int{0, 2}}, {[]int{5, 2, 3}, []int{1}},
		{[]int{7}, nil}, {[]int{1, 1}, nil}, {[]int{2, 3, 4, 5}, []int{3, 0}},
		{[]int{4, 5}, []int{}}, {[]int{}, nil},
	}
	for _, c := range cases {
		total := 1
		for _, s := range c.shape {
			total *= s
		}
		x := make([]int, total)
		for i := range x {
			x[i] = i
		}
		axes := normalizedAxes(c.axes, len(c.shape))
		shifted := make([]bool, len(c.shape))
		for _, a := range axes {
			shifted[a] = true
		}
		// The definition: each shifted axis of length n is rolled by n//2
		// (fftshift) or -(n//2) (ifftshift), element by element.
		fwd, inv := make([]int, total), make([]int, total)
		idx := make([]int, len(c.shape))
		for i := range x {
			r := i
			for a := len(c.shape) - 1; a >= 0; a-- {
				idx[a] = r % c.shape[a]
				r /= c.shape[a]
			}
			fo, io := 0, 0
			for a, j := range idx {
				n := c.shape[a]
				fj, ij := j, j
				if shifted[a] {
					fj = (j + n/2) % n
					ij = ((j-n/2)%n + n) % n
				}
				fo = fo*n + fj
				io = io*n + ij
			}
			fwd[fo] = x[i]
			inv[io] = x[i]
		}
		ctx := fmt.Sprintf("shape %v axes %v", c.shape, c.axes)
		if got := FFTShiftN(x, c.shape, c.axes); !slices.Equal(got, fwd) {
			t.Fatalf("FFTShiftN %s = %v, want %v", ctx, got, fwd)
		}
		if got := IFFTShiftN(x, c.shape, c.axes); !slices.Equal(got, inv) {
			t.Fatalf("IFFTShiftN %s = %v, want %v", ctx, got, inv)
		}
		if back := IFFTShiftN(FFTShiftN(x, c.shape, c.axes), c.shape, c.axes); !slices.Equal(back, x) {
			t.Fatalf("%s: ifftshift(fftshift(x)) = %v", ctx, back)
		}
	}
	// On one axis, FFTShiftN is FFTShift.
	x := []float64{0, 1, 2, 3, 4, 5, 6}
	if !slices.Equal(FFTShiftN(x, []int{7}, nil), FFTShift(x)) {
		t.Fatal("FFTShiftN on one axis differs from FFTShift")
	}
}

// --- NextFastLen ----------------------------------------------------------------

// smoothSet enumerates every 7-smooth number that fits in an int, sorted: an
// oracle independent of nextSmooth7's search (about 60k values).
func smoothSet() []int {
	var out []int
	for p7 := 1; ; p7 *= 7 {
		for p5 := p7; ; p5 *= 5 {
			for p3 := p5; ; p3 *= 3 {
				for p2 := p3; ; p2 *= 2 {
					out = append(out, p2)
					if p2 > math.MaxInt/2 {
						break
					}
				}
				if p3 > math.MaxInt/3 {
					break
				}
			}
			if p5 > math.MaxInt/5 {
				break
			}
		}
		if p7 > math.MaxInt/7 {
			break
		}
	}
	slices.Sort(out)
	return out
}

func TestNextFastLen(t *testing.T) {
	is7Smooth := func(m int) bool {
		for _, p := range []int{2, 3, 5, 7} {
			for m%p == 0 {
				m /= p
			}
		}
		return m == 1
	}
	// Small targets against a linear scan of the definition.
	for target := 0; target <= 3000; target++ {
		want, wantReal := target, target
		if target > 1 {
			for want = target; !is7Smooth(want); want++ {
			}
			for wantReal = target; wantReal%2 != 0 || !is7Smooth(wantReal); wantReal++ {
			}
		}
		if got := NextFastLen(target, false); got != want {
			t.Fatalf("NextFastLen(%d, false) = %d, want %d", target, got, want)
		}
		if got := NextFastLen(target, true); got != wantReal {
			t.Fatalf("NextFastLen(%d, true) = %d, want %d", target, got, wantReal)
		}
	}
	// Large targets, up to the top of the int range, against the enumeration.
	set := smoothSet()
	top := set[len(set)-1]
	var evens []int
	for _, m := range set {
		if m%2 == 0 {
			evens = append(evens, m)
		}
	}
	targets := []int{1 << 40, 1<<40 + 1, 999_999_999_989, top - 1, top, evens[len(evens)-1]}
	for v := 12345; v < math.MaxInt/3; v *= 3 {
		targets = append(targets, v, v+1)
	}
	for _, target := range targets {
		i, _ := slices.BinarySearch(set, target)
		if got := NextFastLen(target, false); got != set[i] {
			t.Fatalf("NextFastLen(%d, false) = %d, want %d", target, got, set[i])
		}
		j, _ := slices.BinarySearch(evens, target)
		if j < len(evens) {
			if got := NextFastLen(target, true); got != evens[j] {
				t.Fatalf("NextFastLen(%d, true) = %d, want %d", target, got, evens[j])
			}
		}
	}
	// Past the last qualifying int: a panic, not a wrong or wrapped value.
	mustPanicWith(t, "NextFastLen past the top", "fft: NextFastLen: no fast length", func() { NextFastLen(top+1, false) })
	mustPanicWith(t, "NextFastLen MaxInt", "fft: NextFastLen: no fast length", func() { NextFastLen(math.MaxInt, false) })
	mustPanicWith(t, "NextFastLen real past the top", "fft: NextFastLen: no fast length", func() { NextFastLen(evens[len(evens)-1]+1, true) })
	mustPanicWith(t, "NextFastLen real MaxInt", "fft: NextFastLen: no fast length", func() { NextFastLen(math.MaxInt, true) })
	mustPanicWith(t, "NextFastLen negative", "fft: NextFastLen: negative", func() { NextFastLen(-1, false) })
}

// --- plans ------------------------------------------------------------------------

func TestPlanNormMethods(t *testing.T) {
	for _, n := range []int{0, 1, 6, 7, 16} {
		x := cmplxSignal(n)
		r := realSignal(n)
		p, rp := NewPlan(n), NewRealPlan(n)
		for _, m := range allNorms {
			ctx := fmt.Sprintf("n=%d %v", n, m)
			o := Options{Norm: m}
			checkC(t, "Plan.FFTNorm "+ctx, p.FFTNorm(make([]complex128, n), x, m), FFTWith(x, o))
			checkC(t, "Plan.IFFTNorm "+ctx, p.IFFTNorm(make([]complex128, n), x, m), IFFTWith(x, o))
			if n == 0 {
				continue
			}
			s := rp.RFFTNorm(make([]complex128, n/2+1), r, m)
			checkC(t, "RealPlan.RFFTNorm "+ctx, s, RFFTWith(r, o))
			checkR(t, "RealPlan.IRFFTNorm "+ctx, rp.IRFFTNorm(make([]float64, n), s, m), r)
			checkR(t, "RealPlan.IRFFTNorm vs IRFFTWith "+ctx, rp.IRFFTNorm(make([]float64, n), s, m), IRFFTWith(s, Options{N: n, Norm: m}))
		}
	}
	shape := []int{4, 6}
	x := complexGrid(shape)
	r := realGridN(shape)
	pn, r2 := NewPlanN(shape...), NewRealPlan2(4, 6)
	for _, m := range allNorms {
		o := Options{Norm: m}
		checkC(t, "PlanN.FFTNorm", pn.FFTNorm(make([]complex128, 24), x, m), FFTNWith(x, shape, o))
		checkC(t, "PlanN.IFFTNorm", pn.IFFTNorm(make([]complex128, 24), x, m), IFFTNWith(x, shape, o))
		s := r2.RFFTNorm(make([]complex128, r2.SpectrumLen()), r, m)
		checkC(t, "RealPlan2.RFFTNorm", s, RFFTNWith(r, shape, o))
		checkR(t, "RealPlan2.IRFFTNorm", r2.IRFFTNorm(make([]float64, 24), s, m), r)
	}
	// Under NormBackward the Norm methods are the plain ones, bit for bit.
	p := NewPlan(12)
	a, b := p.FFTNorm(make([]complex128, 12), cmplxSignal(12), NormBackward), p.FFT(make([]complex128, 12), cmplxSignal(12))
	if !slices.Equal(a, b) {
		t.Fatal("Plan.FFTNorm(NormBackward) differs from Plan.FFT")
	}
	a, b = p.IFFTNorm(make([]complex128, 12), cmplxSignal(12), NormBackward), p.IFFT(make([]complex128, 12), cmplxSignal(12))
	if !slices.Equal(a, b) {
		t.Fatal("Plan.IFFTNorm(NormBackward) differs from Plan.IFFT")
	}
	rp := NewRealPlan(12)
	ra, rb := rp.IRFFTNorm(make([]float64, 12), testSpectrum(7), NormBackward), rp.IRFFT(make([]float64, 12), testSpectrum(7))
	if !slices.Equal(ra, rb) {
		t.Fatal("RealPlan.IRFFTNorm(NormBackward) differs from RealPlan.IRFFT")
	}
	// An unknown Norm panics before the destination is written.
	dst := []complex128{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}
	mustPanicWith(t, "Plan.FFTNorm", "fft: unknown Norm", func() { p.FFTNorm(dst, cmplxSignal(12), Norm(5)) })
	mustPanicWith(t, "Plan.IFFTNorm", "fft: unknown Norm", func() { p.IFFTNorm(dst, cmplxSignal(12), Norm(-1)) })
	mustPanicWith(t, "RealPlan.RFFTNorm", "fft: unknown Norm", func() { rp.RFFTNorm(dst[:7], realSignal(12), Norm(3)) })
	if dst[0] != 7 || dst[11] != 7 {
		t.Fatal("a Norm method wrote before rejecting an unknown Norm")
	}
	mustPanicWith(t, "RealPlan.IRFFTNorm", "fft: unknown Norm", func() { rp.IRFFTNorm(make([]float64, 12), dst[:7], Norm(3)) })
	mustPanicWith(t, "PlanN.FFTNorm", "fft: unknown Norm", func() { pn.FFTNorm(make([]complex128, 24), x, Norm(3)) })
	mustPanicWith(t, "PlanN.IFFTNorm", "fft: unknown Norm", func() { pn.IFFTNorm(make([]complex128, 24), x, Norm(3)) })
	mustPanicWith(t, "RealPlan2.RFFTNorm", "fft: unknown Norm", func() { r2.RFFTNorm(make([]complex128, 16), r, Norm(3)) })
	mustPanicWith(t, "RealPlan2.IRFFTNorm", "fft: unknown Norm", func() { r2.IRFFTNorm(make([]float64, 24), make([]complex128, 16), Norm(3)) })
}

// --- validation -------------------------------------------------------------------

func TestOptionsValidation(t *testing.T) {
	x, r := cmplxSignal(8), realSignal(8)
	bad := Norm(3)
	oneD := map[string]func(Options){
		"FFTWith":   func(o Options) { FFTWith(x, o) },
		"IFFTWith":  func(o Options) { IFFTWith(x, o) },
		"RFFTWith":  func(o Options) { RFFTWith(r, o) },
		"IRFFTWith": func(o Options) { IRFFTWith(x, o) },
		"HFFTWith":  func(o Options) { HFFTWith(x, o) },
		"IHFFTWith": func(o Options) { IHFFTWith(r, o) },
	}
	for name, f := range oneD {
		mustPanicWith(t, name+" negative N", "fft: "+name+": Options.N must not be negative", func() { f(Options{N: -1}) })
		mustPanicWith(t, name+" axes", "fft: "+name+": Options.Axes applies only", func() { f(Options{Axes: []int{0}}) })
		mustPanicWith(t, name+" norm", "fft: unknown Norm", func() { f(Options{Norm: bad}) })
	}
	g := complexGrid([]int{2, 4})
	rg := realGridN([]int{2, 4})
	spec := make([]complex128, 2*3)
	nd := map[string]func(Options){
		"FFTNWith":   func(o Options) { FFTNWith(g, []int{2, 4}, o) },
		"IFFTNWith":  func(o Options) { IFFTNWith(g, []int{2, 4}, o) },
		"FFT2With":   func(o Options) { FFT2With(g, [2]int{2, 4}, o) },
		"IFFT2With":  func(o Options) { IFFT2With(g, [2]int{2, 4}, o) },
		"RFFTNWith":  func(o Options) { RFFTNWith(rg, []int{2, 4}, o) },
		"IRFFTNWith": func(o Options) { IRFFTNWith(spec, []int{2, 4}, o) },
	}
	for name, f := range nd {
		mustPanicWith(t, name+" N", "fft: "+name+": Options.N applies only", func() { f(Options{N: 4}) })
		mustPanicWith(t, name+" norm", "fft: unknown Norm", func() { f(Options{Norm: bad}) })
		mustPanicWith(t, name+" axis range", "fft: "+name+": axis 2 out of range", func() { f(Options{Axes: []int{2}}) })
		mustPanicWith(t, name+" negative axis range", "fft: "+name+": axis -3 out of range", func() { f(Options{Axes: []int{-3}}) })
		mustPanicWith(t, name+" duplicate axis", "fft: "+name+": axis -1 listed twice", func() { f(Options{Axes: []int{1, -1}}) })
	}
	mustPanicWith(t, "RFFT2With N", "fft: RFFT2With: Options.N applies only", func() { RFFT2With(rg, [2]int{2, 4}, Options{N: 1}) })
	mustPanicWith(t, "RFFT2With norm", "fft: unknown Norm", func() { RFFT2With(rg, [2]int{2, 4}, Options{Norm: bad}) })
	mustPanicWith(t, "IRFFT2With N", "fft: IRFFT2With: Options.N applies only", func() { IRFFT2With(spec, [2]int{2, 4}, Options{N: 1}) })
	mustPanicWith(t, "IRFFT2With norm", "fft: unknown Norm", func() { IRFFT2With(spec, [2]int{2, 4}, Options{Norm: bad}) })
	mustPanicWith(t, "RFFT2With axes", "fft: RFFTNWith: axis 5", func() { RFFT2With(rg, [2]int{2, 4}, Options{Axes: []int{5}}) })
	mustPanicWith(t, "IRFFT2With axes", "fft: IRFFTNWith: axis 5", func() { IRFFT2With(spec, [2]int{2, 4}, Options{Axes: []int{5}}) })

	// The real N-D transforms need an axis to run the real transform along.
	mustPanicWith(t, "RFFTNWith no axis", "fft: RFFTNWith: no axis", func() { RFFTNWith(rg, []int{2, 4}, Options{Axes: []int{}}) })
	mustPanicWith(t, "RFFTN scalar", "fft: RFFTNWith: no axis", func() { RFFTN([]float64{1}, nil) })
	mustPanicWith(t, "IRFFTNWith no axis", "fft: IRFFTNWith: no axis", func() { IRFFTNWith(spec, []int{2, 4}, Options{Axes: []int{}}) })
	mustPanicWith(t, "IRFFTN scalar", "fft: IRFFTNWith: no axis", func() { IRFFTN([]complex128{1}, nil) })

	// Shapes: mismatch, non-positive, overflow — each before allocating.
	const mismatch = "fft: shape product does not match len(data)"
	mustPanicWith(t, "FFTNWith mismatch", mismatch, func() { FFTNWith(g, []int{3, 4}, Options{}) })
	mustPanicWith(t, "RFFTN mismatch", mismatch, func() { RFFTN(rg, []int{3, 4}) })
	mustPanicWith(t, "RFFT2With mismatch", mismatch, func() { RFFT2With(rg, [2]int{3, 4}, Options{}) })
	mustPanicWith(t, "FFTShiftN mismatch", mismatch, func() { FFTShiftN(rg, []int{3, 4}, nil) })
	const overflow = "fft: shape product overflows int"
	huge := []int{1 << 32, 1 << 32}
	mustPanicWith(t, "FFTNWith overflow", overflow, func() { FFTNWith(nil, huge, Options{}) })
	mustPanicWith(t, "IFFTNWith overflow", overflow, func() { IFFTNWith(nil, huge, Options{Axes: []int{0}}) })
	mustPanicWith(t, "RFFTN overflow", overflow, func() { RFFTN(nil, huge) })
	mustPanicWith(t, "IRFFTN overflow", overflow, func() { IRFFTN(nil, huge) })
	mustPanicWith(t, "RFFT2With overflow", overflow, func() { RFFT2With(nil, [2]int{1 << 32, 1 << 32}, Options{}) })
	mustPanicWith(t, "IRFFT2With overflow", overflow, func() { IRFFT2With(nil, [2]int{1 << 32, 1 << 32}, Options{}) })
	mustPanicWith(t, "IFFTShiftN overflow", overflow, func() { IFFTShiftN([]int(nil), huge, nil) })
	const positive = "fft: shape lengths must be positive"
	mustPanicWith(t, "IRFFTN zero", positive, func() { IRFFTN(nil, []int{4, 0}) })
	mustPanicWith(t, "FFTShiftN negative", positive, func() { FFTShiftN([]int(nil), []int{-1, -1}, nil) })
	mustPanicWith(t, "FFTShiftN axis", "fft: FFTShiftN: axis 1 out of range", func() { FFTShiftN([]int{1, 2}, []int{2}, []int{1}) })
	mustPanicWith(t, "IFFTShiftN duplicate", "fft: IFFTShiftN: axis 0 listed twice", func() { IFFTShiftN([]int{1, 2}, []int{2}, []int{0, 0}) })
}

// TestPlanNForSharesPlans: the all-axes plan is FFTN's own cached plan, and a
// subset plan is built once per shape and axes and plans no unlisted axis.
func TestPlanNForSharesPlans(t *testing.T) {
	if planNFor([]int{3, 5}, []int{1, 0}) != cachedPlanN([]int{3, 5}) {
		t.Fatal("the all-axes plan is not FFTN's cached plan")
	}
	p := planNFor([]int{1000003, 4}, []int{1})
	if p != planNFor([]int{1000003, 4}, []int{1}) {
		t.Fatal("a subset plan was rebuilt")
	}
	if p.axes[0] != nil || p.axes[1] == nil {
		t.Fatalf("subset plan axes = %v, want only axis 1 planned", p.axes)
	}
}
