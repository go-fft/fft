package fft

import (
	"math"

	"github.com/go-fft/fft/internal/kernels"
)

// The opt-in FMA mode (Round 41, Options.FMA).
//
// By default every amd64 pass kernel is bit-identical to the Go passes as
// compiled for GOAMD64=v1, which never fuse a multiply into an add. An FMA
// plan replaces one operation: the product of each output by its twiddle
// between two passes, x·w, which the default computes with three roundings,
// (xr·wr − xi·wi, xi·wr + xr·wi), becomes cmulFMA's two fused operations,
// each rounded once. Everything else (the butterflies, their constant
// rotations, the untangle of the real transforms, the chirp and kernel
// products of Rader and Bluestein) is the default's. FFTW's FMA codelets
// (genfft's fma mode) fuse the butterflies' rotations too; the twiddle
// product is the part of a twiddled pass FMA shortens the most, and the
// part every pass kernel shares (gen.go's twStore), so one change of the
// generator gives every twiddled kernel its twin.
//
// The FMA kernels are pinned to passScalarFMA, the Go pass with cmulFMA in
// place of the product, bit for bit, as the default kernels are pinned to the
// default passes. An FMA plan is a different plan: it is built and cached
// apart from the default one (planKey), so that the two never mix, and it
// is built only where kernels.UseFMA says the CPU has FMA3 (fmaOn).
// Everywhere else an FMA request builds the default plan.

// cmulFMA is x·w with fused multiply-adds:
//
//	re = fma(xr, wr, −(xi·wi)),  im = fma(xi, wr, xr·wi)
//
// the product every FMA kernel computes (skEmit.fmaProduct in gen.go). The
// explicit conversions keep the inner products rounded, whatever the
// compiler would otherwise fuse.
func cmulFMA(x, w complex128) complex128 {
	xr, xi, wr, wi := real(x), imag(x), real(w), imag(w)
	return complex(math.FMA(xr, wr, -float64(xi*wi)), math.FMA(xi, wr, float64(xr*wi)))
}

// fmaAvailable is kernels.UseFMA: whether FMA plans can run here. A variable
// so that the tests can build FMA plans on any machine; they then run the Go
// reference passes wherever the FMA kernels do not.
var fmaAvailable = kernels.UseFMA

// fmaOn reports whether a plan asked for with fma is built as an FMA plan:
// asked for, and the CPU runs the FMA kernels. A pure function, tested both
// ways on any machine.
func fmaOn(fma, available bool) bool { return fma && available }

// fmaRadix reports whether an FMA plan fuses the twiddle products of a pass
// of radix r: the radices with an amd64 pass kernel, every one of which has
// an FMA twin. The radix-7, -11 and -13 passes and the general one run in Go
// and keep the default product, so an FMA plan with such a pass is as fast
// there as the default one.
func fmaRadix(r int) bool {
	switch r {
	case 2, 3, 4, 5, 8, 9, 10, 12, 15, 16, 20:
		return true
	}
	return false
}

// fmaSplit says whether an FMA plan keeps the split layout between its
// passes: on amd64 every split kernel has an FMA twin; arm64's NEON split
// kernels have none, so an FMA plan there (built only by the tests) runs
// interleaved, every pass on its Go reference.
var fmaSplit = fmaSplitDefault()

// markFMA sets the stages of an FMA plan that fuse their products
// (fmaRadix): every one whose radix has FMA kernels, the final pass
// included, whose butterfly may have a constant product to fuse.
func (p *skPlan) markFMA() {
	for k := range p.stages {
		st := &p.stages[k]
		st.fma = fmaRadix(st.r)
	}
}

// passFMA is pass for a stage that fuses its products (st.fma).
func (st *skStage) passFMA(ch, cc, twX []complex128, inverse bool) {
	if !kernels.StockhamPassLayoutFMA(st.split, st.r, st.ido, st.l1, cc, ch, twX, inverse, st.wide) {
		st.passScalarFMA(ch, cc, st.l1, inverse)
	}
}

// passScalarFMA is the Go pass of a stage that fuses its products over l1
// blocks: the butterfly of an FMA plan (bflyAnyFMA), then cmulFMA for every
// output j >= 1 of every point i >= 1 (a final pass, ido == 1, has none).
// It is the FMA kernels' reference, operation for operation, and their
// fallback.
func (st *skStage) passScalarFMA(ch, cc []complex128, l1 int, inverse bool) {
	tw := st.tw
	if inverse {
		tw = st.twc
	}
	r, ido, m, s := st.r, st.ido, st.ido-1, dirSign(inverse)
	var x, y [20]complex128
	for k := 0; k < l1; k++ {
		for i := 0; i < ido; i++ {
			for j := 0; j < r; j++ {
				x[j] = cc[i+ido*(j+r*k)]
			}
			bflyAnyFMA(&y, &x, r, s)
			for j := 0; j < r; j++ {
				v := y[j]
				if i > 0 && j > 0 {
					v = cmulFMA(v, tw[(j-1)*m+i-1])
				}
				ch[i+ido*(k+l1*j)] = v
			}
		}
	}
}

// bflyAnyFMA is the size-r DFT of x[:r] into y[:r] in an FMA plan, for a
// radix fmaRadix accepts: the default pass's butterfly, except for radix 8,
// 9 and 16, whose constant products the FMA kernels fuse (bfly8FMA,
// bfly9FMA, bfly16FMA).
func bflyAnyFMA(y, x *[20]complex128, r int, s float64) {
	switch r {
	case 2, 3, 4, 5, 8:
		var a, b [8]complex128
		copy(a[:r], x[:r])
		bflyRFMA(&b, &a, r, s)
		copy(y[:r], b[:r])
	case 9:
		var a, b [9]complex128
		copy(a[:], x[:9])
		bfly9FMA(&b, &a, s)
		copy(y[:9], b[:])
	case 12:
		var a, b [12]complex128
		copy(a[:], x[:12])
		bfly12(&b, &a, s)
		copy(y[:12], b[:])
	case 16:
		var a, b [16]complex128
		copy(a[:], x[:16])
		bfly16FMA(&b, &a, s)
		copy(y[:16], b[:])
	default: // 10, 15, 20
		comp2Bfly(y, x, comp2Maps[r], s)
	}
}

// bflyRFMA is bflyR in an FMA plan: radix 8 is bfly8FMA.
func bflyRFMA(y, x *[8]complex128, r int, s float64) {
	if r == 8 {
		bfly8FMA(y, x, s)
		return
	}
	bflyR(y, x, r, s)
}

// fmaPM returns a + h·v and a − h·v, each component with one rounding: the
// FMA kernels' fused √2/2 products (hpm in gen.go).
func fmaPM(a, v complex128, h float64) (complex128, complex128) {
	return complex(math.FMA(h, real(v), real(a)), math.FMA(h, imag(v), imag(a))),
		complex(math.FMA(-h, real(v), real(a)), math.FMA(-h, imag(v), imag(a)))
}

// bfly8FMA is bfly8 with its ±45° and ±135° rotations fused into the sums
// that follow them: a5 and a7 keep their unscaled form, a5 + rotS(a5) and
// rotS(a7) − a7, and y1, y5 = a4 ± h·a5, y3, y7 = a6 ± h·a7 are fused
// multiply-adds, as FFTW's FMA codelets do (genfft -fma).
func bfly8FMA(y, x *[8]complex128, s float64) {
	a1, a5 := x[1]+x[5], x[1]-x[5]
	a3, a7 := x[3]+x[7], x[3]-x[7]
	a1, a3 = a1+a3, a1-a3
	a3 = rotS(a3, s)
	a7 = rotS(a7, s)
	a5, a7 = a5+a7, a5-a7
	a5 = complex(real(a5)-s*imag(a5), imag(a5)+s*real(a5))
	a7 = complex(-s*imag(a7)-real(a7), s*real(a7)-imag(a7))
	a0, a4 := x[0]+x[4], x[0]-x[4]
	a2, a6 := x[2]+x[6], x[2]-x[6]
	a0, a2 = a0+a2, a0-a2
	a6 = rotS(a6, s)
	a4, a6 = a4+a6, a4-a6
	y[0], y[4] = a0+a1, a0-a1
	y[2], y[6] = a2+a3, a2-a3
	y[1], y[5] = fmaPM(a4, a5, hsqt2)
	y[3], y[7] = fmaPM(a6, a7, hsqt2)
}

// radix16RotFMA is radix16Rot with the product by c fused into the sum:
// fma(c, a, swap(a)·(−ss, ss)), the FMA kernels' rotC.
func radix16RotFMA(a complex128, c, ss float64) complex128 {
	return complex(math.FMA(c, real(a), float64(-ss*imag(a))), math.FMA(c, imag(a), float64(ss*real(a))))
}

// bfly16FMA is bfly16 with radix16RotFMA for its internal twiddles.
func bfly16FMA(y, x *[16]complex128, s float64) {
	var v [4][4]complex128
	for n2 := 0; n2 < 4; n2++ {
		y0, y1, y2, y3 := bfly4(x[n2], x[n2+4], x[n2+8], x[n2+12], s)
		v[n2] = [4]complex128{y0, y1, y2, y3}
	}
	v[1][1] = radix16RotFMA(v[1][1], r16C1, s*r16S1)
	v[1][2] = radix16RotFMA(v[1][2], hsqt2, s*hsqt2)
	v[1][3] = radix16RotFMA(v[1][3], r16S1, s*r16C1)
	v[2][1] = radix16RotFMA(v[2][1], hsqt2, s*hsqt2)
	v[2][2] = rotS(v[2][2], s)
	v[2][3] = radix16RotFMA(v[2][3], -hsqt2, s*hsqt2)
	v[3][1] = radix16RotFMA(v[3][1], r16S1, s*r16C1)
	v[3][2] = radix16RotFMA(v[3][2], -hsqt2, s*hsqt2)
	v[3][3] = radix16RotFMA(v[3][3], -r16C1, s*-r16S1)
	for k1 := 0; k1 < 4; k1++ {
		y[k1], y[k1+4], y[k1+8], y[k1+12] = bfly4(v[0][k1], v[1][k1], v[2][k1], v[3][k1], s)
	}
}

// bfly9FMA is bfly9 with radix16RotFMA for its internal twiddles.
func bfly9FMA(y, x *[9]complex128, s float64) {
	var v [3][3]complex128
	for n2 := 0; n2 < 3; n2++ {
		v[n2][0], v[n2][1], v[n2][2] = bfly3(x[n2], x[n2+3], x[n2+6], s)
	}
	v[1][1] = radix16RotFMA(v[1][1], r9C1, s*r9S1)
	v[1][2] = radix16RotFMA(v[1][2], r9C2, s*r9S2)
	v[2][1] = radix16RotFMA(v[2][1], r9C2, s*r9S2)
	v[2][2] = radix16RotFMA(v[2][2], r9C4, s*r9S4)
	for k1 := 0; k1 < 3; k1++ {
		y[k1], y[k1+3], y[k1+6] = bfly3(v[0][k1], v[1][k1], v[2][k1], s)
	}
}

// passBatchFMA is passBatch for a stage that fuses its products.
func (st *skStage) passBatchFMA(ch, cc, tw []complex128, w, sIn, sOut int, inverse bool) {
	if !kernels.StockhamBatchPassFMA(st.r, st.ido, st.l1, cc, ch, tw, w, sIn, sOut, inverse) {
		st.passBatchScalarFMA(ch, cc, tw, w, sIn, sOut, inverse)
	}
}

// passBatchScalarFMA is passBatchScalar in an FMA plan: every value of the
// batch gets passScalarFMA's arithmetic, so a strip is bit-identical to its
// lines transformed one by one.
func (st *skStage) passBatchScalarFMA(ch, cc, tw []complex128, w, sIn, sOut int, inverse bool) {
	r, ido, l1 := st.r, st.ido, st.l1
	s := dirSign(inverse)
	var x, y [8]complex128
	for k := 0; k < l1; k++ {
		for i := 0; i < ido; i++ {
			for c := 0; c < w; c++ {
				for j := 0; j < r; j++ {
					x[j] = cc[(i+ido*(j+r*k))*sIn+c]
				}
				bflyRFMA(&y, &x, r, s)
				for j := 0; j < r; j++ {
					v := y[j]
					if i > 0 && j > 0 {
						v = cmulFMA(v, tw[(i-1)*(r-1)+j-1])
					}
					ch[(i+ido*(k+l1*j))*sOut+c] = v
				}
			}
		}
	}
}
