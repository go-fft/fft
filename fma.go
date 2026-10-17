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

// fmaSplit says whether a split pass (mode != 0) of an FMA plan fuses its
// product: on amd64 every split kernel has an FMA twin; arm64's split
// kernels have none, and a split pass there keeps its kernel, whose Go
// fallback would not read the split layout.
var fmaSplit = fmaSplitDefault()

// markFMA sets the stages of an FMA plan that fuse their twiddle products
// (fmaRadix, fmaSplit). The final pass (ido == 1) has no twiddles: marking it
// changes nothing but the entry point, which then runs the default kernel.
func (p *skPlan) markFMA(split bool) {
	for k := range p.stages {
		st := &p.stages[k]
		st.fma = fmaRadix(st.r) && (st.split == 0 || split)
	}
}

// passFMA is pass for a stage that fuses its products (st.fma).
func (st *skStage) passFMA(ch, cc, twX []complex128, inverse bool) {
	if !kernels.StockhamPassLayoutFMA(st.split, st.r, st.ido, st.l1, cc, ch, twX, inverse, st.wide) {
		st.passScalarFMA(ch, cc, st.l1, inverse)
	}
}

// passScalarFMA is the Go pass of a stage that fuses its products over l1
// blocks: the butterfly the default pass runs (bflyAny), then cmulFMA for
// every output j >= 1 of every point i >= 1. It is the FMA kernels'
// reference, operation for operation, and their fallback. The final pass
// has no product and is the default one.
func (st *skStage) passScalarFMA(ch, cc []complex128, l1 int, inverse bool) {
	if st.ido == 1 {
		st.passScalarL(ch, cc, l1, inverse)
		return
	}
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
			bflyAny(&y, &x, r, s)
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

// bflyAny is the size-r DFT of x[:r] into y[:r] for a radix fmaRadix
// accepts: the butterfly the default pass of that radix runs.
func bflyAny(y, x *[20]complex128, r int, s float64) {
	switch r {
	case 2, 3, 4, 5, 8:
		var a, b [8]complex128
		copy(a[:r], x[:r])
		bflyR(&b, &a, r, s)
		copy(y[:r], b[:r])
	case 9:
		var a, b [9]complex128
		copy(a[:], x[:9])
		bfly9(&b, &a, s)
		copy(y[:9], b[:])
	case 12:
		var a, b [12]complex128
		copy(a[:], x[:12])
		bfly12(&b, &a, s)
		copy(y[:12], b[:])
	case 16:
		var a, b [16]complex128
		copy(a[:], x[:16])
		bfly16(&b, &a, s)
		copy(y[:16], b[:])
	default: // 10, 15, 20
		comp2Bfly(y, x, comp2Maps[r], s)
	}
}

// passBatchFMA is passBatch for a stage that fuses its products.
func (st *skStage) passBatchFMA(ch, cc, tw []complex128, w, sIn, sOut int, inverse bool) {
	if !kernels.StockhamBatchPassFMA(st.r, st.ido, st.l1, cc, ch, tw, w, sIn, sOut, inverse) {
		st.passBatchScalarFMA(ch, cc, tw, w, sIn, sOut, inverse)
	}
}

// passBatchScalarFMA is passBatchScalar with cmulFMA: every value of the
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
				bflyR(&y, &x, r, s)
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
