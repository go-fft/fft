package kernels

import "math"

// The split layout on amd64 (genStockhamSplit, stockhamsplit_amd64.s): a run
// of consecutive passes of a split radix keeps its data block-split between
// them. Four points 4q .. 4q+3 occupy the 64 bytes they occupy interleaved,
// as re(4q), re(4q+2), re(4q+1), re(4q+3), then the imaginary parts in the
// same order: the order VUNPCKLPD and VUNPCKHPD give two interleaved YMM
// loads, one in-lane shuffle per register each way. A YMM register then
// holds one part of four points, so the ±i rotations need no shuffle and a
// twiddle product is four multiplies and two adds per four points with no
// shuffle at all. The arithmetic is the interleaved kernels', so the result
// is the same bits.

// The layout modes splitModes assigns: the first pass of a run reads
// interleaved and writes split, the passes between are split on both sides,
// the last reads split and writes interleaved, and a run of one pass reads
// and writes interleaved (it splits its points on loading and joins them on
// storing).
const (
	splitOut  = 1
	splitBoth = 2
	splitIn   = 3
	splitNone = 4
)

// UseStockhamSplit reports whether plans built from now on keep their data
// split between passes (StockhamSplitModes); splitDefault decides. A
// variable so the tests can turn it on and off; a plan keeps the layout it
// was built with, and its passes all fall back to Go, interleaved, when
// UseStockhamAVX2 is off.
var UseStockhamSplit = splitDefault(useAVX2, supportsAVX512F(), IntelCPU)

// splitDefault turns the split layout on with the AVX2 kernels, except where
// the AVX-512 kernels run (they have no split form, and nothing here was
// timed against them) and on Intel CPUs: the only AVX2-only Intel host at
// hand (Haswell, cfarm13) carried a load of 16–29 throughout Round 23, so the
// layout, measured on Zen 3 only, is not imposed on them (the vendor rule of
// r8MaxPow2AMD64). A pure function, tested on any machine.
func splitDefault(avx2, avx512, intel bool) bool { return avx2 && !avx512 && !intel }

// splitK holds the split kernels' constants, one 32-byte row each (the
// split* offsets in gen.go): √2/2, and −0, the sign flip.
var splitK = [2][4]float64{
	{0.707106781186547524400844362104849, 0.707106781186547524400844362104849, 0.707106781186547524400844362104849, 0.707106781186547524400844362104849},
	{math.Copysign(0, -1), math.Copysign(0, -1), math.Copysign(0, -1), math.Copysign(0, -1)},
}

// The split kernels by radix and mode, per direction.
var (
	skSplitFwd = [9][5]skPassFn{
		4: {splitOut: skSplit4ISFwd, splitBoth: skSplit4SSFwd, splitIn: skSplit4SIFwd, splitNone: skSplit4IIFwd},
		8: {splitOut: skSplit8ISFwd, splitBoth: skSplit8SSFwd, splitIn: skSplit8SIFwd, splitNone: skSplit8IIFwd},
	}
	skSplitInv = [9][5]skPassFn{
		4: {splitOut: skSplit4ISInv, splitBoth: skSplit4SSInv, splitIn: skSplit4SIInv, splitNone: skSplit4IIInv},
		8: {splitOut: skSplit8ISInv, splitBoth: skSplit8SSInv, splitIn: skSplit8SIInv, splitNone: skSplit8IIInv},
	}
)

// splitPass reports whether a split kernel runs a pass of radix r with ido
// points per block: a radix with a split body, and ido a positive multiple of
// four, so every stream starts on a four-point block.
func splitPass(r, ido int) bool {
	return r >= 0 && r < len(skSplitFwd) && skSplitFwd[r][splitBoth] != nil && ido >= 4 && ido%4 == 0
}

// StockhamSplitModes returns, for the passes of one transform (pass k of
// radix r[k], ido[k], l1[k]), the layout mode of each: splitModes when
// UseStockhamSplit is on, all zeros (interleaved throughout) otherwise.
func StockhamSplitModes(r, ido, l1 []int) []uint8 {
	if intelSplit512On() {
		return intelSplitModes512(r, ido)
	}
	return splitModes(r, ido, UseStockhamSplit)
}

// splitModes gives every maximal run of consecutive passes that splitPass
// accepts its modes (splitOut, splitBoth…, splitIn; splitNone for a run of
// one) and every other pass 0. The final pass (ido == 1) is never split: it
// runs the interleaved kernel, of any radix, on the interleaved data the run
// before it wrote. A pure function, tested on any machine.
func splitModes(r, ido []int, on bool) []uint8 {
	modes := make([]uint8, len(r))
	if !on {
		return modes
	}
	for k := 0; k < len(r); {
		if !splitPass(r[k], ido[k]) {
			k++
			continue
		}
		e := k + 1
		for e < len(r) && splitPass(r[e], ido[e]) {
			e++
		}
		if e-k == 1 {
			modes[k] = splitNone
		} else {
			modes[k] = splitOut
			for m := k + 1; m < e-1; m++ {
				modes[m] = splitBoth
			}
			modes[e-1] = splitIn
		}
		k = e
	}
	return modes
}

// StockhamSplitTwiddles is StockhamTwiddles for a split pass: per group of
// four points i0 .. i0+3 and per j = 1 .. r-1, the real parts of twiddle j in
// the block order (i0, i0+2, i0+1, i0+3), then the imaginary parts, 64 bytes
// that the kernel loads as two registers. It is stored as complex128 for the
// plan's sake, two float64s per element, (r-1)·ido elements as interleaved.
func StockhamSplitTwiddles(r, ido, l1 int, root []complex128) (fwd, conj []complex128) {
	if intelSplit512On() {
		// The 512-bit layout, which StockhamSplitModes gave the passes.
		return intelSplitTwiddles512(r, ido, l1, root)
	}
	n := len(root)
	fwd = make([]complex128, (r-1)*ido)
	conj = make([]complex128, len(fwd))
	at := 0
	for i0 := 0; i0 < ido; i0 += 4 {
		for j := 1; j < r; j++ {
			var w [4]complex128
			for q, d := range [4]int{0, 2, 1, 3} {
				w[q] = root[(j*l1*(i0+d))%n]
			}
			fwd[at] = complex(real(w[0]), real(w[1]))
			fwd[at+1] = complex(real(w[2]), real(w[3]))
			fwd[at+2] = complex(imag(w[0]), imag(w[1]))
			fwd[at+3] = complex(imag(w[2]), imag(w[3]))
			conj[at], conj[at+1] = fwd[at], fwd[at+1]
			conj[at+2] = complex(-imag(w[0]), -imag(w[1]))
			conj[at+3] = complex(-imag(w[2]), -imag(w[3]))
			at += 4
		}
	}
	return fwd, conj
}

// StockhamPassLayout is StockhamPass with the pass's layout mode from
// StockhamSplitModes; mode 0 is StockhamPass itself. It reports false, as
// StockhamPass does, when the AVX2 kernels are off: every pass of a
// transform then runs in Go, interleaved. A mode the pass cannot run (an ido
// that is not a positive multiple of four, a radix without a split kernel, an
// unknown mode, l1 < 1) panics before any kernel runs: the kernels walk whole
// four-point groups and trust the layout.
func StockhamPassLayout(mode uint8, r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	if mode == 0 {
		return StockhamPass(r, ido, l1, cc, ch, tw, inverse, wide)
	}
	return splitPassLayout(mode, r, ido, l1, cc, ch, tw, inverse)
}

// splitPassLayout is StockhamPassLayout for a split mode.
func splitPassLayout(mode uint8, r, ido, l1 int, cc, ch, tw []complex128, inverse bool) bool {
	if !intelSplitModeOK(mode, r, ido, l1) {
		panic("kernels: StockhamPassLayout: no split kernel for this mode, radix and ido")
	}
	n := r * ido * l1
	_, _ = cc[n-1], ch[n-1] // the kernels trust these lengths
	_ = tw[(r-1)*ido-1]     // and this one
	if !UseStockhamAVX2 {
		return false
	}
	intelSplitKernel(mode, r, inverse)(&cc[0], &ch[0], &tw[0], &splitK[0][0], ido, l1)
	return true
}

//go:noescape
func skSplit4ISFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4SSFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4SIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4IIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4ISInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4SSInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4SIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit4IIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8ISFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SSFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8IIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8ISInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SSInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8SIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit8IIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)
