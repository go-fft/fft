package kernels

// The split layout at 512 bits (genIntelSplit512, stockhamsplit512_amd64.s;
// Round 28): eight points 8q .. 8q+7 occupy the 128 bytes they occupy
// interleaved, as their real parts in the order (0, 4, 1, 5, 2, 6, 3, 7),
// then their imaginary parts in the same order, the order VUNPCKLPD and
// VUNPCKHPD give two interleaved ZMM loads. The modes are splitOut ..
// splitNone plus intelSplit512Base, so a plan's modes say which width wrote
// its data; a run of passes is one width throughout.

// intelSplit512Base is added to a split mode for the 512-bit kernels.
const intelSplit512Base = splitNone

// UseStockhamSplit512 reports whether plans built from now on keep the data
// of a power of two split at 512 bits between passes, instead of the 256-bit
// layout (UseStockhamSplit) or none; intelSplit512Default decides. A variable
// so the tests can turn it on and off; it takes effect only where the CPU
// runs AVX-512 (intelHasAVX512).
var UseStockhamSplit512 = intelSplit512Default(useAVX2, supportsAVX512F(), IntelCPU)

// intelHasAVX512 is the hardware probe the 512-bit split kernels need; a
// variable so the tests can take the refusal on a machine that has it.
var intelHasAVX512 = supportsAVX512F()

// intelSplit512Default turns the 512-bit split layout on where it was
// measured (Round 28, Cascade Lake): AVX-512 on an Intel CPU. A pure
// function, tested on any machine.
func intelSplit512Default(avx2, avx512, intel bool) bool { return avx2 && avx512 && intel }

// intelSplit512On reports whether StockhamSplitModes and
// StockhamSplitTwiddles give the 512-bit layout.
func intelSplit512On() bool { return UseStockhamSplit512 && intelHasAVX512 }

// The 512-bit split kernels by radix and mode (without intelSplit512Base),
// per direction.
var (
	intelSplit512Fwd = [9][5]skPassFn{
		4: {splitOut: intelSplit512R4ISFwd, splitBoth: intelSplit512R4SSFwd, splitIn: intelSplit512R4SIFwd, splitNone: intelSplit512R4IIFwd},
		8: {splitOut: intelSplit512R8ISFwd, splitBoth: intelSplit512R8SSFwd, splitIn: intelSplit512R8SIFwd, splitNone: intelSplit512R8IIFwd},
	}
	intelSplit512Inv = [9][5]skPassFn{
		4: {splitOut: intelSplit512R4ISInv, splitBoth: intelSplit512R4SSInv, splitIn: intelSplit512R4SIInv, splitNone: intelSplit512R4IIInv},
		8: {splitOut: intelSplit512R8ISInv, splitBoth: intelSplit512R8SSInv, splitIn: intelSplit512R8SIInv, splitNone: intelSplit512R8IIInv},
	}
)

// intelSplitPass512 reports whether a 512-bit split kernel runs a pass of
// radix r with ido points per block: radix 4 or 8, and ido a positive
// multiple of eight, so every stream starts on an eight-point block.
func intelSplitPass512(r, ido int) bool {
	return (r == 4 || r == 8) && ido >= 8 && ido%8 == 0
}

// intelSplitModes512 is splitModes at 512 bits, for a power of two of at
// least intelSplit512MinN points (the AVX-512 kernels run powers of two only,
// see the fft package's wide512); every other length stays interleaved. n is
// r[0]·ido[0], the first pass having l1 = 1. A pure function, tested on any
// machine.
func intelSplitModes512(r, ido []int) []uint8 {
	modes := make([]uint8, len(r))
	if len(r) == 0 {
		return modes
	}
	if n := r[0] * ido[0]; n < intelSplit512MinN || n&(n-1) != 0 {
		return modes
	}
	for k := 0; k < len(r); {
		if !intelSplitPass512(r[k], ido[k]) {
			k++
			continue
		}
		e := k + 1
		for e < len(r) && intelSplitPass512(r[e], ido[e]) {
			e++
		}
		if e-k == 1 {
			modes[k] = intelSplit512Base + splitNone
		} else {
			modes[k] = intelSplit512Base + splitOut
			for m := k + 1; m < e-1; m++ {
				modes[m] = intelSplit512Base + splitBoth
			}
			modes[e-1] = intelSplit512Base + splitIn
		}
		k = e
	}
	return modes
}

// intelSplit512MinN is the smallest power of two whose passes run split at
// 512 bits.
const intelSplit512MinN = 256

// intelSplitTwiddles512 is StockhamSplitTwiddles for a 512-bit split pass
// (ido a multiple of eight): per group of eight points i0 .. i0+7 and per j =
// 1 .. r-1, the real parts of twiddle j in the block order (i0, i0+4, i0+1,
// i0+5, i0+2, i0+6, i0+3, i0+7), then the imaginary parts, 128 bytes.
func intelSplitTwiddles512(r, ido, l1 int, root []complex128) (fwd, conj []complex128) {
	n := len(root)
	fwd = make([]complex128, (r-1)*ido)
	conj = make([]complex128, len(fwd))
	at := 0
	for i0 := 0; i0+8 <= ido; i0 += 8 {
		for j := 1; j < r; j++ {
			var w [8]complex128
			for q, d := range [8]int{0, 4, 1, 5, 2, 6, 3, 7} {
				w[q] = root[(j*l1*(i0+d))%n]
			}
			for q := 0; q < 4; q++ {
				fwd[at+q] = complex(real(w[2*q]), real(w[2*q+1]))
				fwd[at+4+q] = complex(imag(w[2*q]), imag(w[2*q+1]))
				conj[at+q] = fwd[at+q]
				conj[at+4+q] = complex(-imag(w[2*q]), -imag(w[2*q+1]))
			}
			at += 8
		}
	}
	return fwd, conj
}

// intelSplitModeOK reports whether a split kernel can run a pass of radix r
// with ido and l1 in this mode: a 256-bit mode (splitOut .. splitNone) as
// splitPass says, a 512-bit one as intelSplitPass512 says and only on a CPU
// with AVX-512. A pure function of its arguments and intelHasAVX512.
func intelSplitModeOK(mode uint8, r, ido, l1 int) bool {
	switch {
	case l1 < 1 || mode == 0:
		return false
	case mode <= splitNone:
		return splitPass(r, ido)
	case mode <= intelSplit512Base+splitNone:
		return intelHasAVX512 && intelSplitPass512(r, ido)
	}
	return false
}

// intelSplitKernel returns the kernel of a mode intelSplitModeOK accepted.
func intelSplitKernel(mode uint8, r int, inverse bool) skPassFn {
	if mode > splitNone {
		if inverse {
			return intelSplit512Inv[r][mode-intelSplit512Base]
		}
		return intelSplit512Fwd[r][mode-intelSplit512Base]
	}
	if inverse {
		return skSplitInv[r][mode]
	}
	return skSplitFwd[r][mode]
}

//go:noescape
func intelSplit512R4ISFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4SSFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4SIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4IIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4ISInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4SSInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4SIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R4IIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8ISFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8SSFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8SIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8IIFwd(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8ISInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8SSInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8SIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R8IIInv(cc, ch, tw *complex128, k *float64, ido, l1 int)
