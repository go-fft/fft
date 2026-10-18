package kernels

// The radix-32 kernels of the FMA plans (Round 42, r32fma_amd64.s, go-asmgen's
// SplitR32 and LastR32): a split-layout pass at 256 and 512 bits and an
// interleaved final pass at 256 bits. No default plan has a radix-32 pass,
// so these kernels have no default twins: the fft package builds radix 32
// into FMA plans only, and its reference is the Go pass with bfly32FMA.

// The split-layout radix-32 kernels by direction (0 forward) and mode.
var (
	skSplit32FMA = [2][5]skPassFn{
		{splitOut: skSplit32ISFwdFMA, splitBoth: skSplit32SSFwdFMA, splitIn: skSplit32SIFwdFMA, splitNone: skSplit32IIFwdFMA},
		{splitOut: skSplit32ISInvFMA, splitBoth: skSplit32SSInvFMA, splitIn: skSplit32SIInvFMA, splitNone: skSplit32IIInvFMA},
	}
	intelSplit512R32FMA = [2][5]skPassFn{
		{splitOut: intelSplit512R32ISFwdFMA, splitBoth: intelSplit512R32SSFwdFMA, splitIn: intelSplit512R32SIFwdFMA, splitNone: intelSplit512R32IIFwdFMA},
		{splitOut: intelSplit512R32ISInvFMA, splitBoth: intelSplit512R32SSInvFMA, splitIn: intelSplit512R32SIInvFMA, splitNone: intelSplit512R32IIInvFMA},
	}
)

// r32SplitKernel returns the radix-32 split kernel of a mode
// intelSplitModeOK accepted (a 512-bit mode above splitNone).
func r32SplitKernel(mode uint8, inverse bool) skPassFn {
	d := 0
	if inverse {
		d = 1
	}
	if mode > splitNone {
		return intelSplit512R32FMA[d][mode-intelSplit512Base]
	}
	return skSplit32FMA[d][mode]
}

// stockhamPass32FMA runs a radix-32 pass of an FMA plan that is not split:
// the final pass (ido == 1) on skLast32AVX512FMA where StockhamPass would
// run the AVX-512 kernels (wide), skLast32AVX2FMA elsewhere; any other
// reports false
// (there is no interleaved radix-32 pass kernel), and the fft package runs
// its Go pass. So does a final pass in place (cc and ch the same: the one
// pass of a 32-point transform run in place), since the kernel keeps its
// values between stages in its output.
func stockhamPass32FMA(ido, l1 int, cc, ch []complex128, inverse, wide bool) bool {
	if !UseStockhamAVX2 || ido != 1 || &cc[0] == &ch[0] {
		return false
	}
	if l1 < 1 {
		panic("kernels: StockhamPass: ido and l1 must be at least 1")
	}
	_, _ = cc[32*l1-1], ch[32*l1-1] // the kernel trusts these lengths
	if UseStockhamAVX512 && wide {
		k := &sk512Fwd[0][0]
		if inverse {
			k = &sk512Inv[0][0]
		}
		skLast32AVX512FMA(&cc[0], &ch[0], k, l1)
		return true
	}
	k := &skFwd[0][0]
	if inverse {
		k = &skInv[0][0]
	}
	skLast32AVX2FMA(&cc[0], &ch[0], k, l1)
	return true
}

//go:noescape
func skSplit32ISFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SSFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SIFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32IIFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32ISInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SSInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SIInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32IIInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32ISFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SSFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SIFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32IIFwdFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32ISInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SSInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SIInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32IIInvFMA(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skLast32AVX2FMA(cc, ch *complex128, k *float64, l1 int)

//go:noescape
func skLast32AVX512FMA(cc, ch *complex128, k *float64, l1 int)
