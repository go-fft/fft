package kernels

// The entry points the fft package calls for the passes of an FMA plan
// (Round 41). Each mirrors its default counterpart (StockhamPass,
// StockhamPassLayout, StockhamBatchPass, StockhamStrided) check for check and
// selects the FMA twin of the kernel that one would run, at the same width:
// the default functions are left as they were, so the default plans' path is
// unchanged. They report false exactly when their counterparts do, when the
// AVX2 kernels are off, and the fft package then runs the Go reference pass
// with fused products (passScalarFMA): a plan keeps one arithmetic whichever
// path runs it. UseFMA is not consulted here: it decided whether the plan was
// built for FMA at all.

// StockhamPassFMA is StockhamPass for a pass of an FMA plan. A final pass
// (ido == 1) has no twiddles: it runs the FMA twin of its kernel where the
// butterfly has a product to fuse (stockhamLastFMAKernel), the default
// kernel otherwise.
func StockhamPassFMA(r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	if r == 32 {
		return stockhamPass32FMA(ido, l1, cc, ch, inverse, wide)
	}
	w := stockhamWidth(r, ido, UseStockhamAVX2, UseStockhamAVX512 && wide)
	if w == 0 {
		return false
	}
	if ido < 1 || l1 < 1 {
		panic("kernels: StockhamPass: ido and l1 must be at least 1")
	}
	n := r * ido * l1
	_, _ = cc[n-1], ch[n-1] // the kernels trust these lengths
	k, _, _ := stockhamKernels(r, w, inverse)
	if ido == 1 {
		stockhamLastFMAKernel(r, w)(&cc[0], &ch[0], k, l1)
		return true
	}
	_ = tw[skTwLen(r, ido)-1]
	stockhamPassFMAKernel(r, w)(&cc[0], &ch[0], &tw[0], k, ido, l1)
	return true
}

// stockhamPassFMAKernel returns the FMA twin of the radix-r pass kernel of a
// width stockhamWidth chose; a pure function, tested for both widths on any
// machine.
func stockhamPassFMAKernel(r, w int) skPassFn {
	if w == 512 {
		return skPass512FMA[r]
	}
	return skPass256FMA[r]
}

// stockhamLastFMAKernel returns the final-pass kernel of an FMA plan for
// radix r at a width stockhamWidth chose: the FMA twin where there is one,
// the default kernel otherwise. A pure function, tested for both widths on
// any machine.
func stockhamLastFMAKernel(r, w int) skLastFn {
	fused, plain := skLast256FMA[r], skLast256[r]
	if w == 512 {
		fused, plain = skLast512FMA[r], skLast512[r]
	}
	if fused != nil {
		return fused
	}
	return plain
}

// StockhamPassLayoutFMA is StockhamPassLayout for a pass of an FMA plan.
func StockhamPassLayoutFMA(mode uint8, r, ido, l1 int, cc, ch, tw []complex128, inverse, wide bool) bool {
	if mode == 0 {
		return StockhamPassFMA(r, ido, l1, cc, ch, tw, inverse, wide)
	}
	if !intelSplitModeOK(mode, r, ido, l1) {
		panic("kernels: StockhamPassLayout: no split kernel for this mode, radix and ido")
	}
	n := r * ido * l1
	_, _ = cc[n-1], ch[n-1] // the kernels trust these lengths
	_ = tw[(r-1)*ido-1]     // and this one
	if !UseStockhamAVX2 {
		return false
	}
	intelSplitKernelFMA(mode, r, ido, l1, inverse)(&cc[0], &ch[0], &tw[0], &splitK[0][0], ido, l1)
	return true
}

// intelSplitKernelFMA is intelSplitKernel's FMA twin.
func intelSplitKernelFMA(mode uint8, r, ido, l1 int, inverse bool) skPassFn {
	if r == 32 {
		return r32SplitKernel(mode, ido, l1, inverse)
	}
	if mode > splitNone {
		if inverse {
			return intelSplit512InvFMA[r][mode-intelSplit512Base]
		}
		return intelSplit512FwdFMA[r][mode-intelSplit512Base]
	}
	if inverse {
		return skSplitInvFMA[r][mode]
	}
	return skSplitFwdFMA[r][mode]
}

// StockhamBatchPassFMA is StockhamBatchPass for a pass of an FMA plan: the
// FMA twins of the batched kernels, which fuse radix 8's √2/2 products in
// the final pass too.
func StockhamBatchPassFMA(r, ido, l1 int, cc, ch, tw []complex128, w, sIn, sOut int, inverse bool) bool {
	if w < 1 || sIn < w || sOut < w || ido < 1 || l1 < 1 {
		panic("kernels: StockhamBatchPass: need w >= 1, sIn and sOut >= w, ido and l1 >= 1")
	}
	if !UseStockhamBatchAVX2 || r >= len(skBatch256FMA) || skBatch256FMA[r] == nil {
		return false
	}
	n := r * ido * l1
	_, _ = cc[(n-1)*sIn+w-1], ch[(n-1)*sOut+w-1] // the kernels trust these lengths
	t := &skFwd[0][0]
	if inverse {
		t = &skInv[0][0]
	}
	tp := (*complex128)(nil)
	if ido > 1 {
		_ = tw[(ido-1)*(r-1)-1]
		tp = &tw[0]
	}
	skBatch256FMA[r](&cc[0], &ch[0], tp, t, ido, l1, w/2, w&1, 16*ido*sIn, 16*l1*ido*sOut, 16*(sIn-w), 16*(sOut-w))
	return true
}

// StockhamStridedFMA is StockhamStrided for a pass of an FMA plan.
func StockhamStridedFMA(r int, cc, ch, tw []complex128, cnt, nb, sin, bin, sout, bout, btw int, first0, firstRest, inverse, wide bool) bool {
	if sin < 0 || bin < 0 || sout < 0 || bout < 0 || btw < 0 {
		panic("kernels: StockhamStrided: strides must be >= 0")
	}
	if !UseStockhamAVX2 || nb < 1 || cnt < 4 || cnt%4 != 0 || skStrided256FMA[r] == nil {
		return false
	}
	_, _ = cc[(nb-1)*bin+(r-1)*sin+cnt-1], ch[(nb-1)*bout+(r-1)*sout+cnt-1]
	_ = tw[(nb-1)*btw+(r-1)*cnt-1]
	w := stockhamWidth(r, cnt, true, UseStockhamAVX512 && wide)
	k, _, _ := stockhamKernels(r, w, inverse)
	stockhamStridedFMAKernel(r, w)(&cc[0], &ch[0], &tw[0], k, cnt, nb, sin, sout,
		16*(bin-cnt), 16*(bout-cnt), 16*btw, flag(first0), flag(firstRest))
	return true
}

// stockhamStridedFMAKernel is stockhamStridedKernel's FMA twin.
func stockhamStridedFMAKernel(r, w int) skStridedFn {
	if w == 512 {
		return skStrided512FMA[r]
	}
	return skStrided256FMA[r]
}

// StockhamLastRunFMA is StockhamLastRun for the final pass of an FMA plan:
// the FMA twin of the kernel for radix 8, the default kernel otherwise.
func StockhamLastRunFMA(r int, cc, ch []complex128, os, runs, run, gap int, inverse, wide bool) bool {
	if os < 0 || gap < 0 {
		panic("kernels: StockhamLastRun: os and gap must be >= 0")
	}
	if !UseStockhamAVX2 || runs < 1 || run < 4 || run%4 != 0 || skLastRun256[r] == nil {
		return false
	}
	_, _ = cc[runs*run*r-1], ch[(runs-1)*gap+run-1+(r-1)*os] // the kernels trust these lengths
	w := stockhamWidth(r, 1, true, UseStockhamAVX512 && wide)
	k, _, _ := stockhamKernels(r, w, inverse)
	stockhamLastRunFMAKernel(r, w)(&cc[0], &ch[0], k, os, runs, run, gap)
	return true
}

// stockhamLastRunFMAKernel is stockhamLastRunKernel for an FMA plan: the
// FMA twin where there is one, the default kernel otherwise.
func stockhamLastRunFMAKernel(r, w int) skLastRunFn {
	fused, plain := skLastRun256FMA[r], skLastRun256[r]
	if w == 512 {
		fused, plain = skLastRun512FMA[r], skLastRun512[r]
	}
	if fused != nil {
		return fused
	}
	return plain
}
