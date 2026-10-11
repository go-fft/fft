package kernels

// The two- and three-pass transforms on a kernel's own stack frame (Round
// 32): genZnTwoPassFile in asmgen/amd64/gen.go writes zntwopass_amd64.s.

// ZnTwoPassMax is the longest transform ZnTwoPass runs: znTwoPassAVX2's
// frame holds that many complex128 points (znTwoPassMax in gen.go).
const ZnTwoPassMax = 256

// ZnThreePassMax is the longest transform ZnThreePass runs: znThreePassAVX2's
// frame holds two buffers of that many points (znThreePassMax in gen.go).
const ZnThreePassMax = 512

// znAddrSlots is the size of znKernelAddrs' table: 21 interleaved pass
// kernels by radix, 21 final-pass kernels, then 2·9·5 split kernels by
// direction, radix and mode, the same for the 512-bit split kernels (Round
// 35), and 21 AVX-512 final-pass kernels by radix.
const znAddrSlots = 42 + 4*9*5 + 21

// Where znKernelAddrs puts the 512-bit split kernels and the AVX-512
// final-pass kernels (gen.go's genZnKernelAddrs).
const (
	csZnSplit512At = 42 + 2*9*5
	csZnLast512At  = 42 + 4*9*5
)

//go:noescape
func znTwoPassAVX2(cc, ch, tw *complex128, k0, k1 *float64, ido, l1 int, pass, last uintptr)

//go:noescape
func znThreePassAVX2(cc, ch, x, tw0, tw1 *complex128, k0, k1, k2 *float64, ido0, ido1, l1b, l1c int, p0, p1, p2 uintptr)

//go:noescape
func znKernelAddrs(t *[znAddrSlots]uintptr)

// znAddrs holds the kernels' ABI0 entry points (0 where there is none).
var znAddrs = func() (t [znAddrSlots]uintptr) {
	znKernelAddrs(&t)
	return t
}()

// znPass selects the kernel of a pass that is not the last one: radix r, ido
// points per block, layout mode (0 interleaved, else a split mode
// intelSplitModeOK accepts, 256- or 512-bit), returning its entry point (0
// when there is none), its constants table and the length of its twiddle
// table, as StockhamPassLayout would run it. wide is StockhamPass's (the
// AVX-512 kernels allowed): an interleaved pass it would run at 512 bits
// gets none, there being no such entry in the table. r is 2 to 20
// (ZnPlanFor checks it).
func znPass(r, ido int, mode uint8, inverse, wide bool) (addr uintptr, k *float64, twLen int) {
	if mode == 0 {
		if stockhamWidth(r, ido, true, wide) == 512 {
			return 0, nil, 0
		}
		k = &skFwd[0][0]
		if inverse {
			k = &skInv[0][0]
		}
		return znAddrs[r], k, skTwLen(r, ido)
	}
	if !intelSplitModeOK(mode, r, ido, 1) {
		return 0, nil, 0
	}
	d := 0
	if inverse {
		d = 1
	}
	at := 42
	if mode > splitNone {
		at, mode = csZnSplit512At, mode-intelSplit512Base
	}
	return znAddrs[at+(9*d+r)*5+int(mode)], &splitK[0][0], (r - 1) * ido
}

// znLast selects the final-pass kernel of radix r (entry point 0 when there
// is none) and its constants table, at the width StockhamPass would run it
// (wide as there); r is 2 to 20.
func znLast(r int, inverse, wide bool) (addr uintptr, k *float64) {
	if stockhamWidth(r, 1, true, wide) == 512 {
		k = &sk512Fwd[0][0]
		if inverse {
			k = &sk512Inv[0][0]
		}
		return znAddrs[csZnLast512At+r], k
	}
	k = &skFwd[0][0]
	if inverse {
		k = &skInv[0][0]
	}
	return znAddrs[21+r], k
}

// znSplitOut and znSplitIn report whether a pass of layout mode writes or
// reads the split layout, at either width.
func znSplitOut(mode uint8) bool {
	mode = csZnBase(mode)
	return mode == splitOut || mode == splitBoth
}

func znSplitIn(mode uint8) bool {
	mode = csZnBase(mode)
	return mode == splitIn || mode == splitBoth
}

// csZnBase is a split mode without intelSplit512Base.
func csZnBase(mode uint8) uint8 {
	if mode > splitNone {
		return mode - intelSplit512Base
	}
	return mode
}

// ZnPlan is a Stockham transform of two or three passes bound to the kernels
// and twiddle tables that run it with the buffers between its passes in a
// kernel's own stack frame (znTwoPassAVX2, znThreePassAVX2): no scratch
// buffer from the caller, and no selection left for each call. ZnPlanFor
// builds it.
type ZnPlan struct {
	n, ido0, ido1, l1b, l1c int
	three                   bool
	// wide: the final pass runs an AVX-512 kernel, which StockhamPass
	// takes only while UseStockhamAVX512 is on.
	wide bool
	addr [2][3]uintptr     // entry points by direction (0 forward), per pass
	k    [2][3]*float64    // constants tables, likewise
	tw   [2][2]*complex128 // twiddles of passes 0 and 1, by direction
}

// ZnPlanFor returns the ZnPlan of the passes of radices r (two or three) and
// layout modes mode (one per pass, the last 0), whose first two passes take
// the twiddle tables tw0 and tw1 (forward) and tw0c and tw1c (conjugate) as
// StockhamTwiddles or StockhamSplitTwiddlesFor build them (tw1 unused for two
// passes), or nil when the kernels cannot run it: the AVX2 kernels off, more
// points than ZnTwoPassMax or ZnThreePassMax, a radix without a kernel, a
// table too short, or layouts that do not chain (pass 0 reads interleaved
// data, each pass reads the layout the one before writes, the pass before the
// final one writes interleaved data). It is ZnPlanForWide without the
// AVX-512 kernels.
//
// Run calls the kernels StockhamPassLayout and StockhamPass would call, with
// the same arguments, so the result is the same bits.
func ZnPlanFor(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128) *ZnPlan {
	return ZnPlanForWide(r, mode, tw0, tw0c, tw1, tw1c, false)
}

// ZnPlanForWide is ZnPlanFor for a transform whose passes StockhamPass runs
// with wide (Round 35, Cascade Lake): its 512-bit split passes (the modes
// of intelSplitModes512) and an AVX-512 final pass run in the frame too. An
// interleaved pass other than the last that StockhamPass would run at 512
// bits has no entry point: such a plan gets nil.
func ZnPlanForWide(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128, wide bool) *ZnPlan {
	s := len(r)
	if !UseStockhamAVX2 || (s != 2 && s != 3) || len(mode) != s || mode[s-1] != 0 || znSplitIn(mode[0]) || znSplitOut(mode[s-2]) {
		return nil
	}
	n := 1
	for _, x := range r {
		if x < 2 || x > 20 {
			return nil
		}
		n *= x
	}
	wide = wide && UseStockhamAVX512
	z := &ZnPlan{n: n, three: s == 3, ido0: n / r[0], l1b: r[0], wide: wide && stockhamWidth(r[s-1], 1, true, true) == 512}
	if (!z.three && n > ZnTwoPassMax) || (z.three && (n > ZnThreePassMax || znSplitOut(mode[0]) != znSplitIn(mode[1]))) {
		return nil
	}
	// A chain joins two layouts of one width: 256- and 512-bit split data
	// are not laid out alike.
	if z.three && znSplitOut(mode[0]) && (mode[0] > splitNone) != (mode[1] > splitNone) {
		return nil
	}
	if z.three {
		z.ido1, z.l1c = r[2], r[0]*r[1]
	}
	tws := [2][2][]complex128{{tw0, tw1}, {tw0c, tw1c}}
	for d := range 2 {
		l1 := 1
		for i := 0; i < s-1; i++ {
			a, k, tl := znPass(r[i], n/(l1*r[i]), mode[i], d == 1, wide)
			if a == 0 || len(tws[d][i]) < tl {
				return nil
			}
			z.addr[d][i], z.k[d][i], z.tw[d][i] = a, k, &tws[d][i][0]
			l1 *= r[i]
		}
		a, k := znLast(r[s-1], d == 1, wide)
		if a == 0 {
			return nil
		}
		z.addr[d][s-1], z.k[d][s-1] = a, k
	}
	return z
}

// Run transforms cc into ch (they may be the same slice) and reports true,
// or reports false (and does nothing) when the kernels it was built with
// have been turned off since: the AVX2 ones, or the AVX-512 final pass's.
func (z *ZnPlan) Run(cc, ch []complex128, inverse bool) bool {
	if !UseStockhamAVX2 || z.wide && !UseStockhamAVX512 {
		return false
	}
	_, _ = cc[z.n-1], ch[z.n-1] // the kernels trust these lengths
	d := 0
	if inverse {
		d = 1
	}
	a, k, tw := &z.addr[d], &z.k[d], &z.tw[d]
	if !z.three {
		znTwoPassAVX2(&cc[0], &ch[0], tw[0], k[0], k[1], z.ido0, z.l1b, a[0], a[1])
		return true
	}
	// Pass 0 writes ch, unless ch is cc, whose input it is still reading:
	// then a second frame buffer (x nil).
	x := &ch[0]
	if x == &cc[0] {
		x = nil
	}
	znThreePassAVX2(&cc[0], &ch[0], x, tw[0], tw[1], k[0], k[1], k[2], z.ido0, z.ido1, z.l1b, z.l1c, a[0], a[1], a[2])
	return true
}

// Wide reports whether the plan's final pass runs an AVX-512 kernel.
func (z *ZnPlan) Wide() bool { return z.wide }
