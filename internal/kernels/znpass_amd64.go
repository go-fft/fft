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
// direction, radix and mode.
const znAddrSlots = 42 + 2*9*5

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
// points per block, layout mode (0 interleaved, else a 256-bit split mode
// splitPass accepts), returning its entry point (0 when there is none), its
// constants table and the length of its twiddle table, as StockhamPassLayout
// would run it. r is 2 to 20 (ZnPlanFor checks it).
func znPass(r, ido int, mode uint8, inverse bool) (addr uintptr, k *float64, twLen int) {
	if mode == 0 {
		k = &skFwd[0][0]
		if inverse {
			k = &skInv[0][0]
		}
		return znAddrs[r], k, skTwLen(r, ido)
	}
	if mode > splitNone || !splitPass(r, ido) {
		return 0, nil, 0
	}
	d := 0
	if inverse {
		d = 1
	}
	return znAddrs[42+(9*d+r)*5+int(mode)], &splitK[0][0], (r - 1) * ido
}

// znLast selects the final-pass kernel of radix r (entry point 0 when there
// is none) and its constants table; r is 2 to 20.
func znLast(r int, inverse bool) (addr uintptr, k *float64) {
	k = &skFwd[0][0]
	if inverse {
		k = &skInv[0][0]
	}
	return znAddrs[21+r], k
}

// znSplitOut and znSplitIn report whether a pass of layout mode writes or
// reads the split layout.
func znSplitOut(mode uint8) bool { return mode == splitOut || mode == splitBoth }
func znSplitIn(mode uint8) bool  { return mode == splitIn || mode == splitBoth }

// ZnPlan is a Stockham transform of two or three passes bound to the kernels
// and twiddle tables that run it with the buffers between its passes in a
// kernel's own stack frame (znTwoPassAVX2, znThreePassAVX2): no scratch
// buffer from the caller, and no selection left for each call. ZnPlanFor
// builds it.
type ZnPlan struct {
	n, ido0, ido1, l1b, l1c int
	three                   bool
	addr                    [2][3]uintptr     // entry points by direction (0 forward), per pass
	k                       [2][3]*float64    // constants tables, likewise
	tw                      [2][2]*complex128 // twiddles of passes 0 and 1, by direction
}

// ZnPlanFor returns the ZnPlan of the passes of radices r (two or three) and
// layout modes mode (one per pass, the last 0), whose first two passes take
// the twiddle tables tw0 and tw1 (forward) and tw0c and tw1c (conjugate) as
// StockhamTwiddles or StockhamSplitTwiddles build them (tw1 unused for two
// passes), or nil when the kernels cannot run it: the AVX2 kernels off, more
// points than ZnTwoPassMax or ZnThreePassMax, a radix without a kernel, a
// table too short, or layouts that do not chain (pass 0 reads interleaved
// data, each pass reads the layout the one before writes, the pass before the
// final one writes interleaved data).
//
// Run calls the kernels StockhamPassLayout and StockhamPass would call, with
// the same arguments, so the result is the same bits.
func ZnPlanFor(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128) *ZnPlan {
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
	z := &ZnPlan{n: n, three: s == 3, ido0: n / r[0], l1b: r[0]}
	if (!z.three && n > ZnTwoPassMax) || (z.three && (n > ZnThreePassMax || znSplitOut(mode[0]) != znSplitIn(mode[1]))) {
		return nil
	}
	if z.three {
		z.ido1, z.l1c = r[2], r[0]*r[1]
	}
	tws := [2][2][]complex128{{tw0, tw1}, {tw0c, tw1c}}
	for d := range 2 {
		l1 := 1
		for i := 0; i < s-1; i++ {
			a, k, tl := znPass(r[i], n/(l1*r[i]), mode[i], d == 1)
			if a == 0 || len(tws[d][i]) < tl {
				return nil
			}
			z.addr[d][i], z.k[d][i], z.tw[d][i] = a, k, &tws[d][i][0]
			l1 *= r[i]
		}
		a, k := znLast(r[s-1], d == 1)
		if a == 0 {
			return nil
		}
		z.addr[d][s-1], z.k[d][s-1] = a, k
	}
	return z
}

// Run transforms cc into ch (they may be the same slice) and reports true,
// or reports false (and does nothing) when the AVX2 kernels have been turned
// off since z was built.
func (z *ZnPlan) Run(cc, ch []complex128, inverse bool) bool {
	if !UseStockhamAVX2 {
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
