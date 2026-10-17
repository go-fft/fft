package kernels

import "testing"

// Round 35: the frame kernels run Cascade Lake's powers of two from 256
// points, whose passes are 512-bit split ones and an AVX-512 final pass.

// TestCsZnWideMatchesPasses: a ZnPlanForWide plan gives, bit for bit, what
// its passes give run one after the other into buffers with wide set
// (StockhamPassLayout, then StockhamPass), for every two- and three-pass
// shape of radices 2, 4 and 8 up to the frame's limits, every chain of
// 512-bit layouts the kernels accept, both directions, in place and not.
// It needs a CPU with AVX-512; TestCsZnWideSelects covers the selection on
// any machine.
func TestCsZnWideMatchesPasses(t *testing.T) {
	if !UseStockhamAVX2 || !UseStockhamAVX512 || !intelHasAVX512 {
		t.Skip("no AVX-512: TestCsZnWideSelects covers the selection")
	}
	const b = intelSplit512Base
	chains := map[int][][]uint8{
		2: {{0, 0}, {b + splitNone, 0}},
		3: {{0, 0, 0}, {b + splitNone, 0, 0}, {0, b + splitNone, 0}, {b + splitNone, b + splitNone, 0}, {b + splitOut, b + splitIn, 0}},
	}
	radices := []int{2, 4, 8}
	var shapes [][]int
	for _, x := range radices {
		for _, y := range radices {
			if x*y <= ZnTwoPassMax {
				shapes = append(shapes, []int{x, y})
			}
			for _, z := range radices {
				if x*y*z <= ZnThreePassMax {
					shapes = append(shapes, []int{x, y, z})
				}
			}
		}
	}
	shapes = append(shapes, []int{8, 32}, []int{16, 16}, []int{4, 8, 8}, []int{8, 8, 8}, []int{8, 4, 16})
	built, wideSplit := 0, 0
	for _, r := range shapes {
		s := len(r)
		n := 1
		for _, x := range r {
			n *= x
		}
		root := znRoot(n)
		for _, m := range chains[s] {
			ok := true
			var tws [2][2][]complex128 // [direction][pass]
			l1 := 1
			for i := 0; i < s-1; i++ {
				ido := n / (l1 * r[i])
				if m[i] != 0 {
					if !intelSplitPass512(r[i], ido) {
						ok = false
						break
					}
					tws[0][i], tws[1][i] = StockhamSplitTwiddlesFor(m[i], r[i], ido, l1, root)
				} else {
					tws[0][i], tws[1][i] = StockhamTwiddles(r[i], ido, l1, root)
				}
				l1 *= r[i]
			}
			if !ok {
				continue
			}
			z := ZnPlanForWide(r, m, tws[0][0], tws[1][0], tws[0][1], tws[1][1], true)
			if z == nil {
				continue // an interleaved pass that would run at 512 bits
			}
			built++
			if m[0] != 0 || m[1] != 0 {
				wideSplit++
			}
			for d, inverse := range []bool{false, true} {
				for si, src := range znSignals(n) {
					want := make([]complex128, n)
					in, l1 := src, 1
					for i := 0; i < s-1; i++ {
						out := make([]complex128, n)
						if !StockhamPassLayout(m[i], r[i], n/(l1*r[i]), l1, in, out, tws[d][i], inverse, true) {
							t.Fatalf("%v: no pass kernel", r)
						}
						in, l1 = out, l1*r[i]
					}
					if !StockhamPass(r[s-1], 1, l1, in, want, nil, inverse, true) {
						t.Fatalf("%v: no final kernel", r)
					}
					got := make([]complex128, n)
					inPlace := append([]complex128(nil), src...)
					if !z.Run(src, got, inverse) || !z.Run(inPlace, inPlace, inverse) {
						t.Fatalf("%v %v: Run refused", r, m)
					}
					for i := range want {
						if !znSame(got[i], want[i]) || !znSame(inPlace[i], want[i]) {
							t.Fatalf("%v modes %v inverse %v signal %d: [%d] = %v (in place %v), want %v",
								r, m, inverse, si, i, got[i], inPlace[i], want[i])
						}
					}
				}
			}
		}
	}
	if built < 20 || wideSplit < 6 {
		t.Fatalf("only %d plans built, %d with 512-bit split passes", built, wideSplit)
	}
}

// TestCsZnWideSelects: on any machine, the entry points and constants
// ZnPlanForWide binds are the ones StockhamPassLayout and StockhamPass would
// run, the plans it must refuse are refused, and Run reports false (running
// nothing) once the AVX-512 kernels its final pass needs are off.
func TestCsZnWideSelects(t *testing.T) {
	defer func(a, b, c bool) { UseStockhamAVX2, UseStockhamAVX512, intelHasAVX512 = a, b, c }(UseStockhamAVX2, UseStockhamAVX512, intelHasAVX512)
	UseStockhamAVX2, UseStockhamAVX512, intelHasAVX512 = true, true, true
	const b = intelSplit512Base
	for _, c := range []struct {
		r, ido int
		mode   uint8
		inv    bool
		slot   int
	}{
		{8, 64, b + splitOut, false, csZnSplit512At + 8*5 + splitOut},
		{4, 64, b + splitIn, true, csZnSplit512At + (9+4)*5 + splitIn},
		{8, 8, b + splitNone, false, csZnSplit512At + 8*5 + splitNone},
		{8, 64, splitOut, false, 42 + 8*5 + splitOut},
	} {
		a, k, tl := znPass(c.r, c.ido, c.mode, c.inv, true, false)
		if a == 0 || a != znAddrs[c.slot] || k != &splitK[0][0] || tl != (c.r-1)*c.ido {
			t.Errorf("znPass(%d, %d, %d): %#x %v %d, want slot %d", c.r, c.ido, c.mode, a, k == &splitK[0][0], tl, c.slot)
		}
	}
	// No 512-bit interleaved pass in the table; a 512-bit split pass needs
	// AVX-512 and ido a multiple of eight.
	if a, _, _ := znPass(8, 64, 0, false, true, false); a != 0 {
		t.Error("an interleaved 512-bit pass got an entry point")
	}
	if a, _, _ := znPass(8, 2, 0, false, true, false); a != znAddrs[8] {
		t.Error("ido 2 does not run at 512 bits: want the AVX2 kernel")
	}
	if a, _, _ := znPass(8, 12, b+splitOut, false, true, false); a != 0 {
		t.Error("512-bit split with ido 12")
	}
	for _, r := range []int{2, 4, 8} {
		for d, inv := range []bool{false, true} {
			want := &sk512Fwd[0][0]
			if inv {
				want = &sk512Inv[0][0]
			}
			if a, k := znLast(r, inv, true); a == 0 || a != znAddrs[csZnLast512At+r] || k != want {
				t.Errorf("znLast(%d, %v, wide): %#x", r, inv, a)
			}
			_ = d
		}
		if a, k := znLast(r, false, false); a != znAddrs[21+r] || k != &skFwd[0][0] {
			t.Errorf("znLast(%d) narrow", r)
		}
	}
	if a, _ := znLast(16, false, true); a != znAddrs[21+16] {
		t.Error("radix 16 has no AVX-512 final pass: want the AVX2 one")
	}
	intelHasAVX512 = false
	if a, _, _ := znPass(8, 64, b+splitOut, false, true, false); a != 0 {
		t.Error("512-bit split without AVX-512")
	}
	intelHasAVX512 = true
	// A plan whose final pass is AVX-512 is built only with wide, and runs
	// only while UseStockhamAVX512 is on.
	tw := make([]complex128, 4096)
	m := []uint8{b + splitOut, b + splitIn, 0}
	z := ZnPlanForWide([]int{4, 8, 8}, m, tw, tw, tw, tw, true)
	if z == nil || !z.wide || z.addr[0][2] != znAddrs[csZnLast512At+8] || z.addr[1][0] != znAddrs[csZnSplit512At+(9+4)*5+splitOut] {
		t.Fatalf("4·8·8 wide: %+v", z)
	}
	if nz := ZnPlanForWide([]int{4, 8, 8}, m, tw, tw, tw, tw, false); nz == nil || nz.wide || nz.addr[0][2] != znAddrs[21+8] {
		t.Error("4·8·8 narrow: want the AVX2 final pass")
	}
	if ZnPlanForWide([]int{4, 8, 8}, []uint8{0, 0, 0}, tw, tw, tw, tw, true) != nil {
		t.Error("4·8·8 interleaved, wide: its first pass would run at 512 bits")
	}
	for _, mm := range [][]uint8{{b + splitOut, splitIn, 0}, {splitOut, b + splitIn, 0}} {
		if ZnPlanForWide([]int{4, 8, 8}, mm, tw, tw, tw, tw, true) != nil {
			t.Errorf("%v: a chain across the two widths", mm)
		}
	}
	UseStockhamAVX512 = false
	buf := make([]complex128, 256)
	if z.Run(buf, buf, false) {
		t.Error("ran an AVX-512 final pass with UseStockhamAVX512 off")
	}
	if nz := ZnPlanForWide([]int{4, 8, 8}, m, tw, tw, tw, tw, true); nz == nil || nz.wide {
		t.Error("wide without UseStockhamAVX512: want the AVX2 final pass")
	}
}
