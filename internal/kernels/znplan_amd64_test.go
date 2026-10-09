package kernels

import "testing"

// znTables builds a ZnPlan's twiddle tables for passes 0 and 1 of radices r
// and modes m, forward and conjugate.
func znTables(r []int, m []uint8) (tw0, tw0c, tw1, tw1c []complex128) {
	n := 1
	for _, x := range r {
		n *= x
	}
	root := znRoot(n)
	tab := func(k, l1 int) ([]complex128, []complex128) {
		ido := n / (l1 * r[k])
		if m[k] != 0 {
			return StockhamSplitTwiddles(r[k], ido, l1, root)
		}
		return StockhamTwiddles(r[k], ido, l1, root)
	}
	tw0, tw0c = tab(0, 1)
	if len(r) == 3 {
		tw1, tw1c = tab(1, r[0])
	}
	return
}

// TestZnPlanMatchesPasses: a ZnPlan gives, bit for bit, what its passes
// give run one after the other into buffers (StockhamPassLayout, then
// StockhamPass for the final one), for every pair and triple of radices with
// kernels up to ZnTwoPassMax and ZnThreePassMax points, every chain of
// layouts the kernels accept, both directions, in place and not.
func TestZnPlanMatchesPasses(t *testing.T) {
	if !UseStockhamAVX2 {
		t.Skip("no AVX2")
	}
	radices := []int{2, 3, 4, 5, 8, 10, 12, 15, 16, 20}
	var shapes [][]int
	for _, a := range radices {
		for _, b := range radices {
			if a*b <= ZnTwoPassMax {
				shapes = append(shapes, []int{a, b})
			}
			for _, c := range radices {
				if a*b*c <= ZnThreePassMax {
					shapes = append(shapes, []int{a, b, c})
				}
			}
		}
	}
	chains := map[int][][]uint8{
		2: {{0, 0}, {splitNone, 0}},
		3: {{0, 0, 0}, {splitNone, 0, 0}, {0, splitNone, 0}, {splitNone, splitNone, 0}, {splitOut, splitIn, 0}},
	}
	cases, split := map[int]int{}, 0
	for _, r := range shapes {
		s := len(r)
		n := 1
		for _, x := range r {
			n *= x
		}
		for _, m := range chains[s] {
			if m[0] != 0 && !splitPass(r[0], n/r[0]) || m[1] != 0 && (s == 2 || !splitPass(r[1], r[2])) {
				continue // no split kernel for that pass: no split table either
			}
			tw0, tw0c, tw1, tw1c := znTables(r, m)
			z := ZnPlanFor(r, m, tw0, tw0c, tw1, tw1c)
			if z == nil {
				if m[0] == 0 && m[1] == 0 {
					t.Fatalf("%v: refused interleaved", r)
				}
				continue
			}
			if m[0] != 0 || m[1] != 0 {
				split++
			}
			for _, inverse := range []bool{false, true} {
				t0, t1 := tw0, tw1
				if inverse {
					t0, t1 = tw0c, tw1c
				}
				for si, src := range znSignals(n) {
					want := make([]complex128, n)
					in, l1 := src, 1
					for i := 0; i < s-1; i++ {
						out := make([]complex128, n)
						tw := t0
						if i == 1 {
							tw = t1
						}
						if !StockhamPassLayout(m[i], r[i], n/(l1*r[i]), l1, in, out, tw, inverse, false) {
							t.Fatalf("%v: no pass kernel", r)
						}
						in, l1 = out, l1*r[i]
					}
					if !StockhamPass(r[s-1], 1, l1, in, want, nil, inverse, false) {
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
					cases[s]++
				}
			}
		}
	}
	if cases[2] < 7*2*40 || cases[3] < 7*2*100 || split < 20 {
		t.Fatalf("only %v cases compared, %d split chains", cases, split)
	}
}

// TestZnPlanRefuses: ZnPlanFor returns nil where the kernels cannot run the
// transform or the layouts do not chain; Run reports false once the AVX2
// kernels are off, and panics before any kernel runs on a short slice.
func TestZnPlanRefuses(t *testing.T) {
	if !UseStockhamAVX2 {
		t.Skip("no AVX2")
	}
	for _, c := range []struct {
		r []int
		m []uint8
	}{
		{[]int{8}, []uint8{0}}, {[]int{2, 2, 2, 2}, []uint8{0, 0, 0, 0}}, {[]int{8, 8}, []uint8{0}},
		{[]int{1, 8}, []uint8{0, 0}}, {[]int{8, 1}, []uint8{0, 0}}, {[]int{21, 2}, []uint8{0, 0}}, {[]int{2, 21}, []uint8{0, 0}},
		{[]int{20, 20}, []uint8{0, 0}}, {[]int{16, 20}, []uint8{0, 0}}, {[]int{7, 8}, []uint8{0, 0}}, {[]int{8, 7}, []uint8{0, 0}},
		{[]int{8, 8}, []uint8{splitOut, 0}}, {[]int{8, 8}, []uint8{splitBoth, 0}}, {[]int{8, 8}, []uint8{splitIn, 0}},
		{[]int{8, 8}, []uint8{9, 0}}, {[]int{8, 8}, []uint8{0, splitNone}}, {[]int{2, 8}, []uint8{splitNone, 0}},
		{[]int{8, 2}, []uint8{splitNone, 0}}, {[]int{8, 6}, []uint8{splitNone, 0}},
		{[]int{8, 8, 16}, []uint8{0, 0, 0}}, {[]int{7, 8, 8}, []uint8{0, 0, 0}}, {[]int{8, 7, 8}, []uint8{0, 0, 0}},
		{[]int{8, 8, 7}, []uint8{0, 0, 0}}, {[]int{4, 8, 8}, []uint8{splitIn, 0, 0}},
		{[]int{4, 8, 8}, []uint8{splitBoth, splitIn, 0}}, {[]int{4, 8, 8}, []uint8{0, splitOut, 0}},
		{[]int{4, 8, 8}, []uint8{splitOut, 0, 0}}, {[]int{4, 8, 8}, []uint8{0, splitIn, 0}},
		{[]int{4, 8, 2}, []uint8{0, splitNone, 0}},
	} {
		tw := make([]complex128, 4096)
		if ZnPlanFor(c.r, c.m, tw, tw, tw, tw) != nil {
			t.Errorf("%v %v: built", c.r, c.m)
		}
	}
	// Tables one entry short, each in turn, then exactly long enough.
	for _, c := range []struct {
		r  []int
		m  []uint8
		tl [2]int
	}{
		{[]int{8, 8}, []uint8{0, 0}, [2]int{7 * 8, 0}},
		{[]int{16, 8}, []uint8{0, 0}, [2]int{2 * 15 * 8, 0}},
		{[]int{4, 8, 8}, []uint8{splitOut, splitIn, 0}, [2]int{3 * 64, 7 * 8}},
	} {
		full := make([]complex128, 4096)
		for q := range 4 {
			if c.tl[q/2] == 0 {
				continue
			}
			tws := [4][]complex128{full, full, full, full}
			tws[q] = full[:c.tl[q/2]-1]
			if ZnPlanFor(c.r, c.m, tws[0], tws[1], tws[2], tws[3]) != nil {
				t.Errorf("%v: built with table %d short", c.r, q)
			}
		}
		if ZnPlanFor(c.r, c.m, full[:c.tl[0]], full[:c.tl[0]], full[:c.tl[1]], full[:c.tl[1]]) == nil {
			t.Errorf("%v: refused exact tables", c.r)
		}
	}
	defer func(v bool) { UseStockhamAVX2 = v }(UseStockhamAVX2)
	two := ZnPlanFor([]int{8, 8}, []uint8{0, 0}, make([]complex128, 56), make([]complex128, 56), nil, nil)
	three := ZnPlanFor([]int{4, 8, 8}, []uint8{0, 0, 0}, make([]complex128, 192), make([]complex128, 192), make([]complex128, 56), make([]complex128, 56))
	buf := make([]complex128, 256)
	short := func(name string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic", name)
			}
		}()
		f()
	}
	short("two cc", func() { two.Run(buf[:63], buf, false) })
	short("two ch", func() { two.Run(buf, buf[:63], false) })
	short("three cc", func() { three.Run(buf[:255], buf, true) })
	short("three ch", func() { three.Run(buf, buf[:255], true) })
	UseStockhamAVX2 = false
	if two.Run(buf, buf, false) || three.Run(buf, buf, false) {
		t.Error("ran without AVX2")
	}
	if ZnPlanFor([]int{8, 8}, []uint8{0, 0}, buf, buf, nil, nil) != nil {
		t.Error("built without AVX2")
	}
}
