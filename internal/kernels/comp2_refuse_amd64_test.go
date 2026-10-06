package kernels

import "testing"

// TestComp2PassRefuses: StockhamPass refuses, before the radix-10, -15 or -20
// kernel runs, buffers or a twiddle table shorter than the last element the
// kernel touches, and counts below one (two negative ones multiply to a
// positive length).
func TestComp2PassRefuses(t *testing.T) {
	if !UseStockhamAVX2 {
		t.Skip("no AVX2: StockhamPass reports false before any check")
	}
	for _, r := range []int{10, 15, 20} {
		ido, l1 := 4, 3
		n := r * ido * l1
		cc, ch := make([]complex128, n), make([]complex128, n)
		tw := make([]complex128, skTwLen(r, ido))
		for _, c := range []struct {
			name string
			f    func()
		}{
			{"cc short", func() { StockhamPass(r, ido, l1, cc[:n-1], ch, tw, false, false) }},
			{"ch short", func() { StockhamPass(r, ido, l1, cc, ch[:n-1], tw, false, false) }},
			{"tw short", func() { StockhamPass(r, ido, l1, cc, ch, tw[:len(tw)-1], false, false) }},
			{"last cc short", func() { StockhamPass(r, 1, l1, cc[:r*l1-1], ch, nil, false, false) }},
			{"negative counts", func() { StockhamPass(r, -ido, -l1, cc, ch, tw, false, false) }},
			{"zero ido", func() { StockhamPass(r, 0, l1, cc, ch, tw, false, false) }},
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("r=%d %s: no panic", r, c.name)
					}
				}()
				c.f()
			}()
		}
	}
}
