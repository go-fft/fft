package kernels

import "testing"

// TestStockhamPassLayoutModeZero: mode 0 is StockhamPass, on every
// architecture: the same report and the same output.
func TestStockhamPassLayoutModeZero(t *testing.T) {
	const r, ido, l1 = 4, 4, 4
	n := r * ido * l1
	root := make([]complex128, n)
	for k := range root {
		root[k] = complex(float64(k%7)-3, float64(k%5)-2)
	}
	tw, _ := StockhamTwiddles(r, ido, l1, root)
	if tw == nil {
		tw = make([]complex128, (r-1)*ido)
	}
	a, b := make([]complex128, n), make([]complex128, n)
	okA := StockhamPass(r, ido, l1, root, a, tw, false, false)
	okB := StockhamPassLayout(0, r, ido, l1, root, b, tw, false, false)
	if okA != okB {
		t.Fatalf("StockhamPass reported %v, StockhamPassLayout(0) %v", okA, okB)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("index %d: %v vs %v", i, a[i], b[i])
		}
	}
}
