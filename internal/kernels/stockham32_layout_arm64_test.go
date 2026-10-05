package kernels

import "testing"

// twiddle32At reads twiddle j of point i from an arm64 StockhamTwiddles32
// table: groups of g points (twGroups), each storing, per j, the g real parts
// and then the g imaginary parts as float32.
func twiddle32At(t []complex64, r, ido, i, j int) complex64 {
	f := make([]float32, 0, 2*len(t))
	for _, v := range t {
		f = append(f, real(v), imag(v))
	}
	at, i0 := 0, 0
	for _, g := range twGroups(ido) {
		if i < i0+g {
			p := at + 2*g*(j-1) + i - i0
			return complex(f[p], f[p+g])
		}
		at += 2 * (r - 1) * g
		i0 += g
	}
	panic("point out of range")
}

// TestStockham32KernelFinalPass: a final pass (ido == 1) runs on a kernel only
// for l1 a multiple of four.
func TestStockham32KernelFinalPass(t *testing.T) {
	for l1, want := range map[int]bool{1: false, 2: false, 4: true, 6: false, 8: true} {
		if got := stockham32Kernel(5, 1, l1); got != want {
			t.Errorf("stockham32Kernel(5, 1, %d) = %v", l1, got)
		}
	}
}
