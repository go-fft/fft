package fft

import "testing"

func TestPow2StockhamMaxAMD64(t *testing.T) {
	for _, c := range []struct {
		avx2, avx512 bool
		want         int
	}{{true, true, 16384}, {false, true, 16384}, {true, false, 4096}, {false, false, 0}} {
		if got := pow2StockhamMaxAMD64(c.avx2, c.avx512); got != c.want {
			t.Errorf("pow2StockhamMaxAMD64(avx2=%v, avx512=%v) = %d, want %d", c.avx2, c.avx512, got, c.want)
		}
	}
}
