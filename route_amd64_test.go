package fft

import (
	"math"
	"testing"
)

func TestPow2StockhamMaxAMD64(t *testing.T) {
	for _, c := range []struct {
		avx2, avx512 bool
		want         int
	}{{true, true, math.MaxInt}, {false, true, math.MaxInt}, {true, false, math.MaxInt}, {false, false, 0}} {
		if got := pow2StockhamMaxAMD64(c.avx2, c.avx512); got != c.want {
			t.Errorf("pow2StockhamMaxAMD64(avx2=%v, avx512=%v) = %d, want %d", c.avx2, c.avx512, got, c.want)
		}
	}
}

func TestR8MaxPow2AMD64(t *testing.T) {
	if got := r8MaxPow2AMD64(true); got != math.MaxInt {
		t.Errorf("r8MaxPow2AMD64(avx512) = %d, want MaxInt", got)
	}
	if got := r8MaxPow2AMD64(false); got != 4096 {
		t.Errorf("r8MaxPow2AMD64(no avx512) = %d, want 4096", got)
	}
}
