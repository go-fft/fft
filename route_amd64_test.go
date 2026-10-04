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
	for _, c := range []struct {
		avx512, intel bool
		want          int
	}{{true, true, math.MaxInt}, {true, false, math.MaxInt}, {false, true, math.MaxInt}, {false, false, 4096}} {
		if got := r8MaxPow2AMD64(c.avx512, c.intel); got != c.want {
			t.Errorf("r8MaxPow2AMD64(avx512=%v, intel=%v) = %d, want %d", c.avx512, c.intel, got, c.want)
		}
	}
}
