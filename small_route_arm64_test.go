package fft

import (
	"slices"
	"testing"
)

// TestSmallArmPow2 pins Round 30's arm64 factorizations of the powers of two
// from 64 to 4096 points, and that every other length keeps
// skFactorizeOrder's.
func TestSmallArmPow2(t *testing.T) {
	want := map[int][]int{
		64: {8, 8}, 128: {4, 4, 8}, 256: {4, 8, 8}, 512: {8, 8, 8},
		1024: {4, 4, 8, 8}, 2048: {4, 8, 8, 8}, 4096: {8, 8, 8, 8},
	}
	for n, f := range want {
		if got := compFactorize(n); !slices.Equal(got, f) {
			t.Errorf("compFactorize(%d) = %v, want %v", n, got, f)
		}
	}
	for _, n := range []int{2, 4, 8, 16, 32, 8192, 65536, 96, 1000, 4095} {
		if smallArmPow2(n) != nil {
			t.Errorf("smallArmPow2(%d) = %v, want nil", n, smallArmPow2(n))
		}
		if got, old := compFactorize(n), skFactorizeOrder(n, compOddFirst); !slices.Equal(got, old) {
			t.Errorf("compFactorize(%d) = %v, want skFactorizeOrder's %v", n, got, old)
		}
	}
}
