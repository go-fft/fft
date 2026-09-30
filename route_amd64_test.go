package fft

import "testing"

func TestPow2StockhamMaxAMD64(t *testing.T) {
	if got := pow2StockhamMaxAMD64(true); got != 4096 {
		t.Errorf("with AVX2: %d, want 4096", got)
	}
	if got := pow2StockhamMaxAMD64(false); got != 0 {
		t.Errorf("without AVX2: %d, want 0 (the pow2 kernel for every power of two)", got)
	}
}
