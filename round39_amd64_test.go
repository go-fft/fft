package fft

import (
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestCl3FrameRows pins cl3FrameRows: the N-D rows run on the frame kernels
// on AMD and on Intel with AVX-512 (Round 39), not on Intel without it; and
// zmFrameRows and cl3WideStrips are what this CPU's flags make them.
func TestCl3FrameRows(t *testing.T) {
	for _, avx512 := range []bool{false, true} {
		if !cl3FrameRows(false, avx512) {
			t.Errorf("AMD, AVX-512 %v: off", avx512)
		}
	}
	if !cl3FrameRows(true, true) {
		t.Error("Intel with AVX-512: off")
	}
	if cl3FrameRows(true, false) {
		t.Error("Intel without AVX-512: on")
	}
	if zmFrameRowsDefault() != cl3FrameRows(kernels.IntelCPU, kernels.UseStockhamAVX512) {
		t.Error("zmFrameRowsDefault does not follow cl3FrameRows")
	}
	if cl3WideStrips != kernels.UseStockhamSplit512 {
		t.Errorf("cl3WideStrips %v, UseStockhamSplit512 %v", cl3WideStrips, kernels.UseStockhamSplit512)
	}
}
