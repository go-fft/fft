package fft

import (
	"math/cmplx"
	"testing"
)

// TestStockhamMatchesMixedRadix cross-checks the Stockham passes against the
// independent recursive mixed-radix engine at every smooth length up to 3000,
// in both directions and with dst aliasing src. That range reaches every pass
// (radix 2/3/4/5/7 and the general radix 11/13) at both ido == 1 and ido > 1,
// and plans with both an odd and an even number of passes.
func TestStockhamMatchesMixedRadix(t *testing.T) {
	for n := 2; n <= 3000; n++ {
		if !factorsAreSmall(n) {
			continue
		}
		x := cmplxSignal(n)
		sk, ct := newSKPlan(n), newCTPlan(n)
		tolN := 1e-12 * float64(n)
		for _, inverse := range []bool{false, true} {
			want := make([]complex128, n)
			ct.transform(want, x, inverse)
			got := make([]complex128, n)
			sk.transform(got, x, inverse)
			aliased := append([]complex128(nil), x...)
			sk.transform(aliased, aliased, inverse)
			for i := range want {
				if d := cmplx.Abs(got[i] - want[i]); d > tolN {
					t.Fatalf("n=%d inverse=%v index %d: Stockham %v vs mixed-radix %v", n, inverse, i, got[i], want[i])
				}
				if aliased[i] != got[i] {
					t.Fatalf("n=%d inverse=%v index %d: aliased %v vs %v", n, inverse, i, aliased[i], got[i])
				}
			}
		}
	}
}

// TestPooledConvolutionBufferIsCleared runs each convolution engine twice on
// one plan, first on a large input and then on a small one, and checks the
// second result against a fresh plan: the pooled buffer comes back dirty, so a
// zero pad that is not re-cleared would leak the first call into the second.
// 10007 takes Rader's padded-linear path (10006 = 2·5003 is not smooth) and
// 641 takes Bluestein; both zero-pad their convolution.
func TestPooledConvolutionBufferIsCleared(t *testing.T) {
	for _, n := range []int{641, 10007} {
		dirty := NewPlan(n)
		big := make([]complex128, n)
		for i := range big {
			big[i] = complex(1e6, -1e6)
		}
		scratch := make([]complex128, n)
		dirty.FFT(scratch, big)

		x := cmplxSignal(n)
		got := dirty.FFT(make([]complex128, n), x)
		want := NewPlan(n).FFT(make([]complex128, n), x)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d index %d: reused plan %v vs fresh plan %v", n, i, got[i], want[i])
			}
		}
	}
}
