package fft

import "testing"

// TestNDStripAxisRadix9Route forces Round 35's radix-9 route (Intel with
// AVX-512) on any machine: with it, compTakes16(9·p) is true while the axis
// plan of 9·p is Rader or Bluestein, which an N-D plan used to replace with a
// "Stockham" plan of the smooth part only. Without the sk != nil guard in
// initScratch this test fails on 9·p lengths on every amd64 CPU.
func TestNDStripAxisRadix9Route(t *testing.T) {
	defer func(o bool) { csRadix9On = o; ndForgetPlans() }(csRadix9On)
	csRadix9On = true
	ndForgetPlans()
	if !compTakes16(171) {
		t.Fatal("the forced route does not give 171 a radix-9 pass: the test is not on the defect's route")
	}
	lens := []int{153, 171, 207, 261, 342, 513, 1197, 1998, 9, 81, 1296, 45, 97}
	if bad := ndStripWrong(lens); len(bad) > 0 {
		t.Fatalf("on the radix-9 route, strip axes of %v transform wrongly", bad)
	}
}
