//go:build !amd64

package fft

import "testing"

// forceFMA builds FMA plans for the rest of the test: there are no FMA
// kernels here, so their passes run the Go reference. The default arithmetic
// stays unfused meanwhile (defaultFuses off), so the tests that compare an FMA
// plan with "the default" keep comparing it with the unfused plan; a test of
// the fused default turns defaultFuses back on after calling forceFMA.
func forceFMA(t *testing.T) {
	t.Helper()
	a, d := fmaAvailable, defaultFuses
	t.Cleanup(func() { fmaAvailable, defaultFuses = a, d; forgetFMAPlans() })
	fmaAvailable, defaultFuses = true, false
	forgetFMAPlans()
}
