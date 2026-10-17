//go:build !amd64

package fft

import "testing"

// forceFMA builds FMA plans for the rest of the test: there are no FMA
// kernels here, so their passes run the Go reference.
func forceFMA(t *testing.T) {
	t.Helper()
	a := fmaAvailable
	t.Cleanup(func() { fmaAvailable = a; forgetFMAPlans() })
	fmaAvailable = true
	forgetFMAPlans()
}
