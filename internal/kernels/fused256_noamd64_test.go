//go:build !amd64

package kernels

import "testing"

// Off amd64 there is no fused kernel: the route is off and the stubs do nothing.
func TestFused256Stubs(t *testing.T) {
	if Fused256On {
		t.Error("Fused256On is true off amd64")
	}
	Fused256(nil, nil, false)
	if Fused256Table(false) != nil || Fused256Table(true) != nil {
		t.Error("Fused256Table is not nil off amd64")
	}
}
