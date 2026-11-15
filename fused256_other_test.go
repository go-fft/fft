//go:build !amd64

package fft

import "testing"

// Off amd64 the fused 256-point kernel does not exist: the route is off, and
// the stub reports that it did nothing.
func TestFused256NotHere(t *testing.T) {
	if p := NewPlan(256); fused256Route(256) || (p.sk != nil && p.sk.f256) {
		t.Error("the fused 256-point route is on off amd64")
	}
	if fused256Run(nil, nil, false) {
		t.Error("fused256Run reports a transform off amd64")
	}
}
