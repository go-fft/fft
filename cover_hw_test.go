package fft

import "testing"

// Statements that a route reaches only on some CPUs (the fused MDCT gate is behind a
// short-circuit that an AVX-512 host never evaluates, the identity scale is skipped
// by the plans that fold the factor elsewhere): the 100% gate must not depend on
// the CPU, so each is called directly here.

func TestMDCTFusedLenGate(t *testing.T) {
	for m, want := range map[int]bool{32: false, 64: true, 128: false, 256: false} {
		if got := mdctFusedLen(m); got != want {
			t.Errorf("mdctFusedLen(%d) = %v, want %v", m, got, want)
		}
	}
}

func TestScaleComplexIdentityAndFactor(t *testing.T) {
	x := []complex128{1 + 2i, -3 + 0.5i}
	scaleComplex(x, 1)
	if x[0] != 1+2i || x[1] != -3+0.5i {
		t.Fatalf("scaling by 1 changed x: %v", x)
	}
	scaleComplex(x, 2)
	if x[0] != 2+4i || x[1] != -6+1i {
		t.Fatalf("scaling by 2: %v", x)
	}
}
