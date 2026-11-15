//go:build !amd64

package kernels

import "testing"

// TestFusedSOffAmd64: the complex64 kernels are off off amd64: their gates are
// false, they report false and their tables are nil.
func TestFusedSOffAmd64(t *testing.T) {
	if FusedS64On || FusedS128On || FusedS256On || FusedS512On || FusedS1024On || FusedS2048On || FusedS4096On || FusedS8192On {
		t.Error("a complex64 gate is true off amd64")
	}
	var a [64]complex64
	var b [128]complex64
	var c [256]complex64
	var d [512]complex64
	var e [1024]complex64
	var f [2048]complex64
	var g [4096]complex64
	var h [8192]complex64
	if FusedS64(&a, &a, false) || FusedS128(&b, &b, true) || FusedS256(&c, &c, false) || FusedS512(&d, &d, true) || FusedS1024(&e, &e, false) ||
		FusedS2048(&f, &f, true) || FusedS4096(&g, &g, false) || FusedS8192(&h, &h, true) {
		t.Error("a complex64 kernel reports true off amd64")
	}
	for _, tab := range []func(bool) []float32{FusedS64Table, FusedS128Table, FusedS256Table, FusedS512Table, FusedS1024Table, FusedS2048Table, FusedS4096Table, FusedS8192Table} {
		if tab(false) != nil || tab(true) != nil {
			t.Error("a complex64 table is not nil off amd64")
		}
	}
}
