package fft

import (
	"slices"
	"testing"
)

// TestR42FMATableAMD64: Cascade Lake's table (512-bit split layout), Zen 3's
// (256-bit split layout on AMD), none elsewhere; forced, whatever the CPU.
func TestR42FMATableAMD64(t *testing.T) {
	cl, zn := fmaTableAMD64(true, false), fmaTableAMD64(false, true)
	if fmaTableAMD64(false, false) != nil {
		t.Error("a CPU that was not measured got a table")
	}
	if c := cl[1024]; !slices.Equal(c.factors, []int{32, 32}) || c.fuse != fmaAll {
		t.Errorf("Cascade Lake 1024: %v", c)
	}
	if c := cl[2048]; !slices.Equal(c.factors, []int{32, 8, 8}) || c.fuse != 0b011 {
		t.Errorf("Cascade Lake 2048: %v", c)
	}
	for _, n := range []int{4096, 8192, 16384} {
		if c, ok := cl[n]; !ok || c.factors != nil || c.fuse != 0 {
			t.Errorf("Cascade Lake %d: %v", n, c)
		}
	}
	if c := cl[1<<16]; c.factors != nil || c.fuse != 0b011111 {
		t.Errorf("Cascade Lake 65536: %v", c)
	}
	if len(zn) != 1 || zn[128].factors != nil || zn[128].fuse != 0b01 {
		t.Errorf("Zen 3: %v", zn)
	}
	for n, c := range cl {
		if c.factors != nil && factorProduct(c.factors) != n {
			t.Errorf("Cascade Lake %d: factors %v", n, c.factors)
		}
	}
}
