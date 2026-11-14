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

// TestZen4FMATable: only a Zen 4 changes the Zen 3 table, and only at 1024
// (32·32, two radix-32 passes, every pass fused) and 2048 (32·8·8); the table of another CPU,
// and nil, are returned as given.
func TestZen4FMATable(t *testing.T) {
	if zen4FMATable(nil, true) != nil {
		t.Fatal("nil table")
	}
	if _, ok := zen4FMATable(fmaTableAMD64(false, true), false)[1024]; ok {
		t.Fatal("a Zen 3 takes 32·32 at 1024")
	}
	z := zen4FMATable(fmaTableAMD64(false, true), true)
	c, ok := z[1024]
	if !ok || !slices.Equal(c.factors, []int{32, 32}) || c.fuse != fmaAll {
		t.Fatalf("Zen 4 1024: %+v", c)
	}
	c, ok = z[2048]
	if !ok || !slices.Equal(c.factors, []int{32, 8, 8}) || c.fuse != fmaAll {
		t.Fatalf("Zen 4 2048: %+v", c)
	}
	if _, ok := z[128]; !ok || len(z) != 3 {
		t.Fatalf("Zen 4 table %v keeps Zen 3's 128 and adds only 1024 and 2048", z)
	}
	// The table drives the plan: a Zen 4 FMA plan of 1024 points runs two passes.
	defer func(m map[int]fmaChoice) { fmaTable = m }(fmaTable)
	fmaTable = z
	isolatePlans(t)
	if hasScalarRadix32(newSKPlanFactorsOpt(1024, []int{32, 32}, true)) {
		t.Skip("no radix-32 split kernel on this CPU: the planner refuses the table's plan and takes skFactorize's")
	}
	p := newSKPlanOpt(1024, true)
	if len(p.stages) != 2 {
		t.Fatalf("an FMA plan of 1024 points has %d passes under the Zen 4 table, want 2", len(p.stages))
	}
}
