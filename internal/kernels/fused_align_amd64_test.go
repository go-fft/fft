package kernels

import (
	"testing"
	"unsafe"
)

// TestFusedTablesAligned: the tables the fused kernels read start at a multiple
// of 64 bytes (a 32 mod 64 table splits every 64-byte load across two cache
// lines), hold the generated values, and have the inverse right after the forward.
func TestFusedTablesAligned(t *testing.T) {
	for name, c := range map[string]struct {
		got  func(bool) []float64
		want [2][]float64
	}{
		"fused256":  {Fused256Table, [2][]float64{fused256Tw[0][:], fused256Tw[1][:]}},
		"fused1024": {Fused1024Table, [2][]float64{fused1024Tw[0][:], fused1024Tw[1][:]}},
		"fused2048": {Fused2048Table, [2][]float64{fused2048Tw[0][:], fused2048Tw[1][:]}},
	} {
		fwd, inv := c.got(false), c.got(true)
		if a := uintptr(unsafe.Pointer(&fwd[0])); a%64 != 0 {
			t.Errorf("%s: the forward table is at %#x, %d mod 64", name, a, a%64)
		}
		if unsafe.Pointer(&inv[0]) != unsafe.Add(unsafe.Pointer(&fwd[0]), len(fwd)*8) {
			t.Errorf("%s: the inverse table does not follow the forward one", name)
		}
		for d, tab := range [][]float64{fwd, inv} {
			if len(tab) != len(c.want[d]) {
				t.Fatalf("%s: table %d has %d entries, generated %d", name, d, len(tab), len(c.want[d]))
			}
			for i := range tab {
				if tab[i] != c.want[d][i] {
					t.Fatalf("%s: table %d entry %d is %v, generated %v", name, d, i, tab[i], c.want[d][i])
				}
			}
		}
	}
}
