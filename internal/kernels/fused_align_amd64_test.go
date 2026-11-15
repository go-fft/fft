package kernels

import (
	"testing"
	"unsafe"
)

// TestFusedTablesAligned: the tables the fused kernels read start at a multiple of
// 64 bytes (a 32 mod 64 table splits every 64-byte load across two cache lines),
// lie inside the generated static array (no heap copy), have the inverse right
// after the forward one (cosines equal, sines of opposite sign: the tables are
// moved, not changed), and start with the twiddle of w^0 = 1.
func TestFusedTablesAligned(t *testing.T) {
	for name, c := range map[string]struct {
		got   func(bool) []float64
		store []float64
	}{
		"fused256":  {Fused256Table, fused256TwStore[:]},
		"fused1024": {Fused1024Table, fused1024TwStore[:]},
		"fused2048": {Fused2048Table, fused2048TwStore[:]},
	} {
		fwd, inv := c.got(false), c.got(true)
		if a := uintptr(unsafe.Pointer(&fwd[0])); a%64 != 0 {
			t.Errorf("%s: the forward table is at %#x, %d mod 64", name, a, a%64)
		}
		lo := uintptr(unsafe.Pointer(&c.store[0]))
		hi := lo + uintptr(len(c.store))*8
		if a := uintptr(unsafe.Pointer(&fwd[0])); a < lo || a+uintptr(2*len(fwd))*8 > hi {
			t.Errorf("%s: the table is not inside the static array (a heap copy)", name)
		}
		if unsafe.Pointer(&inv[0]) != unsafe.Add(unsafe.Pointer(&fwd[0]), len(fwd)*8) {
			t.Errorf("%s: the inverse table does not follow the forward one", name)
		}
		if len(fwd) != len(inv) || len(fwd)*2+8 != len(c.store) {
			t.Fatalf("%s: table lengths %d, %d for a store of %d", name, len(fwd), len(inv), len(c.store))
		}
		for i := range fwd {
			if (i%16 < 8) && fwd[i] != inv[i] { // real parts: equal
				t.Fatalf("%s: real part %d differs between the directions: %v, %v", name, i, fwd[i], inv[i])
			}
			if (i%16 >= 8) && fwd[i] != -inv[i] { // imaginary parts: opposite signs
				t.Fatalf("%s: imaginary part %d is %v and %v", name, i, fwd[i], inv[i])
			}
		}
	}
}

// TestAlignedTableNoAlloc: alignedTable allocates nothing and keeps the table's
// values and order, whatever the alignment of the array it is given (all eight
// 8-byte offsets of a larger buffer are tried).
func TestAlignedTableNoAlloc(t *testing.T) {
	const n = 64
	buf := make([]float64, n+8+8)
	for off := 0; off < 8; off++ {
		store := buf[off : off+n+8]
		fill := func() {
			for i := range store {
				store[i] = -1
			}
			for i := 0; i < n; i++ {
				store[4+i] = float64(i) // the data starts at index 4
			}
		}
		fill()
		var got []float64
		if a := testing.AllocsPerRun(1, func() { fill(); got = alignedTable(store) }); a != 0 {
			t.Errorf("offset %d: alignedTable allocates %v times", off, a)
		}
		if p := uintptr(unsafe.Pointer(&got[0])); p%64 != 0 {
			t.Errorf("offset %d: the table is at %d mod 64", off, p%64)
		}
		if len(got) != n {
			t.Fatalf("offset %d: %d elements", off, len(got))
		}
		for i, v := range got {
			if v != float64(i) {
				t.Fatalf("offset %d: element %d is %v", off, i, v)
			}
		}
	}
}
