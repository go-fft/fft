package fft

import (
	"math"
	"testing"
)

func TestPow2StockhamMaxAMD64(t *testing.T) {
	for _, c := range []struct {
		avx2, avx512 bool
		want         int
	}{{true, true, math.MaxInt}, {false, true, math.MaxInt}, {true, false, math.MaxInt}, {false, false, 0}} {
		if got := pow2StockhamMaxAMD64(c.avx2, c.avx512); got != c.want {
			t.Errorf("pow2StockhamMaxAMD64(avx2=%v, avx512=%v) = %d, want %d", c.avx2, c.avx512, got, c.want)
		}
	}
}

func TestR8MaxPow2AMD64(t *testing.T) {
	for _, c := range []struct {
		avx512, intel bool
		want          int
	}{{true, true, math.MaxInt}, {true, false, math.MaxInt}, {false, true, math.MaxInt}, {false, false, 4096}} {
		if got := r8MaxPow2AMD64(c.avx512, c.intel); got != c.want {
			t.Errorf("r8MaxPow2AMD64(avx512=%v, intel=%v) = %d, want %d", c.avx512, c.intel, got, c.want)
		}
	}
}

func TestParMinChunkAMD64(t *testing.T) {
	if parMinChunkAMD64(true) != 1<<14 || parMinChunkAMD64(false) != 1<<13 {
		t.Error("parMinChunkAMD64: want 16384 with AVX2, 8192 without")
	}
}

// TestRadix16TableAMD64 pins which machines take radix-16 passes and the
// factorizations they take.
func TestRadix16TableAMD64(t *testing.T) {
	if radix16TableAMD64(false, false, true) != nil || radix16TableAMD64(true, true, true) != nil || radix16TableAMD64(false, true, false) != nil {
		t.Error("radix 16 without AVX2, or with AVX-512")
	}
	amd, intel := radix16TableAMD64(true, false, false), radix16TableAMD64(true, false, true)
	if len(amd) != 3 || len(intel) != 4 || intel[2048] == nil || amd[2048] != nil {
		t.Errorf("tables: AMD %v, Intel %v", amd, intel)
	}
	for _, tab := range []map[int][]int{amd, intel} {
		for n, f := range tab {
			p := 1
			for _, r := range f {
				p *= r
			}
			if p != n {
				t.Errorf("%d factored as %v", n, f)
			}
		}
	}
}

// TestSplitTableAMD64 pins which powers of two change their factorization
// with the split layout.
func TestSplitTableAMD64(t *testing.T) {
	if splitTableAMD64(false) != nil {
		t.Error("split factorizations without the split layout")
	}
	tab := splitTableAMD64(true)
	for _, n := range []int{16, 32, 64, 512, 1024} {
		if tab[n] != nil {
			t.Errorf("%d: %v, want its interleaved factorization", n, tab[n])
		}
	}
	for n, f := range tab {
		if p := product(f); p != n {
			t.Errorf("%d factored as %v", n, f)
		}
	}
	for _, n := range []int{128, 256, 4096, 1 << 20, 1 << 30} {
		if tab[n] == nil {
			t.Errorf("%d: not in the table", n)
		}
	}
}
