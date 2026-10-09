package fft

import (
	"math"
	"slices"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
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
	if radix16TableAMD64(false, false, true) != nil || radix16TableAMD64(true, true, false) != nil || radix16TableAMD64(false, true, false) != nil || radix16TableAMD64(false, true, true) != nil {
		t.Error("radix 16 without AVX2, or with AVX-512 on AMD")
	}
	if tab := radix16TableAMD64(true, true, true); len(tab) != 1 || !slices.Equal(tab[128], []int{16, 8}) {
		t.Errorf("AVX-512 Intel: %v, want 128 = 16·8 only", tab)
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

// TestIntelSplitTableAMD64: the 512-bit split layout takes its own table,
// whatever the 256-bit setting, and only 1024 and 8192 change.
func TestIntelSplitTableAMD64(t *testing.T) {
	for _, on := range []bool{false, true} {
		for _, intel := range []bool{false, true} {
			tab := intelSplitTableAMD64(on, true, intel)
			if len(tab) != 2 || !slices.Equal(tab[1024], []int{8, 4, 4, 8}) || !slices.Equal(tab[8192], []int{8, 8, 4, 4, 8}) {
				t.Errorf("512 on, 256 %v, intel %v: %v", on, intel, tab)
			}
		}
	}
	if intelSplitTableAMD64(false, false, false) != nil || intelSplitTableAMD64(false, false, true) != nil ||
		len(intelSplitTableAMD64(true, false, false)) != len(splitTableAMD64(true)) {
		t.Error("512 off: want splitTableAMD64")
	}
	if len(intelSplitTableAMD64(true, false, true)) != len(hswSplitTable()) {
		t.Error("256-bit layout on Intel: want hswSplitTable")
	}
}

// TestHswSplitTable pins Round 29's Haswell table: 512, 8192, 16384 and
// splitPow2Factors from 2^19; 1024 to 2^18 but those keep their
// factorization.
func TestHswSplitTable(t *testing.T) {
	tab := hswSplitTable()
	for n, f := range tab {
		if p := product(f); p != n {
			t.Errorf("%d factored as %v", n, f)
		}
	}
	for _, n := range []int{64, 128, 256, 1024, 2048, 4096, 32768, 65536, 1 << 17, 1 << 18} {
		if tab[n] != nil {
			t.Errorf("%d: %v, want its interleaved factorization", n, tab[n])
		}
	}
	if !slices.Equal(tab[512], []int{8, 4, 16}) || !slices.Equal(tab[8192], []int{8, 8, 4, 4, 8}) ||
		!slices.Equal(tab[16384], []int{8, 4, 8, 8, 8}) || !slices.Equal(tab[1<<20], splitPow2Factors(20)) || tab[1<<30] == nil {
		t.Errorf("hswSplitTable %v", tab)
	}
}

// TestIntelStripOrderAMD64: with the 512-bit layout, the strips of the
// lengths intelSplitTable512 changes keep skFactorizeOrder's factorization.
func TestIntelStripOrderAMD64(t *testing.T) {
	if intelStripOrderAMD64(false, false) != nil {
		t.Error("strip orders without the 512-bit layout or Haswell's table")
	}
	for _, hsw := range []bool{false, true} {
		m := intelStripOrderAMD64(true, hsw)
		if len(m) != len(intelSplitTable512()) || !m[1024] || !m[8192] {
			t.Errorf("strip orders %v", m)
		}
	}
	m := intelStripOrderAMD64(false, true)
	if len(m) != len(hswSplitTable()) || !m[512] || !m[8192] || !m[16384] || m[1024] {
		t.Errorf("Haswell strip orders %v", m)
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

// TestClRowOrderAMD64: with the 512-bit layout, rows of 1024 points of a
// large N-D plan take 4·8·4·8; without it, nothing changes.
func TestClRowOrderAMD64(t *testing.T) {
	if clRowOrderAMD64(false) != nil {
		t.Error("row orders without the 512-bit layout")
	}
	m := clRowOrderAMD64(true)
	if len(m) != 1 || !slices.Equal(m[1024], []int{4, 8, 4, 8}) {
		t.Errorf("row orders %v", m)
	}
}

// TestZnFrameMaxAMD64: the frame kernels run on AMD only, up to the kernels
// package's limits.
func TestZnFrameMaxAMD64(t *testing.T) {
	if got := znFrameMaxAMD64(false, 77); got != 77 {
		t.Errorf("AMD: %d", got)
	}
	if got := znFrameMaxAMD64(true, 77); got != 0 {
		t.Errorf("Intel: %d", got)
	}
	if kernels.ZnTwoPassMax < 128 || kernels.ZnThreePassMax < 256 {
		t.Errorf("limits %d, %d", kernels.ZnTwoPassMax, kernels.ZnThreePassMax)
	}
}
