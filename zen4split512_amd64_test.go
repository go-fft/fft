package fft

import (
	"math"
	"testing"

	"github.com/go-fft/fft/internal/kernels"
)

// TestZen4Split512Lengths pins the lengths the Zen 4 route sends to the
// 512-bit split layout (Round 51) and their factorizations.
func TestZen4Split512Lengths(t *testing.T) {
	want := map[int]bool{1 << 16: true, 1 << 17: true, 1 << 18: true}
	got := zen4Split512Lengths()
	if len(got) != len(want) {
		t.Fatalf("zen4Split512Lengths = %v, want %v", got, want)
	}
	for n := range want {
		if !got[n] {
			t.Errorf("zen4Split512Lengths lacks %d", n)
		}
	}
	for n, skip := range zen4Zen3RouteSkip() {
		if !skip || !(want[n] || n == 2048) || n > zen4Zen3RouteMax {
			t.Errorf("zen4Zen3RouteSkip has %d", n)
		}
	}
	if !kernels.AMDZen4 {
		if zen4Split512 != nil || zen4Modes != nil {
			t.Error("the 512-bit split route is on a CPU that is not a Zen 4")
		}
		return
	}
	for n := range want {
		if !zen4Split512[n] || zen3Route(n) || !hswSplitKept(n) || !wide512(n) {
			t.Errorf("Zen 4 %d: split512 %v, zen3Route %v, kept %v, wide %v", n, zen4Split512[n], zen3Route(n), hswSplitKept(n), wide512(n))
		}
	}
	if zen3Route(1024) != true || zen4Split512[1024] {
		t.Error("Zen 4 1024 must stay on the AVX2 route")
	}
}

// TestZen4Split512Bits: a plan on the 512-bit split layout gives the same bits
// as the 256-bit split one and as the interleaved one, in both directions (the
// layouts move data, the arithmetic is the same), and its passes do run split
// at 512 bits.
func TestZen4Split512Bits(t *testing.T) {
	if !kernels.UseStockhamAVX2 || !kernels.UseStockhamAVX512 {
		t.Skip("needs AVX2 and AVX-512")
	}
	defer func(z map[int]bool, m func(r, ido, l1 []int) []uint8, tab map[int][]int, mx int, sk map[int]bool, c int) {
		zen4Split512, zen4Modes, splitTable, zen3RouteMax, zen3RouteSkip, cascadeMin = z, m, tab, mx, sk, c
	}(zen4Split512, zen4Modes, splitTable, zen3RouteMax, zen3RouteSkip, cascadeMin)
	cascadeMin = 0 // an Intel CPU runs these lengths blocked, interleaved
	for _, n := range []int{1024, 2048, 1 << 16, 1 << 17, 1 << 18} {
		x := make([]complex128, n)
		for i := range x {
			x[i] = complex(math.Sin(float64(i)*0.37)+float64(i%7), math.Cos(float64(i)*0.11)-float64(i%5))
		}
		build := func(route string) *skPlan {
			zen3RouteMax, zen3RouteSkip, zen4Split512 = 0, nil, nil
			switch route {
			case "512":
				zen4Split512, zen4Modes = map[int]bool{n: true}, kernels.StockhamSplitModes512
				splitTable = map[int][]int{n: splitTableFor512(n)}
			case "256":
				zen3RouteMax = 1 << 20
			}
			return newSKPlan(n)
		}
		p512, p256, pI := build("512"), build("256"), build("")
		split := 0
		for _, st := range p512.stages {
			if st.split > 3 {
				split++
			}
		}
		if split < 2 && n >= 1<<16 {
			t.Errorf("n=%d: %d 512-bit split passes", n, split)
		}
		for _, inv := range []bool{false, true} {
			a, b, c := make([]complex128, n), make([]complex128, n), make([]complex128, n)
			p512.transform(a, x, inv)
			p256.transform(b, x, inv)
			pI.transform(c, x, inv)
			for i := range a {
				if a[i] != b[i] || a[i] != c[i] {
					t.Fatalf("n=%d inverse=%v: bin %d: 512-bit split %v, 256-bit split %v, interleaved %v", n, inv, i, a[i], b[i], c[i])
				}
			}
		}
	}
}

// splitTableFor512 is the factorization the Zen 4 route gives n.
func splitTableFor512(n int) []int {
	if f := intelSplitTable512()[n]; f != nil {
		return f
	}
	return skFactorizeOrder(n, compOddFirst)
}
