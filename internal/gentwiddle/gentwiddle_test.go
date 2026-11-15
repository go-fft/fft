package gentwiddle

import (
	"math"
	"math/big"
	"testing"
)

// CosSin against closed forms, the identity c²+s² = 1, and math.Sincos (the
// series is correctly rounded, math's argument reduction is within an ulp).
func TestCosSin(t *testing.T) {
	r := math.Sqrt2 / 2
	for _, tc := range []struct {
		num, den int
		c, s     float64
	}{{1, 8, r, r}, {3, 8, -r, r}, {1, 6, 0.5, math.Sqrt(3) / 2}, {1, 12, math.Sqrt(3) / 2, 0.5}} {
		c, s := CosSin(tc.num, tc.den)
		if math.Abs(c-tc.c) > 2e-16 || math.Abs(s-tc.s) > 2e-16 {
			t.Errorf("CosSin(%d, %d) = (%v, %v), want (%v, %v)", tc.num, tc.den, c, s, tc.c, tc.s)
		}
	}
	for den := 3; den <= 256; den += 13 {
		for num := 1; num < den; num++ {
			c, s := CosSin(num, den)
			if d := math.Abs(c*c + s*s - 1); d > 4e-16 {
				t.Fatalf("CosSin(%d, %d): c²+s²-1 = %g", num, den, d)
			}
			ms, mc := math.Sincos(2 * math.Pi * float64(num) / float64(den))
			if math.Abs(c-mc) > 1e-14 || math.Abs(s-ms) > 1e-14 {
				t.Fatalf("CosSin(%d, %d) = (%v, %v), math gives (%v, %v)", num, den, c, s, mc, ms)
			}
		}
	}
}

// CosSin32 is the float32 nearest to the 400-bit value: no neighbour is closer,
// for every angle 2πk/n of n = 8 .. 4096 (the fused complex64 kernels' tables).
// Rounding float64 then float32 can differ by one ulp in the rare cases where the
// float64 lands on a float32 midpoint; they are counted, not required.
func TestCosSin32(t *testing.T) {
	dist := func(x float32, exact *big.Float) *big.Float {
		d := new(big.Float).SetPrec(400).SetFloat64(float64(x))
		return d.Sub(d, exact).Abs(d)
	}
	double := 0
	dens := []int{8, 16, 64, 256, 1024, 2048, 4096}
	if testing.Short() {
		dens = dens[:4]
	}
	for _, den := range dens {
		for num := 1; num < den; num++ {
			cs, sn := bigCosSin(num, den)
			c, s := CosSin32(num, den)
			for _, p := range []struct {
				got   float32
				exact *big.Float
			}{{c, cs}, {s, sn}} {
				for _, nb := range []float32{math.Nextafter32(p.got, -2), math.Nextafter32(p.got, 2)} {
					if dist(nb, p.exact).Cmp(dist(p.got, p.exact)) < 0 {
						t.Fatalf("CosSin32(%d, %d): %v is not the nearest float32 (%v is closer)", num, den, p.got, nb)
					}
				}
			}
			c64, s64 := CosSin(num, den)
			if float32(c64) != c || float32(s64) != s {
				double++
			}
		}
	}
	t.Logf("%d values where float64 then float32 differs from the single rounding", double)
	if c, s := CosSin32(1, 4); c != 0 && math.Abs(float64(c)) > 1e-7 || s != 1 {
		t.Errorf("CosSin32(1, 4) = (%v, %v)", c, s)
	}
}

// bigPi is π: 3.14159265358979323846..., rounded to float64 it is math.Pi.
func TestBigPi(t *testing.T) {
	if p, _ := bigPi().Float64(); p != math.Pi {
		t.Errorf("bigPi = %v, want %v", p, math.Pi)
	}
}
