package gentwiddle

import (
	"math"
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

// bigPi is π: 3.14159265358979323846..., rounded to float64 it is math.Pi.
func TestBigPi(t *testing.T) {
	if p, _ := bigPi().Float64(); p != math.Pi {
		t.Errorf("bigPi = %v, want %v", p, math.Pi)
	}
}
