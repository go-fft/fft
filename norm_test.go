package fft

import (
	"math"
	"strings"
	"testing"
)

func TestNormString(t *testing.T) {
	for m, want := range map[Norm]string{
		NormBackward: "backward", NormOrtho: "ortho", NormForward: "forward", Norm(7): "Norm(7)",
	} {
		if got := m.String(); got != want {
			t.Errorf("Norm(%d).String() = %q, want %q", int(m), got, want)
		}
	}
	if Norm(0) != NormBackward {
		t.Error("the zero Norm must be NormBackward, numpy's default")
	}
}

func TestNormScale(t *testing.T) {
	const n = 16
	cases := []struct {
		m                Norm
		forward, inverse float64
	}{
		{NormBackward, 1, 1.0 / n},
		{NormOrtho, 0.25, 0.25},
		{NormForward, 1.0 / n, 1},
	}
	for _, c := range cases {
		if got := c.m.scale(n, false); got != c.forward {
			t.Errorf("%v forward scale = %v, want %v", c.m, got, c.forward)
		}
		if got := c.m.scale(n, true); got != c.inverse {
			t.Errorf("%v inverse scale = %v, want %v", c.m, got, c.inverse)
		}
		// A round trip under any mode is scaled by 1/n overall.
		if got := c.m.scale(n, false) * c.m.scale(n, true); math.Abs(got-1.0/n) > 1e-15 {
			t.Errorf("%v round-trip scale = %v, want 1/n", c.m, got)
		}
		if got := c.m.scale(0, true); got != 1 {
			t.Errorf("%v scale for n=0 = %v, want 1", c.m, got)
		}
	}
}

func TestNormScaleUnknownPanics(t *testing.T) {
	for _, m := range []Norm{-1, 3} {
		func() {
			defer func() {
				r := recover()
				if s, ok := r.(string); !ok || !strings.HasPrefix(s, "fft: unknown Norm") {
					t.Errorf("Norm(%d).scale: recovered %v, want an fft: panic", int(m), r)
				}
			}()
			m.scale(8, false)
		}()
	}
}
