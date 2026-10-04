package kernels

import "testing"

// TestRadixStagesRejectInconsistentArguments: the amd64 stage kernels trust n,
// span and the slice lengths; the dispatch checks them so that a caller
// passing inconsistent values panics instead of overrunning in assembly.
func TestRadixStagesRejectInconsistentArguments(t *testing.T) {
	a := make([]complex128, 16)
	tw := make([]complex128, 8)
	for _, c := range []struct {
		name string
		f    func()
	}{
		{"radix2 n not a multiple", func() { Radix2Stage(a, 12, 4, tw) }},
		{"radix2 a too short", func() { Radix2Stage(a[:8], 16, 4, tw) }},
		{"radix2 tw too short", func() { Radix2Stage(a, 16, 8, tw[:4]) }},
		{"radix4 n not a multiple", func() { Radix4Stage(a, 12, 2, tw, tw, tw, false) }},
		{"radix4 a too short", func() { Radix4Stage(a[:8], 16, 4, tw, tw, tw, false) }},
		{"radix4 w3 too short", func() { Radix4Stage(a, 16, 4, tw, tw, tw[:2], true) }},
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", c.name)
				}
			}()
			c.f()
		}()
	}
}
