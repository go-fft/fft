package kernels

import "testing"

// The simulated CPUs of go-fft/fft#86. Each is a cpuFeatures value fed to
// dispatchFor, the one function every AVX-512 default goes through.
var (
	cpuKNL   = cpuFeatures{AVX2: true, AVX512F: true, Intel: true}                                  // Knights Landing/Mill: F, no DQ
	cpuKNLZ  = cpuFeatures{AVX2: true, AVX512F: true, AMD: true, AMDFamily19: true}                 // F without DQ on a Zen-4-looking vendor/family
	cpuDQnoF = cpuFeatures{AVX2: true, AVX512DQ: true, Intel: true}                                 // not a real CPU: DQ alone must not open the gate
	cpuSKX   = cpuFeatures{AVX2: true, AVX512F: true, AVX512DQ: true, Intel: true}                  // Skylake-X / Cascade Lake
	cpuZen4  = cpuFeatures{AVX2: true, AVX512F: true, AVX512DQ: true, AMD: true, AMDFamily19: true} // Zen 4: F, DQ, BW, VL
	cpuZen3  = cpuFeatures{AVX2: true, AMD: true, AMDFamily19: true}                                // Zen 3: family 0x19, no AVX-512
	cpuNone  = cpuFeatures{}
	cpuAVX2  = cpuFeatures{AVX2: true, Intel: true}
)

// TestNoAVX512KernelWithoutEverySubset: a CPU missing any subset the kernels
// execute selects no AVX-512 kernel at any dispatch site, on any vendor.
func TestNoAVX512KernelWithoutEverySubset(t *testing.T) {
	for _, c := range []struct {
		name string
		cpu  cpuFeatures
	}{
		{"F without DQ (Intel, Knights Landing)", cpuKNL},
		{"F without DQ (AMD family 0x19)", cpuKNLZ},
		{"DQ without F", cpuDQnoF},
		{"Zen 3, no AVX-512", cpuZen3},
		{"AVX2 only", cpuAVX2},
		{"nothing", cpuNone},
	} {
		d := dispatchFor(c.cpu)
		if d.Stockham512 || d.Untangle512 || d.Split512 || d.HasAVX512 || d.Zen4 {
			t.Errorf("%s: selected an AVX-512 route: %+v", c.name, d)
		}
		// The routing that follows reaches no 512-bit kernel either.
		for _, r := range []int{2, 4, 8} {
			for _, ido := range []int{1, 4, 64} {
				if w := stockhamWidth(r, ido, c.cpu.AVX2, d.Stockham512); w == 512 {
					t.Errorf("%s: stockhamWidth(%d, %d) = 512", c.name, r, ido)
				}
			}
		}
		for _, m := range splitModes512(c.cpu.AVX2, d.HasAVX512, []int{8, 8, 8}, []int{1, 8, 64}) {
			if m != 0 {
				t.Errorf("%s: splitModes512 gave a split mode %d", c.name, m)
			}
		}
		if q := clQuads(d.Untangle512, 127, 512, UntangleWideMax); q != 0 {
			t.Errorf("%s: clQuads(512) = %d four-bin AVX-512 untangle steps", c.name, q)
		}
	}
}

// TestKnightsLandingFallsBackToAVX2: F without DQ keeps the AVX2 split layout,
// exactly what an AVX2-only Intel core gets (Round 29); it does not lose its
// AVX2 route.
func TestKnightsLandingFallsBackToAVX2(t *testing.T) {
	if got, want := dispatchFor(cpuKNL), dispatchFor(cpuAVX2); got != want {
		t.Errorf("Knights Landing dispatch %+v differs from the AVX2-only Intel dispatch %+v", got, want)
	}
	if !dispatchFor(cpuKNL).Split {
		t.Error("Knights Landing lost the AVX2 split layout")
	}
}

// TestFullFeatureSetKeepsTheAVX512Routes: with F and DQ the routes are what
// they were before the gate changed, on Intel and on Zen 4.
func TestFullFeatureSetKeepsTheAVX512Routes(t *testing.T) {
	skx := dispatchFor(cpuSKX)
	if want := (dispatch{Stockham512: true, Untangle512: true, Split512: true, HasAVX512: true}); skx != want {
		t.Errorf("Skylake-X dispatch = %+v, want %+v", skx, want)
	}
	zen4 := dispatchFor(cpuZen4)
	if want := (dispatch{Stockham512: true, Zen4: true, HasAVX512: true, Split: true}); zen4 != want {
		t.Errorf("Zen 4 dispatch = %+v, want %+v (Zen 4 keeps AVX-512 and the 256-bit split routing, Round 49/51)", zen4, want)
	}
	if z3 := dispatchFor(cpuZen3); z3.Zen4 {
		t.Error("Zen 3 (family 0x19, no AVX-512) read as Zen 4")
	}
}

// TestDetectedCPUIsConsistent: the real probes agree with each other and with
// the package defaults on whatever machine runs the suite (the tests that
// change the Use* variables restore them).
func TestDetectedCPUIsConsistent(t *testing.T) {
	c := detectedCPU
	if c.AVX512DQ && !c.AVX512F {
		t.Errorf("the DQ probe is true where the F probe is false: %+v", c)
	}
	if got := dispatchFor(c); got != detectedDispatch {
		t.Errorf("detectedDispatch = %+v, want %+v", detectedDispatch, got)
	}
	if detectedDispatch.Stockham512 != (c.AVX512F && c.AVX512DQ) {
		t.Errorf("UseStockhamAVX512 default %v, F=%v DQ=%v", detectedDispatch.Stockham512, c.AVX512F, c.AVX512DQ)
	}
	t.Logf("detected: %+v", c)
}
