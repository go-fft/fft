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

// TestCascadeLakeDefault: the fused 256- and 2048-point kernels' Intel gate is
// the CPUID signature (family 6, model 0x55, stepping 5, 6 or 7) with AVX-512
// and the Intel vendor, and nothing else on Intel: Cooper Lake (steppings 10
// and 11), the other steppings of model 0x55 (0 to 4, 8, 9, 12 to 15), Ice
// Lake, Sapphire Rapids, an AVX2-only Haswell, Zen 4.
func TestCascadeLakeDefault(t *testing.T) {
	for _, c := range []struct {
		name   string
		avx512 bool
		intel  bool
		sig    uint32
		want   bool
	}{
		{"Cascade Lake stepping 5 (the measured host)", true, true, 0x50655, true},
		{"Cascade Lake stepping 6", true, true, 0x50656, true},
		{"Cascade Lake stepping 7", true, true, 0x50657, true},
		{"Cooper Lake stepping 10 (unmeasured)", true, true, 0x5065A, false},
		{"Cooper Lake stepping 11 (unmeasured)", true, true, 0x5065B, false},
		{"model 0x55 stepping 8", true, true, 0x50658, false},
		{"model 0x55 stepping 9", true, true, 0x50659, false},
		{"model 0x55 stepping 12", true, true, 0x5065C, false},
		{"model 0x55 stepping 13", true, true, 0x5065D, false},
		{"model 0x55 stepping 14", true, true, 0x5065E, false},
		{"model 0x55 stepping 15", true, true, 0x5065F, false},
		{"Skylake-SP stepping 4", true, true, 0x50654, false},
		{"Skylake-SP stepping 3", true, true, 0x50653, false},
		{"Skylake-SP stepping 0", true, true, 0x50650, false},
		{"Ice Lake-SP", true, true, 0x606A6, false},
		{"Ice Lake client", true, true, 0x706E5, false},
		{"Sapphire Rapids", true, true, 0x806F8, false},
		{"Haswell, no AVX-512", false, true, 0x306C3, false},
		{"model 0x55 without AVX-512", false, true, 0x50655, false},
		{"model 0x55 reported by a non-Intel vendor", true, false, 0x50655, false},
		{"Zen 4", true, false, 0xA10F11, false},
		{"family 7 with the model nibbles of 0x55", true, true, 0x50755, false},
		{"extended family set", true, true, 0x01050655, false},
	} {
		if got := cascadeLakeDefault(c.avx512, c.intel, c.sig); got != c.want {
			t.Errorf("%s (%#x): %v, want %v", c.name, c.sig, got, c.want)
		}
	}
	if d := dispatchFor(cpuFeatures{AVX2: true, AVX512F: true, AVX512DQ: true, Intel: true, Signature: 0x50655}); !d.CascadeLake {
		t.Error("a Cascade Lake signature with F and DQ did not set CascadeLake")
	}
	if d := dispatchFor(cpuFeatures{AVX2: true, AVX512F: true, Intel: true, Signature: 0x50655}); d.CascadeLake {
		t.Error("F without DQ set CascadeLake")
	}
	if dispatchFor(cpuZen4).CascadeLake || dispatchFor(cpuSKX).CascadeLake {
		t.Error("a CPU without a signature set CascadeLake")
	}
}

// TestFusedGates: the fused kernels' gates on this machine follow the rules above:
// 256 and 2048 on a Zen 4 or a Cascade Lake with AVX-512, 1024 on a Zen 4 only.
func TestFusedGates(t *testing.T) {
	if got, want := IntelCascadeLake, cascadeLakeDefault(detectedCPU.AVX512F && detectedCPU.AVX512DQ, detectedCPU.Intel, detectedCPU.Signature); got != want {
		t.Errorf("IntelCascadeLake = %v, want %v (signature %#x)", got, want, detectedCPU.Signature)
	}
	if want := UseStockhamAVX512 && (AMDZen4 || IntelCascadeLake); Fused256On != want || Fused2048On != want {
		t.Errorf("Fused256On = %v, Fused2048On = %v, want %v", Fused256On, Fused2048On, want)
	}
	if want := UseStockhamAVX512 && AMDZen4; Fused1024On != want {
		t.Errorf("Fused1024On = %v, want %v", Fused1024On, want)
	}
	if want := UseStockhamAVX512 && AMDZen4; Dct23On != want {
		t.Errorf("Dct23On = %v, want %v", Dct23On, want)
	}
	t.Logf("signature %#x, cascade lake %v", detectedCPU.Signature, IntelCascadeLake)
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
