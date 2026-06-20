package kernels

import "testing"

// skipIfNoSIMD skips the calling SIMD-vs-scalar test when the running riscv64
// CPU lacks the V (vector) extension. There, cmulSIMD falls back to scalar and
// calling the RVV kernel directly would SIGILL, so the comparison is skipped
// with an explicit, logged reason (t.Skip) rather than faked as a trivial
// scalar==scalar pass. On real RVV hardware (cfarm95, RVV 1.0) haveRVV is true,
// so this returns without skipping and the comparison RUNS and proves
// bit-identity. This is exactly what keeps CI green: under the non-V
// qemu-riscv64 the test logs "no RVV" and skips; on cfarm95 it runs and passes.
func skipIfNoSIMD(t *testing.T) {
	if !haveRVV {
		t.Skip("no RVV (V extension absent on this riscv64 CPU); RVV kernel is hardware-validated on cfarm95")
	}
}
