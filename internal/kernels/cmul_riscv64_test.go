package kernels

import "testing"

// TestISAHasV pins the /proc/cpuinfo isa-string parser that gates the RVV
// kernel. A FALSE POSITIVE here is a SIGILL on a non-V CPU (e.g. CI's qemu), so
// the classification of the real strings matters for soundness.
func TestISAHasV(t *testing.T) {
	cases := []struct {
		name string
		isa  string
		want bool
	}{
		// cfarm95 (SpacemiT X60): base string ends in ...dcv, plus zve64*.
		{"cfarm95", "rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt", true},
		{"base-v-only", "rv64imafdcv", true},
		{"zve64d-only", "rv64imafdc_zve64d", true},
		{"zve64x-only", "rv64imac_zve64x", true},
		// No vector: the common CI qemu-riscv64 default CPU.
		{"no-vector", "rv64imafdc", false},
		{"gc-only", "rv64gc", false},
		{"no-vector-with-zb", "rv64imafdc_zba_zbb_zbs", false},
		// zve32* alone is 32-bit-element only (no e64 double vectors); the base
		// string has no trailing v, so it must be rejected.
		{"zve32-only", "rv64imafdc_zve32f_zve32x", false},
		{"empty", "", false},
	}
	for _, c := range cases {
		if got := isaHasV(c.isa); got != c.want {
			t.Errorf("%s: isaHasV(%q) = %v, want %v", c.name, c.isa, got, c.want)
		}
	}
}
