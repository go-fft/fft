package kernels

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// Which AVX-512 subset each instruction the amd64 kernels execute needs
// (go-fft/fft#86). Sources: Intel SDM vol. 2 (the "CPUID Flag" line of each
// instruction) and Go's cmd/asm testdata avx512enc/avx512f.s, avx512dq.s,
// avx512bw.s, which group the encodings by subset.
var avx512Subset = map[string]string{
	// AVX512F
	"VADDPD": "F", "VSUBPD": "F", "VMULPD": "F", "VBLENDMPD": "F",
	"VBROADCASTF64X4": "F", "VFMADD231PD": "F", "VFMADDSUB231PD": "F",
	"VFMSUB231PD": "F", "VFNMADD231PD": "F", "VINSERTF32X4": "F",
	"VMOVAPD": "F", "VMOVDDUP": "F", "VMOVUPD": "F", "VPERMILPD": "F",
	"VPORQ": "F", "VPXORQ": "F", "VSHUFF64X2": "F", "VUNPCKHPD": "F",
	"VUNPCKLPD": "F",
	// AVX512DQ: the opmask move from a general register, and the bitwise
	// floating-point ops, which exist on ZMM only with DQ.
	"KMOVB": "DQ", "VXORPD": "DQ", "VORPD": "DQ", "VANDPD": "DQ", "VANDNPD": "DQ",
}

// avx512EVEXOnly lists the mnemonics with no VEX form: on an X or Y operand
// they are AVX512VL.
var avx512EVEXOnly = map[string]bool{
	"VBLENDMPD": true, "VPORQ": true, "VPXORQ": true, "VSHUFF64X2": true,
	"VINSERTF32X4": true, "VBROADCASTF64X4": true, "KMOVB": false,
}

// gatedSubsets is what dispatchFor requires (cpuFeatures.avx512): every subset
// an AVX-512 kernel executes must be in it.
var gatedSubsets = map[string]bool{"F": true, "DQ": true}

var (
	asmLine   = regexp.MustCompile(`^\s*(?:[A-Za-z0-9_]+:\s*)?([A-Z][A-Z0-9]+)\s+(.*)$`)
	asmZmm    = regexp.MustCompile(`\bZ\d+\b`)
	asmOpmask = regexp.MustCompile(`\bK[0-7]\b|\{K[0-7]\}|\.Z\b`)
	asmHigh   = regexp.MustCompile(`\b[XY](1[6-9]|2\d|3[01])\b`)
	asmXY     = regexp.MustCompile(`\b[XY]\d+\b`)
)

// TestAVX512KernelsUseOnlyFandDQ scans every amd64 assembly file, classifies
// each AVX-512 instruction by the subset it needs, and fails on one whose
// subset the dispatch gate does not require (or that is not classified at
// all): a kernel that executes BW or VL instructions must extend the gate
// first, or a CPU with F and DQ only would fault on it.
func TestAVX512KernelsUseOnlyFandDQ(t *testing.T) {
	files, err := filepath.Glob("*_amd64.s")
	if err != nil || len(files) == 0 {
		t.Fatalf("no amd64 assembly found (err %v)", err)
	}
	used := map[string]map[string]bool{} // subset -> mnemonics
	note := func(sub, mn, where string) {
		if used[sub] == nil {
			used[sub] = map[string]bool{}
		}
		used[sub][mn] = true
		if !gatedSubsets[sub] {
			t.Errorf("%s: %s needs AVX512%s, which the dispatch gate does not require", where, mn, sub)
		}
	}
	for _, f := range files {
		fh, err := os.Open(f)
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(fh)
		n := 0
		for sc.Scan() {
			n++
			text := sc.Text()
			if i := strings.Index(text, "//"); i >= 0 {
				text = text[:i]
			}
			m := asmLine.FindStringSubmatch(text)
			if m == nil {
				continue
			}
			mn, ops, where := m[1], m[2], f+":"+strconv.Itoa(n)
			zmm, mask, high := asmZmm.MatchString(ops), asmOpmask.MatchString(ops), asmHigh.MatchString(ops)
			xy := asmXY.MatchString(ops) && !zmm
			evex := zmm || mask || high || mn == "KMOVB"
			if !evex && !(xy && avx512EVEXOnly[mn]) {
				continue // VEX or legacy: AVX2's business
			}
			sub, ok := avx512Subset[mn]
			if !ok {
				t.Errorf("%s: %s is an AVX-512 instruction with no subset classification", where, mn)
				continue
			}
			note(sub, mn, where)
			if xy && (mask || high || avx512EVEXOnly[mn]) {
				note("VL", mn, where) // a 128/256-bit EVEX form
			}
		}
		fh.Close()
	}
	if len(used["F"]) == 0 || len(used["DQ"]) == 0 {
		t.Errorf("scan found F=%d DQ=%d mnemonics: the scanner has stopped seeing the kernels", len(used["F"]), len(used["DQ"]))
	}
	var dq []string
	for mn := range used["DQ"] {
		dq = append(dq, mn)
	}
	sort.Strings(dq)
	t.Logf("AVX512DQ instructions used: %v; F: %d mnemonics", dq, len(used["F"]))
}
