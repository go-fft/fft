package fft

import (
	"bufio"
	"flag"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// Golden digests: the SHA-256 of the bits every exactCases transform returned
// at v0.28.16, the last release whose default arithmetic was the reference one.
// They are per architecture (testdata/exact_golden_<GOARCH>.txt) because gc
// fuses multiply-adds on arm64: the amd64 file was generated on amd64 and the
// arm64 file on arm64, each from a clean v0.28.16 tree running the default
// calls (-golden.write; the copy of exactCases there defines exact(o) as o).
// Options.Exact must reproduce them; so must the default until a release
// changes it on purpose (#121 step 3), whose PR then moves the default's
// check to the new bound and keeps this file for Exact.
//
// v0.28.16's default is NOT the same on every amd64 CPU. The amd64 digests
// were generated on 12 GitHub runners (AMD EPYC 7763 / Zen 3, 9V74 / Zen 4,
// 9V45 / Zen 5, Intel Xeon Platinum 8573C) and kept only where all twelve
// agree: 269 of 458 cases (the 41 FFTReal and FFTRealWith cases were added later,
// from the same v0.28.16 tree and the same twelve runners). The other 189 (FFT at 1000, 1009, 2018, 4096,
// 4099, 16384 points, most N-D shapes, padded and truncated lengths, ...) gave
// two or three different outputs depending on the CPU, which contradicts "a
// result never depends on which CPU ran it"; they are listed in
// testdata/exact_golden_amd64_cpu_dependent.txt and filed on #121. The arm64
// file has all 458 (one NEON route). An architecture
// without a file (s390x, ppc64le, riscv64, loong64, 386) is skipped.

var goldenWrite = flag.Bool("golden.write", false, "write testdata/exact_golden_<GOARCH>.txt from the default calls (run against v0.28.16)")

func goldenPath() string { return filepath.Join("testdata", "exact_golden_"+runtime.GOARCH+".txt") }

func readGolden(t *testing.T) map[string]string {
	t.Helper()
	f, err := os.Open(goldenPath())
	if err != nil {
		return nil
	}
	defer f.Close()
	m := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if name, digest, ok := strings.Cut(sc.Text(), "\t"); ok {
			m[name] = digest
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestExactGolden(t *testing.T) {
	cases := exactCases()
	if *goldenWrite {
		var lines []string
		for _, c := range cases {
			lines = append(lines, c.name+"\t"+digestAny(c.def()))
		}
		sort.Strings(lines)
		if err := os.WriteFile(goldenPath(), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d digests to %s", len(lines), goldenPath())
		return
	}
	if runtime.GOARCH == "arm64" && (raceEnabled || testing.CoverMode() != "") {
		// gc fuses multiply-adds differently when it instruments the code
		// (round36_test.go saw an ulp on macOS arm64 under -race and -coverpkg):
		// the arm64 digests are those of the plain build.
		t.Skip("arm64 digests are for the build without -race or -cover")
	}
	golden := readGolden(t)
	if golden == nil {
		t.Skipf("no golden digests for %s", runtime.GOARCH)
	}
	dep := readCPUDependent(t)
	for name := range dep {
		if _, both := golden[name]; both {
			t.Errorf("%s is in the golden file and in the CPU-dependent list", name)
		}
	}
	checked, skipped := 0, 0
	for _, c := range cases {
		want, ok := golden[c.name]
		if !ok {
			if !dep[c.name] {
				t.Errorf("%s: in neither the golden file nor the CPU-dependent list: regenerate from v0.28.16", c.name)
				continue
			}
			skipped++
			t.Logf("skip %s: its output at v0.28.16 depends on the CPU", c.name)
			continue
		}
		checked++
		if got := digestAny(c.exact()); got != want {
			t.Errorf("%s: Options{Exact: true} is not v0.28.16's output", c.name)
		}
	}
	if checked == 0 {
		t.Errorf("none of the %d golden digests matched a case: regenerate from v0.28.16", len(golden))
	}
	// The full case set is exactly the golden digests plus the CPU-dependent
	// names; a shortened one (-race, qemu) is a subset of it.
	if len(cases) == len(golden)+len(dep) && checked+skipped != len(cases) {
		t.Errorf("%d checked + %d skipped != %d cases", checked, skipped, len(cases))
	}
	if len(cases) > len(golden)+len(dep) {
		t.Errorf("%d cases but only %d golden and %d CPU-dependent names", len(cases), len(golden), len(dep))
	}
	t.Logf("%d of %d cases checked against v0.28.16, %d skipped as CPU-dependent", checked, len(cases), skipped)
}

// readCPUDependent reads the names of the cases whose v0.28.16 output varied
// between CPUs of the architecture (testdata/exact_golden_<GOARCH>_cpu_dependent.txt).
func readCPUDependent(t *testing.T) map[string]bool {
	t.Helper()
	m := map[string]bool{}
	f, err := os.Open(strings.TrimSuffix(goldenPath(), ".txt") + "_cpu_dependent.txt")
	if err != nil {
		return m
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if name, _, ok := strings.Cut(sc.Text(), "\t"); ok {
			m[name] = true
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return m
}
