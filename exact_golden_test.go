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
// agree: 252 of 417 cases. The other 165 (FFT at 1000, 1009, 2018, 4096,
// 4099, 16384 points, most N-D shapes, padded and truncated lengths, ...) gave
// two or three different outputs depending on the CPU, which contradicts "a
// result never depends on which CPU ran it"; they are listed in
// testdata/exact_golden_amd64_cpu_dependent.txt and filed on #121. The arm64
// file has all 417 (one NEON route). An architecture
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
	checked := 0
	for _, c := range cases {
		want, ok := golden[c.name]
		if !ok {
			continue // no digest: the case varies with the CPU (see above)
		}
		checked++
		if got := digestAny(c.exact()); got != want {
			t.Errorf("%s: Options{Exact: true} is not v0.28.16's output", c.name)
		}
	}
	if checked != len(golden) {
		t.Errorf("%d golden digests, %d matched a case: regenerate from v0.28.16", len(golden), checked)
	}
	t.Logf("%d of %d cases checked against v0.28.16", checked, len(cases))
}
