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
// check to the new bound and keeps this file for Exact. An architecture
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
	golden := readGolden(t)
	if golden == nil {
		t.Skipf("no golden digests for %s", runtime.GOARCH)
	}
	if len(golden) != len(cases) {
		t.Errorf("%d golden digests for %d cases: regenerate from v0.28.16", len(golden), len(cases))
	}
	for _, c := range cases {
		want, ok := golden[c.name]
		if !ok {
			t.Errorf("%s: no golden digest", c.name)
			continue
		}
		if got := digestAny(c.exact()); got != want {
			t.Errorf("%s: Options{Exact: true} is not v0.28.16's output", c.name)
		}
	}
}
