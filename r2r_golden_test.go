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

// Golden digests of the DCT, DST and MDCT outputs as released (r2r_cases_test.go),
// per architecture: testdata/r2r_golden_<GOARCH>.txt, and, for the cases whose
// output differs between CPUs of the architecture,
// testdata/r2r_golden_<GOARCH>_cpu_dependent.txt. Both were generated from a
// clean v0.29.1 tree: amd64 on 12 GitHub runners (AMD EPYC 7763, 9V74, 9V45,
// Intel Xeon Platinum 8370C and 6973P-C), keeping the 883 cases on which all
// twelve agree and listing the 229 that differ with the CPU; arm64 on one
// host, all 1,112. Written with -r2rgolden.write.

var r2rGoldenWrite = flag.Bool("r2rgolden.write", false, "write testdata/r2r_golden_<GOARCH>.txt from the current outputs")

func r2rGoldenPath() string { return filepath.Join("testdata", "r2r_golden_"+runtime.GOARCH+".txt") }

func readNameMap(t *testing.T, path string) map[string]string {
	t.Helper()
	f, err := os.Open(path)
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

func TestR2RGolden(t *testing.T) {
	isolatePlans(t)
	cases := r2rCases()
	if *r2rGoldenWrite {
		var lines []string
		for _, c := range cases {
			lines = append(lines, c.name+"\t"+digestAny(c.run()))
		}
		sort.Strings(lines)
		if err := os.WriteFile(r2rGoldenPath(), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d digests to %s", len(lines), r2rGoldenPath())
		return
	}
	if runtime.GOARCH == "arm64" && (raceEnabled || testing.CoverMode() != "") {
		t.Skip("arm64 digests are for the build without -race or -cover")
	}
	golden := readNameMap(t, r2rGoldenPath())
	if golden == nil {
		t.Skipf("no golden digests for %s", runtime.GOARCH)
	}
	dep := readNameMap(t, strings.TrimSuffix(r2rGoldenPath(), ".txt")+"_cpu_dependent.txt")
	checked, skipped := 0, 0
	for _, c := range cases {
		want, ok := golden[c.name]
		if !ok {
			if _, d := dep[c.name]; !d {
				t.Errorf("%s: in neither the golden file nor the CPU-dependent list", c.name)
				continue
			}
			skipped++
			t.Logf("skip %s: its released output depends on the CPU", c.name)
			continue
		}
		checked++
		if got := digestAny(c.run()); got != want {
			t.Errorf("%s: output differs from the released one (got %s)", c.name, got)
		}
	}
	if checked == 0 {
		t.Errorf("none of the %d golden digests matched a case", len(golden))
	}
	t.Logf("%d of %d cases checked, %d skipped as CPU-dependent", checked, len(cases), skipped)
}
