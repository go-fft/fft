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
// host, all 1,112. Written with -r2rgolden.write. Since the DCT-IV and MDCT
// twiddle passes became go-simd/floats kernels, 67 more amd64 cases (DCT, DST and
// their plans at n = 257, MDCT and IMDCT at 2,048) are listed as CPU-dependent: a
// qemu run, the CI runner and Rosetta gave three digests, so one machine's digest
// would fail on another route. Every skipped case is named in the -v output, and
// the golden and CPU-dependent files together must name every case.
// The Intel AVX2 window uses the separate fold/gather at MDCT n=256. Its two
// FMA digests override the common file; the other CPUs retain their old goldens.

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
	intelWindow := runtime.GOARCH == "amd64" && mdctFoldMin <= 256 && mdctFoldMax >= 256
	const intelPath = "testdata/r2r_golden_amd64_intel_avx2.txt"
	if *r2rGoldenWrite {
		sf, sfx, su, sux := mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax
		defer func() { mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = sf, sfx, su, sux }()
		if intelWindow {
			// Keep the common file on the original window for the other CPUs.
			mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = 512, 4096, 256, 4096
		}
		var lines []string
		for _, c := range cases {
			lines = append(lines, c.name+"\t"+digestAny(c.run()))
		}
		sort.Strings(lines)
		if err := os.WriteFile(r2rGoldenPath(), []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %d digests to %s", len(lines), r2rGoldenPath())
		if intelWindow {
			mdctFoldMin, mdctFoldMax, mdctUnfoldMin, mdctUnfoldMax = sf, sfx, su, sux
			lines = nil
			for _, c := range cases {
				if strings.HasPrefix(c.name, "MDCT/") && strings.HasSuffix(c.name, "/n=256") {
					lines = append(lines, c.name+"\t"+digestAny(c.run()))
				}
			}
			sort.Strings(lines)
			if err := os.WriteFile(intelPath, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		return
	}
	if runtime.GOARCH == "arm64" && (raceEnabled || testing.CoverMode() != "") {
		t.Skip("arm64 digests are for the build without -race or -cover")
	}
	if runtime.GOARCH == "amd64" && !fmaAvailable {
		// The DCT, DST and MDCT run complex plans of the default arithmetic, which
		// fuses where the CPU has FMA3: the amd64 digests are those of CPUs that
		// have it (every CI runner does).
		t.Skip("the amd64 digests are for CPUs with FMA3")
	}
	golden := readNameMap(t, r2rGoldenPath())
	if golden == nil {
		t.Skipf("no golden digests for %s", runtime.GOARCH)
	}
	if intelWindow {
		variant := readNameMap(t, intelPath)
		if variant == nil {
			t.Fatal("missing Intel AVX2 MDCT goldens")
		}
		for name, digest := range variant {
			if _, ok := golden[name]; !ok {
				t.Fatalf("Intel AVX2 golden %s has no common case", name)
			}
			golden[name] = digest
		}
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

// TestR2RCPUDependentFloor: the CPU-dependent list can only grow. A golden
// regeneration from a few runners finds the cases whose output differs between
// THOSE runners; the route-dependent cases the sample happened to agree on fall
// out of the list and get pinned to whichever runner it was (#144, then #155,
// both silently). testdata/r2r_known_cpu_dependent_<GOARCH>.txt is the floor:
// every name in it must stay in the CPU-dependent list. Shrinking the floor is a
// deliberate edit of that file in the same diff.
func TestR2RCPUDependentFloor(t *testing.T) {
	floor := readNames(t, filepath.Join("testdata", "r2r_known_cpu_dependent_"+runtime.GOARCH+".txt"))
	if floor == nil {
		t.Skipf("no floor for %s", runtime.GOARCH)
	}
	dep := readNameMap(t, strings.TrimSuffix(r2rGoldenPath(), ".txt")+"_cpu_dependent.txt")
	golden := readNameMap(t, r2rGoldenPath())
	for n := range floor {
		if _, ok := dep[n]; !ok {
			t.Errorf("%s is known to depend on the CPU and is missing from the CPU-dependent list", n)
		}
		if _, ok := golden[n]; ok {
			t.Errorf("%s is known to depend on the CPU and is pinned in the golden file", n)
		}
	}
	if len(floor) < 100 {
		t.Errorf("the floor holds %d names: it was emptied", len(floor))
	}
}

func readNames(t *testing.T, path string) map[string]bool {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	m := map[string]bool{}
	for _, l := range strings.Split(string(b), "\n") {
		if l = strings.TrimRight(l, "\r"); l != "" { // Windows checkouts may use CRLF
			m[l] = true
		}
	}
	return m
}
