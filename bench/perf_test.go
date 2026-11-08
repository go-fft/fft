package bench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with scipy.fft (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, denominator
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run 'TestDataParity|TestValueParity' -perf.parity .   same inputs, same values (needs uv)
//
// Both sides time one call on the same arrays. Plan and setup cost is EXCLUDED on
// both sides: a Plan row builds its plan before the timed loop; a one-shot row
// (fft.FFT, scipy.fft.fft) finds its plan in the package's cache (go-fft memoizes
// plans by length, pocketfft by a 16-entry LRU) after the calibration call. The
// constructors are not timed (docs/perf-exempt.md).

var parityFlag = flag.Bool("perf.parity", false, "TestDataParity, TestValueParity: compare with scipy.fft (starts uv)")

func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func uvPath() string {
	if p, err := exec.LookPath("uv"); err == nil {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "bin", "uv")
}

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uvPath(), refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, s := range specs() {
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(s.setup, s.call, 1)
		}
		parity := refbench.ParityTolerance
		if s.tol < 0 {
			parity = refbench.ParityNone
		}
		out = append(out, refbench.Case{
			Name:      s.name,
			Func:      s.fn,
			Reference: "scipy.fft 1.18.1 " + s.call,
			Parity:    parity,
			Go:        s.run,
			Ref:       ref,
		})
	}
	return out
}

func TestPerf(t *testing.T) {
	// go-fft reads GOMAXPROCS once, when the package is initialised (parWorkers
	// in parallel.go), so refbench's own GOMAXPROCS=1 comes too late to stop the
	// N-D transforms from fanning out over every core. The process must start
	// with GOMAXPROCS=1 in its environment; a count or a timing taken without
	// it is a multi-core one (go-fft/fft issue filed with this change).
	if os.Getenv("GOMAXPROCS") != "1" {
		t.Fatal("run with GOMAXPROCS=1 in the environment: GOMAXPROCS=1 go test -run TestPerf -cpu 1 .")
	}
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-fft/fft"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

// TestDataParity checks that both sides are given the same bytes: a ratio
// between two transforms of different data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, s := range specs() {
		h := sha256.New()
		for _, in := range s.ins {
			h.Write(raw(in))
		}
		got, err := w.DataHash(s.setup, 1)
		if err != nil {
			t.Fatalf("%s: %v", s.name, err)
		}
		if want := hex.EncodeToString(h.Sum(nil)); got != want {
			t.Errorf("%s: NumPy data %s, Go data %s", s.name, got, want)
		}
	}
}

// TestValueParity checks that both sides compute the same transform: the
// largest difference from scipy.fft's result is below tol times its largest value.
func TestValueParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (runs parity.py through uv)")
	}
	type req struct {
		Name  string `json:"name"`
		Setup string `json:"setup"`
		Call  string `json:"call"`
	}
	var reqs []req
	all := specs()
	for _, s := range all {
		reqs = append(reqs, req{s.name, s.setup, s.call})
	}
	in, _ := json.Marshal(reqs)
	refs, _ := filepath.Abs("refs.txt")
	cmd := exec.Command(uvPath(), "run", "--no-project", "--python", refbench.PythonVersion,
		"--with-requirements", refs, "python", "-I", "parity.py")
	cmd.Stdin = bytes.NewReader(in)
	cmd.Env = refbench.SingleThreadEnv()
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("parity.py: %v\n%s", err, stderr.String())
	}
	var want map[string][]float64
	if err := json.Unmarshal(stdout.Bytes(), &want); err != nil {
		t.Fatal(err)
	}
	for _, s := range all {
		if s.tol < 0 {
			continue
		}
		ref, got := want[s.name], s.res()
		if len(ref) != len(got) {
			t.Errorf("%s: scipy returns %d values, Go %d", s.name, len(ref), len(got))
			continue
		}
		var maxRef, maxDiff float64
		for i := range ref {
			maxRef = math.Max(maxRef, math.Abs(ref[i]))
			maxDiff = math.Max(maxDiff, math.Abs(ref[i]-got[i]))
		}
		t.Logf("%s: %d values, max |Go - scipy| / max |scipy| = %.2e (tolerance %.0e)", s.name, len(ref), maxDiff/maxRef, s.tol)
		if !(maxRef > 0) || !(maxDiff <= s.tol*maxRef) {
			t.Errorf("%s: max |Go - scipy| = %g, tolerance %g (max |scipy| %g)", s.name, maxDiff, s.tol*maxRef, maxRef)
		}
	}
}
