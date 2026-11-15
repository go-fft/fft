package fft

import (
	"bytes"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// fused256Bound is the c2c accuracy bound at 256 points (c·ε·log2 n).
func fused256Bound() float64 { return accBound["c2c"] * eps64 * math.Log2(256) }

// rmsRel is the relative RMS distance between two results.
func rmsRel(a, b []complex128) float64 {
	var num, den float64
	for i := range a {
		d := a[i] - b[i]
		num += real(d)*real(d) + imag(d)*imag(d)
		den += real(b[i])*real(b[i]) + imag(b[i])*imag(b[i])
	}
	return math.Sqrt(num / den)
}

// TestFused256Generated: the committed assembly is what internal/genfused256
// writes, on every platform (CI also regenerates in place and diffs).
func TestFused256Generated(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the generator")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go command")
	}
	if _, err := os.Stat("internal/genfused256/main.go"); err != nil {
		t.Skip("generator sources are not alongside the test binary")
	}
	dir := t.TempDir()
	if b, err := exec.Command(goBin, "run", "internal/genfused256/main.go", dir).CombinedOutput(); err != nil {
		t.Fatalf("generator: %v\n%s", err, b)
	}
	for _, name := range []string{"fused256_amd64.s", "fused256_tw_amd64.go"} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join("internal/kernels", name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("internal/kernels/%s is stale: run go generate and commit", name)
		}
	}
}
