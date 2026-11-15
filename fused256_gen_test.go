package fft

import (
	"bytes"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fused256Bound is the c2c accuracy bound at 256 points (c·ε·log2 n).
func fused256Bound() float64 { return accBound["c2c"] * eps64 * math.Log2(256) }

// TestFused256RewritePresent: the committed body has the subtract-to-FMA
// rewrite (fma(b, -1, a), 96 per direction) and nothing else of the kind (no
// fused add, no 132 form: the generator has no branch that emits them), and the
// reference body has none, so a generator that silently stopped rewriting, or a
// reference that started to, fails here rather than in a timing.
func TestFused256RewritePresent(t *testing.T) {
	b, err := os.ReadFile("internal/kernels/fused256_amd64.s")
	if err != nil {
		t.Skip("the generated assembly is not alongside the test binary")
	}
	s := string(b)
	i := strings.Index(s, "TEXT ·fused256ref(SB)")
	if i < 0 {
		t.Fatal("no reference body")
	}
	count := func(text string) int {
		n := 0
		for _, l := range strings.Split(text, "\n") {
			if strings.Contains(l, "VFMADD231PD.BCST") && strings.Contains(l, "f256_bff0000000000000") {
				n++
			}
		}
		return n
	}
	if n := count(s[:i]); n != 192 {
		t.Errorf("the body has %d subtract-to-FMA rewrites, want 192 (96 per direction)", n)
	}
	if strings.Contains(s, "VFMADD132PD") || strings.Contains(s, "VFMADD231PD.BCST f256_3ff0000000000000") {
		t.Error("a fused add or a 132-form rewrite is in the assembly")
	}
	if n := count(s[i:]); n != 0 {
		t.Errorf("the reference body has %d rewrites", n)
	}
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
	for _, name := range []string{"fused256_amd64.s", "fused256_tw_amd64.go", "fused1024_amd64.s", "fused1024_tw_amd64.go", "fused2048_amd64.s", "fused2048_tw_amd64.go"} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join("internal/kernels", name))
		if err != nil {
			t.Fatal(err)
		}
		// A Windows checkout may have turned the committed file's LF into CRLF.
		want = bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
		if !bytes.Equal(got, want) {
			t.Errorf("internal/kernels/%s is stale: run go generate and commit", name)
		}
	}
}
