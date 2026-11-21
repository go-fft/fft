package fft

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

// TestDCT23Generated: the committed internal/kernels/dct23_amd64.s is what
// internal/gendct23 writes (the generator also verifies its dataflow graph against a
// naive DCT before writing anything).
func TestDCT23Generated(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the generator")
	}
	switch runtime.GOARCH {
	case "s390x", "ppc64le", "loong64", "riscv64":
		t.Skip("the generator is slow under qemu; the native lanes check the file")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go command")
	}
	if _, err := os.Stat("internal/gendct23/main.go"); err != nil {
		t.Skip("generator sources are not alongside the test binary")
	}
	dir := t.TempDir()
	if b, err := exec.Command(goBin, "run", "internal/gendct23/main.go", dir).CombinedOutput(); err != nil {
		t.Fatalf("generator: %v\n%s", err, b)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) == 0 {
		t.Fatalf("the generator wrote nothing: %v", err)
	}
	for _, e := range entries {
		got, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		want, err := os.ReadFile(filepath.Join("internal/kernels", e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		want = bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
		if !bytes.Equal(got, want) {
			t.Errorf("internal/kernels/%s is stale: run go generate and commit", e.Name())
		}
	}
}
