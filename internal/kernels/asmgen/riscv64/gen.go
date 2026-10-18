//go:build ignore

// Command gen produces cmul_riscv64.s, the RISC-V Vector (RVV 1.0) pointwise complex multiply
// kernel, via go-asmgen (fft/riscv64, where the kernel and the account of its
// bit identity with CMulScalar live). Run with: go run gen.go (or `go
// generate` from the kernels package).
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	fft "github.com/go-asmgen/asmgen/fft/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	fft.CMulRVV(f, "cmulRVV")
	if err := os.WriteFile("cmul_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmul_riscv64.s")
}
