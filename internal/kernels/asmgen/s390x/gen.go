//go:build ignore

// Command gen produces cmul_s390x.s, the z/Architecture vector-facility pointwise complex multiply
// kernel, via go-asmgen (fft/s390x, where the kernel and the account of its
// bit identity with CMulScalar live). Run with: go run gen.go (or `go
// generate` from the kernels package).
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	fft "github.com/go-asmgen/asmgen/fft/s390x"
)

func main() {
	f := emit.NewFile("s390x")
	fft.CMulVX(f, "cmulVX")
	if err := os.WriteFile("cmul_s390x.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote cmul_s390x.s")
}
