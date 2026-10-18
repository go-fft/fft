//go:build ignore

// Command gen writes the arm64 kernels of the kernels package, via go-asmgen:
// cmul_arm64.s (the NEON pointwise complex product), stockham_arm64.s and
// stockham32_arm64.s (the float64 and float32 NEON Stockham passes),
// batch32_arm64.s, untangle32_arm64.s and untangle_arm64.s. Run with: go run
// gen.go (or `go generate` from the kernels package).
//
// The kernels themselves are built by go-asmgen's fft/arm64 package; this
// file says which kernels the package has, under which names and in which
// files. Every kernel reproduces the fused multiply-adds gc emits for its Go
// reference in this module, which the tests hold it to bit for bit.
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	fft "github.com/go-asmgen/asmgen/fft/arm64"
)

var radices = []int{2, 3, 4, 5, 8}

// inv is the suffix of the inverse kernels' symbols.
func inv(inverse bool) string {
	if inverse {
		return "Inv"
	}
	return ""
}

func main() {
	f := emit.NewFile("arm64")
	fft.CMulNEON(f, "cmulNEON")
	write("cmul_arm64.s", f)

	sk := emit.NewFile("arm64")
	for _, inverse := range []bool{false, true} {
		for _, r := range radices {
			fft.Pass(sk, fmt.Sprintf("skPass%dNEON%s", r, inv(inverse)), r, inverse)
		}
		for _, r := range radices {
			fft.Last(sk, fmt.Sprintf("skLast%dNEON%s", r, inv(inverse)), r, inverse)
		}
	}
	for _, inverse := range []bool{false, true} {
		for _, r := range radices {
			fft.Batch(sk, fmt.Sprintf("skBatch%dNEON%s", r, inv(inverse)), r, inverse)
		}
	}
	// The block-split layout: IS reads interleaved and writes split, SS
	// reads and writes split, SI (the final pass) reads split.
	for _, inverse := range []bool{false, true} {
		for _, inSplit := range []bool{false, true} {
			lay := map[bool]string{false: "IS", true: "SS"}[inSplit]
			for _, r := range radices {
				fft.PassSplit(sk, fmt.Sprintf("skPass%dNEON%s%s", r, inv(inverse), lay), r, inverse, inSplit)
			}
		}
		for _, r := range []int{2, 4, 8} {
			fft.LastSplit(sk, fmt.Sprintf("skLast%dNEON%sSI", r, inv(inverse)), r, inverse)
		}
	}
	write("stockham_arm64.s", sk)

	f = emit.NewFile("arm64")
	for _, inverse := range []bool{false, true} {
		for _, r := range radices {
			fft.Pass32(f, fmt.Sprintf("sk32Pass%dNEON%s", r, inv(inverse)), r, inverse)
		}
		for _, r := range radices {
			fft.Last32(f, fmt.Sprintf("sk32Last%dNEON%s", r, inv(inverse)), r, inverse)
		}
	}
	write("stockham32_arm64.s", f)

	f = emit.NewFile("arm64")
	fft.Untangle32(f, "f32rUntangleNEON", false)
	fft.Untangle32(f, "f32rRetangleNEON", true)
	write("untangle32_arm64.s", f)

	f = emit.NewFile("arm64")
	for _, inverse := range []bool{false, true} {
		for _, r := range radices {
			fft.Batch32(f, fmt.Sprintf("sk32Batch%dNEON%s", r, inv(inverse)), r, inverse)
		}
	}
	write("batch32_arm64.s", f)

	f = emit.NewFile("arm64")
	fft.Untangle(f, "armrealUntangleNEON", false)
	fft.Untangle(f, "armrealRetangleNEON", true)
	write("untangle_arm64.s", f)
}

func write(name string, f *emit.File) {
	if err := os.WriteFile(name, []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote", name)
}
