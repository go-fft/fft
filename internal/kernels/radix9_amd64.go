package kernels

// The radix-9 pass kernels (Round 35): csGenRadix9File in asmgen/amd64/gen.go
// writes radix9_amd64.s. They run the fft package's pass9 and pass9last
// (radix9.go there), operation for operation, at 256 bits.

// The radix-9 butterfly's internal twiddles W9^e (θ = 2π·e/9), the values
// of the fft package's radix-9 constants.
const (
	cs9C1 = 0.76604444311897803520239265055541667393583245708040 // cos(2π/9)
	cs9S1 = 0.64278760968653932632264340990726343290755988420568 // sin(2π/9)
	cs9C2 = 0.17364817766693034885171662676931479600037567718407 // cos(4π/9)
	cs9S2 = 0.98480775301220805936674302458952301367064325171984 // sin(4π/9)
	cs9C4 = -0.9396926207859083840541092773247314699362081342645 // cos(8π/9)
	cs9S4 = 0.34202014332566873304409961468225958076308336751416 // sin(8π/9)
)

//go:noescape
func csPass9AVX2(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func csLast9AVX2(cc, ch *complex128, k *float64, l1 int)
