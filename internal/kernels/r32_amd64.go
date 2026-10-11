package kernels

// The radix-32 pass kernels (Round 38): genR32File in asmgen/amd64/gen.go
// writes r32_amd64.s. They are the fft package's pass32 and pass32last
// (radix32.go), operation for operation; StockhamPass runs them as it runs
// every other AVX2 radix, with StockhamTwiddles' table (not radix 16's
// pre-duplicated one) and skConst's radix-32 rows.

//go:noescape
func skPass32AVX2(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skLast32AVX2(cc, ch *complex128, k *float64, l1 int)

// r32Es are the exponents e of the radix-32 butterfly's general internal
// twiddles W32^e (e = n2·k1, n2 in 1..3, k1 in 1..7, without the exact e =
// 8), in the order of their rows in skConst: the same list as gen.go's.
var r32Es = [...]int{1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21}

// r32CosK is cos(k·π/16) for k = 0..8, the fft package's table.
var r32CosK = [9]float64{
	1,
	0.98078528040323044912618223613423903697393373089334, // cos(π/16)
	0.92387953251128675612818318939678828682241662586364, // cos(π/8)
	0.83146961230254523707878837761790575673856081198725, // cos(3π/16)
	0.707106781186547524400844362104849,                  // cos(π/4)
	0.55557023301960222474283081394853287437493719075480, // cos(5π/16)
	0.38268343236508977172845998403039886676134456248563, // cos(3π/8)
	0.19509032201612826784828486847702224092769161775195, // cos(7π/16)
	0,
}

// r32Trig returns cos θ and sin θ for θ = π·e/16, 0 <= e < 32, each exactly
// a value of r32CosK or its negation, as the fft package's r32Trig does.
func r32Trig(e int) (c, s float64) { return r32Cos(e), r32Cos((e + 24) % 32) }

func r32Cos(e int) float64 {
	switch {
	case e <= 8:
		return r32CosK[e]
	case e <= 16:
		return -r32CosK[16-e]
	case e <= 24:
		return -r32CosK[e-16]
	}
	return r32CosK[32-e]
}
