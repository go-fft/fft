package kernels

// The radix-32 butterfly's constants (Round 42), shared by the amd64 kernels'
// tables and the fft package's Go reference (bfly32FMA), so that the two
// read the same float64 values.

// R32Cos[k] is cos(kπ/16), k = 0 .. 8.
var R32Cos = [9]float64{
	1,
	0.98078528040323044912618223613423903697393373089333, // cos(π/16)
	0.92387953251128675612818318939678828682241662586364, // cos(π/8)
	0.83146961230254523707878837761790575673856081757400, // cos(3π/16)
	0.70710678118654752440084436210484903928483593768847, // cos(π/4)
	0.55557023301960222474283081394853287437493719075480, // cos(5π/16)
	0.38268343236508977172845998403039886676134456248563, // cos(3π/8)
	0.19509032201612826784828486847702224092769161775195, // cos(7π/16)
	0,
}

// r32Es are the exponents e = n2·k1 (1 <= n2 <= 3, 1 <= k1 <= 7) of the
// radix-32 butterfly's internal twiddles other than W32^8 = s·i, in the order
// of their rows in the interleaved constants table.
var r32Es = [...]int{1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21}

// R32Trig returns cos θ and sin θ, θ = 2π·e/32, for 0 <= e < 24, from
// R32Cos by the exact symmetries: the internal twiddle W32^e is cos θ +
// s·i·sin θ (s = −1 forward, +1 inverse).
func R32Trig(e int) (c, s float64) {
	switch {
	case e <= 8:
		return R32Cos[e], R32Cos[8-e]
	case e <= 16:
		return -R32Cos[16-e], R32Cos[e-8]
	}
	return -R32Cos[e-16], -R32Cos[24-e]
}
