package fft

import "github.com/go-fft/fft/internal/kernels"

// fmaSplitDefault: every amd64 split kernel has an FMA twin (fmasplit*.s).
func fmaSplitDefault() bool { return true }

// znFMABind is the frame plans' binder for an FMA plan: the FMA twins of the
// kernels of the stages that fuse (skStage.fma), the default kernels of the
// others (kernels.ZnPlanForWideFMA's per-pass flags).
func znFMABind(st []skStage) znBinder {
	f := make([]bool, len(st))
	for i := range st {
		f[i] = st[i].fma
	}
	return func(r []int, mode []uint8, tw0, tw0c, tw1, tw1c []complex128, wide bool) *kernels.ZnPlan {
		return kernels.ZnPlanForWideFMA(r, mode, tw0, tw0c, tw1, tw1c, wide, f)
	}
}

// fmaTableDefault is fmaTableAMD64 for this machine.
func fmaTableDefault() map[int]fmaChoice {
	return zen4FMATable(fmaTableAMD64(kernels.UseStockhamSplit512, kernels.UseStockhamSplit && !kernels.IntelCPU), kernels.AMDZen4)
}

// zen4FMATable is znFMATable's table on a Zen 4: complex 1024 takes the
// two-pass radix-32 plan (32·32, r32fma_amd64.s). Measured on a Ryzen 9 7950X,
// one pinned core, fifteen rotated rounds (#121): 903 ns against 913 for the
// FMA plan of the default factorization (8·8·16) and 959 unfused; 512 stays 8·8·8
// (353 ns against 418 for 32·16, 437 for 16·32, 366 for 4·8·16) and 2048 as
// 32·8·8 is 4.8 times slower, so only 1024 changes. Zen 3 measured radix 32
// slower (0.96 times) and keeps znFMATable.
func zen4FMATable(t map[int]fmaChoice, zen4 bool) map[int]fmaChoice {
	if !zen4 || t == nil {
		return t
	}
	t[1024] = fmaChoice{factors: []int{32, 32}, fuse: fmaAll}
	return t
}

// fmaTableAMD64 is fmaTable: Cascade Lake's (cl, the 512-bit split layout)
// or Zen 3's (zn, the 256-bit split layout on AMD), measured in Round 42;
// nil elsewhere (Haswell was not measured: its FMA plans fuse every pass
// of skFactorize's factorization, Round 41's plans).
func fmaTableAMD64(cl, zn bool) map[int]fmaChoice {
	switch {
	case cl:
		return clFMATable()
	case zn:
		return znFMATable()
	}
	return nil
}

// clFMATable is Cascade Lake's (Round 42, one pinned core, nine rotated
// rounds at four heap placements; default time ÷ FMA time):
//   - 1024 takes 32·32, two passes where the default plan takes four
//     (8·4·4·8): 1.24 (1.22–1.26), against 1.01–1.02 for Round 41's plan;
//   - 2048 takes 32·8·8 with the last pass unfused: 1.07, against 0.98;
//   - 4096 to 16384 fuse nothing: every pass of their 512-bit split plans
//     ran slower fused (whole transforms 0.97–0.99, every mask of fused
//     passes below 1.00 at 4096 and 8192);
//   - 65536 leaves its final pass unfused (1.052 against 1.045).
//
// Every other length fuses every pass of its default factorization.
func clFMATable() map[int]fmaChoice {
	return map[int]fmaChoice{
		1024:    {factors: []int{32, 32}, fuse: fmaAll},
		2048:    {factors: []int{32, 8, 8}, fuse: 0b011},
		4096:    {fuse: 0},
		8192:    {fuse: 0},
		16384:   {fuse: 0},
		1 << 16: {fuse: 0b011111},
	}
}

// znFMATable is Zen 3's (Round 42, one pinned core, fifteen rotated rounds
// at eight heap placements): RFFT 256, whose half transform is 128 = 8·16
// on the frame kernels, ran 0.956× as fast with both passes fused and 0.997×
// with the final radix-16 pass unfused (complex 128: 1.030 and 1.016), so
// 128 leaves its final pass unfused. Radix 32 did not pay on this CPU
// (complex 1024 as 32·32 0.96× Round 41's FMA plan, 32·4·8 0.98×; 2048 as
// 32·8·8 0.87–0.90×; 256 as 32·8 0.72×), so no length takes it. Every
// other length fuses every pass, which ran at least as fast as the default
// plan on every row timed.
func znFMATable() map[int]fmaChoice {
	return map[int]fmaChoice{
		128: {fuse: 0b01},
	}
}
