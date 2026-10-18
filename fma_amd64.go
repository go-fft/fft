package fft

import "github.com/go-fft/fft/internal/kernels"

// fmaSplitDefault: every amd64 split kernel has an FMA twin (fmasplit*.s).
func fmaSplitDefault() bool { return true }

// fmaTableDefault is fmaTableAMD64 for this machine.
func fmaTableDefault() map[int]fmaChoice {
	return fmaTableAMD64(kernels.UseStockhamSplit512, kernels.UseStockhamSplit && !kernels.IntelCPU)
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

func clFMATable() map[int]fmaChoice { return map[int]fmaChoice{} }

func znFMATable() map[int]fmaChoice { return map[int]fmaChoice{} }
