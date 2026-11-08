package kernels

// One CPU-feature source for every dispatch decision (go-fft/fft#86).
//
// The AVX-512 kernels use two ISA subsets: AVX512F (the 512-bit arithmetic,
// shuffles and blends) and AVX512DQ (KMOVB, the opmask move the split-512,
// untangle and interleaved Stockham kernels execute, and VXORPD/VORPD on ZMM).
// Intel Knights Landing and Knights Mill have F without DQ and raise #UD on
// KMOVB, so "AVX-512 present" means F and DQ together. TestAVX512KernelsUseOnly
// FandDQ scans every amd64 .s file and fails when a kernel uses an instruction
// whose subset is not in that set, so a later kernel needing BW or VL has to
// extend the gate before it can land.
//
// Every default below is a pure function of one cpuFeatures value, so the tests
// can drive the whole dispatch with a simulated CPU (dispatchFor) on any
// machine.

// cpuFeatures is what the dispatch asks of the CPU. The detected value comes
// from the go-asmgen probes in cpu_amd64.s, which also check the OS state.
type cpuFeatures struct {
	AVX2        bool
	AVX512F     bool
	AVX512DQ    bool
	Intel       bool // CPUID vendor GenuineIntel (tuning only)
	AMD         bool // CPUID vendor AuthenticAMD (tuning only)
	AMDFamily19 bool // displayed family 0x19 (tuning only)
}

// avx512 reports whether every subset the AVX-512 kernels execute is present.
func (c cpuFeatures) avx512() bool { return c.AVX512F && c.AVX512DQ }

// supportsAVX512DQ is the go-asmgen AVX512DQ probe (cpu_amd64.s): AVX512F and
// CPUID.7.0:EBX bit 17, with the OS saving opmask and ZMM state.
func supportsAVX512DQ() bool

// detectedCPU is the hardware this process runs on.
var detectedCPU = cpuFeatures{
	AVX2:        useAVX2,
	AVX512F:     supportsAVX512F(),
	AVX512DQ:    supportsAVX512DQ(),
	Intel:       isGenuineIntel(),
	AMD:         isAuthenticAMD(),
	AMDFamily19: isFamily19(),
}

// dispatch holds the defaults derived from a cpuFeatures value.
type dispatch struct {
	Stockham512 bool // UseStockhamAVX512
	Zen4        bool // AMDZen4
	Untangle512 bool // UseUntangleAVX512
	Split512    bool // UseStockhamSplit512
	HasAVX512   bool // intelHasAVX512
	Split       bool // UseStockhamSplit
}

// dispatchFor computes every default from c: the single place the AVX-512
// gate is applied, so no site can test a narrower probe than the others.
func dispatchFor(c cpuFeatures) dispatch {
	wide := c.avx512()
	zen4 := zen4Default(wide, c.AMD, c.AMDFamily19)
	return dispatch{
		Stockham512: wide,
		Zen4:        zen4,
		Untangle512: untangle512Default(c.AVX2, wide, c.Intel),
		Split512:    intelSplit512Default(c.AVX2, wide, c.Intel),
		HasAVX512:   wide,
		Split:       splitDefault(c.AVX2, wide, zen4),
	}
}

var detectedDispatch = dispatchFor(detectedCPU)
