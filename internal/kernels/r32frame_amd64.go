package kernels

// The split radix-32 kernels with the values between their stages in their
// own frame (Round 42, r32frame_amd64.s, go-asmgen's SplitR32InFrame),
// which r32InFrame selects.

// r32InFrame reports whether a split radix-32 pass of ido points per block
// and l1 blocks keeps the values between its stages in its frame rather
// than in its output: when its output streams are at least 1 KB apart (l1·ido
// >= 64 points), so that its 32 output slots fall into few L1 sets. On
// Cascade Lake (Round 42, nine rotated rounds, four placements) the frame
// made 2048 = 32·8·8 1.04× and RFFT 4096 1.09× faster, and 1024 = 32·32
// (streams 512 bytes apart) 0.87× as fast. A pure function, tested both
// ways on any machine.
func r32InFrame(ido, l1 int) bool {
	switch R32FrameForce {
	case 1:
		return true
	case 2:
		return false
	}
	return ido*l1 >= 64
}

// R32FrameForce (prototype, Round 44): 1 forces the in-frame radix-32
// kernels, 2 the through-the-output ones, 0 keeps r32InFrame's rule.
var R32FrameForce int

var (
	skSplit32FMAF = [2][5]skPassFn{
		{splitOut: skSplit32ISFwdFMAF, splitBoth: skSplit32SSFwdFMAF, splitIn: skSplit32SIFwdFMAF, splitNone: skSplit32IIFwdFMAF},
		{splitOut: skSplit32ISInvFMAF, splitBoth: skSplit32SSInvFMAF, splitIn: skSplit32SIInvFMAF, splitNone: skSplit32IIInvFMAF},
	}
	intelSplit512R32FMAF = [2][5]skPassFn{
		{splitOut: intelSplit512R32ISFwdFMAF, splitBoth: intelSplit512R32SSFwdFMAF, splitIn: intelSplit512R32SIFwdFMAF, splitNone: intelSplit512R32IIFwdFMAF},
		{splitOut: intelSplit512R32ISInvFMAF, splitBoth: intelSplit512R32SSInvFMAF, splitIn: intelSplit512R32SIInvFMAF, splitNone: intelSplit512R32IIInvFMAF},
	}
)

//go:noescape
func skSplit32ISFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SSFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SIFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32IIFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32ISInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SSInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32SIInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func skSplit32IIInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32ISFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SSFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SIFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32IIFwdFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32ISInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SSInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32SIInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)

//go:noescape
func intelSplit512R32IIInvFMAF(cc, ch, tw *complex128, k *float64, ido, l1 int)
