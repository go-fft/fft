package kernels

// On amd64 the routed butterfly kernel is hand-written SSE2 assembly whose
// contract is BIT-FOR-BIT identity with the scalar oracle (separately-rounded
// MULPD/ADDPD matching the non-fused GOAMD64=v1 oracle). So the SIMD-vs-scalar
// test requires exact equality here.
const butterflyBitExact = true
