#include "go_asm.h"
#include "textflag.h"

// func DistancesWideNEON(a []uint64, b [][]uint64, out []uint32)
//
// Computes the Hamming distance between 'a' and each slice in 'b',
// storing the results in 'out'.
//
// Inputs:
//   a_base+0(FP)  : base address of slice a
//   a_len+8(FP)   : length of slice a
//   (a_cap+16(FP)  : capacity of slice a)
//   bs_base+24(FP) : base address of slice b (slice of slices)
//   bs_len+32(FP)  : length of slice b (number of slices)
//   (bs_cap+40(FP)  : capacity of slice b)
//   out_base+48(FP): base address of output slice
//   (out_len+56(FP)): length of output slice
//   (out_cap+64(FP)): capacity of output slice
//
// Assumes that all slices in 'b' have the same length as 'a',
// and that 'out' has at least 'bs_len' elements.

//go:linkname DistancesWideNEON DistancesWideNEON
//go:noescape
TEXT ·DistancesWideNEON(SB), NOSPLIT, $0-72
    // Load input parameters
    MOVD a_base+0(FP), R0
    MOVD a_len+8(FP), R1
    MOVD bs_base+24(FP), R2
    MOVD bs_len+32(FP), R3
    MOVD out_base+48(FP), R4

    // Outer loop counter
    MOVD R3, R5
    CBZ R5, done

outer_loop:
    MOVD a_base+0(FP), R0

    // Load the base address of the current slice in 'b'
    MOVD (R2), R6
    ADD $24, R2  // Move to the next slice in 'b'

    // Initialize the result for this slice to 0
    MOVD $0, R7

    // Inner loop counter (number of uint64 in 'a')
    MOVD R1, R8

    VEOR V1.B16,V1.B16,V1.B16
    VEOR V2.B16,V2.B16,V2.B16
    VEOR V3.B16,V3.B16,V3.B16
    // Check if the length is at least 2 (16 bytes)
    CMP $2, R8
    BLT inner_remainder

inner_loop:
    // Load 16 bytes (2 uint64s) from each slice
    VLD1.P 16(R0), [V0.D2]
    VLD1.P 16(R6), [V1.D2]

    // XOR the loaded vectors
    VEOR V0.B16, V1.B16, V2.B16

    // Count the set bits
    VCNT V2.B16, V2.B16

    // Sum up the counts
    VUADDLV V2.B16, V3

    // Add the result to the total
    FMOVD F3, R9
    ADD R9, R7

    // Decrement the counter by 2 and continue if there are more elements
    SUB $2, R8
    CMP $2, R8
    BGE inner_loop

inner_remainder:
    // Handle the remaining element if the length is odd
    CBZ R8, inner_done
    MOVD (R0), R9
    MOVD (R6), R10
    EOR R9, R10, R9
    FMOVD R9, F0
    VCNT V0.B8, V0.B8
    VUADDLV V0.B8, V0
    FMOVD F0, R9
    ADD R9, R7

inner_done:
    // Store the distance in the output slice
    MOVW R7, (R4)
    ADD $4, R4  // Move to the next element in 'out'

    // Decrement the outer loop counter and continue if there are more slices
    SUB $1, R5
    CBNZ R5, outer_loop

done:
    RET
