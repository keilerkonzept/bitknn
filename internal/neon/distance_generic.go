package neon

import "math/bits"

func distancesWideGeneric(a []uint64, bs [][]uint64, out []uint32) {
	for i, b := range bs {
		dist := 0
		for j, aj := range a {
			dist += bits.OnesCount64(aj ^ b[j])
		}
		out[i] = uint32(dist)
	}
}
