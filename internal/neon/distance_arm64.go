package neon

import (
	"golang.org/x/sys/cpu"
)

func init() {
	if cpu.ARM64.HasASIMD {
		DistancesWide = DistancesWideNEON
	}
}

var DistancesWide = distancesWideGeneric

func DistancesWideNEON(a []uint64, bs [][]uint64, out []uint32)
