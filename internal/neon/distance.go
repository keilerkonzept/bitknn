//go:build !arm64

package neon

func DistancesWide(a []uint64, bs [][]uint64, out []uint32) {
	distancesWideGeneric(a, bs, out)
}
