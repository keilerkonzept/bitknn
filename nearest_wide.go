package bitknn

import (
	"math/bits"

	"github.com/keilerkonzept/bitknn/internal/heap"
)

// [bitknn.Nearest], but for wide data.
func NearestWide(data [][]uint64, k int, x []uint64, distances, indices []int) int {
	heap := heap.MakeMax(distances, indices)
	distance0 := &distances[0]

	k0 := min(k, len(data))

	for i, d := range data[:k0] {
		dist := 0
		for j, d := range d {
			dist += bits.OnesCount64(x[j] ^ d)
		}
		heap.Push(dist, i)
	}

	if k0 < k {
		return k0
	}

	maxDist := *distance0
	for i := k; i < len(data); i++ {
		dist := 0
		d := data[i]
		for j, d := range d {
			dist += bits.OnesCount64(x[j] ^ d)
		}
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i)
		maxDist = *distance0
	}
	return k
}
