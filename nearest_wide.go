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
		for j, x := range x {
			dist += bits.OnesCount64(d[j] ^ x)
		}
		heap.Push(dist, i)
	}

	if len(data) <= k {
		return k0
	}

	maxDist := *distance0
	_ = data[k]
	for i := k; i < len(data); i++ {
		dist := 0
		d := data[i]
		for j, x := range x {
			dist += bits.OnesCount64(d[j] ^ x)
		}
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i)
		maxDist = *distance0
	}
	return k
}
