package bitknn

import (
	"math/bits"

	"github.com/keilerkonzept/bitknn/internal/heap"
)

// Nearest finds the nearest neighbors of the given point `x` by Hamming distance in `data`.
// The neighbor's distances and indices (in `data`) are written to the slices `distances` and `indices`.
// The two slices should be pre-allocated to length `k+1`.
// pre:
//
//	cap(distances) = cap(indices) = k+1 >= 1
func Nearest(data []uint64, k int, x uint64, distances, indices []int) int {
	heap := heap.MakeMax(distances, indices)
	distance0 := &distances[0]

	k0 := min(k, len(data))

	for i, d := range data[:k0] {
		dist := bits.OnesCount64(x ^ d)
		heap.Push(dist, i)
	}

	if k0 < k {
		return k0
	}

	maxDist := *distance0
	for i := k; i < len(data); i++ {
		dist := bits.OnesCount64(x ^ data[i])
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i)
		maxDist = *distance0
	}
	return k
}
