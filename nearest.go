package bitknn

import (
	"math/bits"
)

// Nearest finds the nearest neighbors of the given point `x` by Hamming distance in `data`.
// The neighbor's distances and indices (in `data`) are written to the slices `distances` and `indices`.
// The two slices should be pre-allocated to length `k+1`.
// pre:
//
//	cap(distances) = cap(indices) = k+1 >= 1
func Nearest(data []uint64, k int, x uint64, distances, indices []int) int {
	heap := makeNeighborHeap(distances, indices)

	var maxDist int
	for i := range data {
		dist := bits.OnesCount64(x ^ data[i])
		if i < k {
			heap.push(dist, i)
			maxDist = distances[0]
			continue
		}
		if dist >= maxDist {
			continue
		}
		heap.pushpop(dist, i)
		maxDist = distances[0]
	}
	return min(len(data), k)
}
