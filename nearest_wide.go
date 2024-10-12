package bitknn

import (
	"math/bits"

	"github.com/keilerkonzept/bitknn/internal/heap"
	"github.com/keilerkonzept/bitknn/internal/neon"
)

// [Nearest], but for wide data.
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

// [NearestWide], but vectorizable (currently only on ARM64 with NEON instructions).
// The `batch` array must have at least length `k`, and is used to pre-compute batches of distances.
func NearestWideV(data [][]uint64, k int, x []uint64, batch []uint32, distances, indices []int) int {
	if k == 0 || len(data) == 0 {
		return 0
	}
	_ = batch[k-1]
	heap := heap.MakeMax(distances, indices)
	distance0 := &distances[0]

	k0 := min(k, len(data))
	datak0 := data[:k0:k0]

	batchk0 := batch[:k0:k0]
	neon.DistancesWide(x, datak0, batchk0)

	for i, dist := range batchk0 {
		heap.Push(int(dist), i)
	}

	if len(data) <= k {
		return k0
	}

	maxDist := *distance0

	b := len(batch)
	_ = data[k]
	i := k
	for ; i <= len(data)-b; i += b {
		neon.DistancesWide(x, data[i:i+b], batch)
		for j := range batch {
			dist := int(batch[j])
			if dist >= maxDist {
				continue
			}
			heap.PushPop(dist, i+j)
			maxDist = *distance0
		}
	}

	remainder := len(data) - i
	if remainder <= 0 {
		return k
	}
	_ = batch[remainder-1]

	neon.DistancesWide(x, data[i:], batch)
	for j := range remainder {
		dist := int(batch[j])
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i+j)
		maxDist = *distance0
	}
	return k
}
