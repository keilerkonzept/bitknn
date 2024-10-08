package lsh

import (
	"math/bits"

	"github.com/keilerkonzept/bitknn/internal/heap"
	"github.com/keilerkonzept/bitknn/internal/slice"
)

// Nearest finds the nearest neighbors for a given data point within the nearest buckets by hash Hamming distance.
//
// Parameters:
//   - data: The dataset.
//   - bucketIDs: All bucket IDs (hashes of dataset points)
//   - buckets: A map from bucket IDs to their index ranges in the dataset.
//   - k: The number of neighbors to find.
//   - xh: The hashed query point.
//   - x: The original query point.
//   - distances, heapBucketIDs, indices: Pre-allocated slices of length (k+1) for the neighbor heaps.
//
// Returns:
//   - The number of nearest neighbors found.
//   - The total number of data points examined.
func Nearest(data []uint64, bucketIDs []uint64, buckets map[uint64]slice.IndexRange, k int, xh uint64, x uint64, distances []int, heapBucketIDs []uint64, indices []int) (int, int) {
	k0 := nearestBuckets(bucketIDs, k, xh, distances, heapBucketIDs)
	k1, n := nearestInBuckets(data, heapBucketIDs[:k0], buckets, k, x, distances, indices)
	return k1, n
}

// nearestInBuckets finds the nearest neighbors within specific buckets.
// It returns the number of neighbors found and the total number of points examined.
func nearestInBuckets(data []uint64, inBuckets []uint64, buckets map[uint64]slice.IndexRange, k int, x uint64, distances []int, indices []int) (int, int) {
	heap := heap.MakeMax[int](distances, indices)
	var maxDist int
	j := 0
	t := 0
	for _, bid := range inBuckets {
		b := buckets[bid]
		end := b.Offset + b.Length
		t += b.Length
		if j >= k {
			for i := b.Offset; i < end; i++ {
				dist := bits.OnesCount64(x ^ data[i])
				if dist >= maxDist {
					continue
				}
				heap.PushPop(dist, i)
				maxDist = distances[0]
			}
			continue
		}
		for i := b.Offset; i < end; i++ {
			dist := bits.OnesCount64(x ^ data[i])
			if j < k {
				heap.Push(dist, i)
				maxDist = distances[0]
				j++
				continue
			}
			if dist >= maxDist {
				continue
			}
			heap.PushPop(dist, i)
			maxDist = distances[0]
		}
	}
	if j < k {
		return j, t
	}
	return k, t
}

// nearestBuckets finds the buckets with IDs that are (Hamming-)nearest to a query point hash.
// It returns the number of nearest buckets found.
func nearestBuckets(bucketIDs []uint64, k int, x uint64, distances []int, heapBucketIDs []uint64) int {
	heap := heap.MakeMax[uint64](distances, heapBucketIDs)

	k0 := min(k, len(bucketIDs))
	var maxDist int
	for _, b := range bucketIDs[:k0] {
		dist := bits.OnesCount64(x ^ b)
		heap.Push(dist, b)
	}
	if k0 < k {
		return k0
	}
	maxDist = distances[0]
	for _, b := range bucketIDs[k0:] {
		dist := bits.OnesCount64(x ^ b)
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, b)
		maxDist = distances[0]
	}
	return k
}
