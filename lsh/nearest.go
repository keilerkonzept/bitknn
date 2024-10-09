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
func Nearest(data []uint64, bucketIDs []uint64, buckets map[uint64]slice.IndexRange, k int, xh uint64, x uint64, bucketDistances []int, heapBucketIDs []uint64, distances []int, indices []int) (int, int) {
	dataHeap := heap.MakeMax[int](distances, indices)
	exactBucket := buckets[xh]
	numExamined := exactBucket.Length
	nearestInBucket(data, exactBucket, k, x, &distances[0], &dataHeap)

	// if the exact bucket already contains k neighbors, stop and return them
	if dataHeap.Len() == k {
		return k, exactBucket.Length
	}

	// otherwise, determine the k nearest buckets and find the k nearest neighbors in these buckets.
	bucketHeap := heap.MakeMax[uint64](bucketDistances, heapBucketIDs)
	nearestBuckets(bucketIDs, k, xh, &bucketDistances[0], &bucketHeap)
	n := nearestInBuckets(data, heapBucketIDs[:bucketHeap.Len()], buckets, k, x, xh, &distances[0], &dataHeap)

	return dataHeap.Len(), numExamined + n
}

func nearestInBucket(data []uint64, b slice.IndexRange, k int, x uint64, distance0 *int, heap *heap.Max[int]) {
	if b.Length == 0 {
		return
	}

	end := b.Offset + b.Length
	end0 := b.Offset + min(b.Length, k)

	for i := b.Offset; i < end0; i++ {
		dist := bits.OnesCount64(x ^ data[i])
		heap.Push(dist, i)
	}

	if b.Length < k {
		return
	}

	maxDist := *distance0
	for i := b.Offset + k; i < end; i++ {
		dist := bits.OnesCount64(x ^ data[i])
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i)
		maxDist = *distance0
	}
}

// nearestInBuckets finds the nearest neighbors within specific buckets.
// Returns the number of points examined.
func nearestInBuckets(data []uint64, inBuckets []uint64, buckets map[uint64]slice.IndexRange, k int, x, xh uint64, distance0 *int, heap *heap.Max[int]) int {
	var maxDist int
	j := heap.Len()
	if j > 0 {
		maxDist = *distance0
	}
	t := 0
	for _, bid := range inBuckets {
		if bid == xh { // skip exact bucket
			continue
		}
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
				maxDist = *distance0
			}
			continue
		}
		for i := b.Offset; i < end; i++ {
			dist := bits.OnesCount64(x ^ data[i])
			if j < k {
				heap.Push(dist, i)
				maxDist = *distance0
				j++
				continue
			}
			if dist >= maxDist {
				continue
			}
			heap.PushPop(dist, i)
			maxDist = *distance0
		}
	}
	return t
}

// nearestBuckets finds the buckets with IDs that are (Hamming-)nearest to a query point hash.
func nearestBuckets(bucketIDs []uint64, k int, x uint64, distance0 *int, heap *heap.Max[uint64]) {
	k0 := min(k, len(bucketIDs))
	var maxDist int
	for _, b := range bucketIDs[:k0] {
		dist := bits.OnesCount64(x ^ b)
		heap.Push(dist, b)
	}
	if k0 < k {
		return
	}
	maxDist = *distance0
	for _, b := range bucketIDs[k0:] {
		dist := bits.OnesCount64(x ^ b)
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, b)
		maxDist = *distance0
	}
}
