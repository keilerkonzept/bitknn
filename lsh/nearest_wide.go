package lsh

import (
	"math/bits"

	"github.com/keilerkonzept/bitknn/internal/heap"
	"github.com/keilerkonzept/bitknn/internal/slice"
)

// Nearest, but for wide data.
func NearestWide(data [][]uint64, bucketIDs []uint64, buckets map[uint64]slice.IndexRange, k int, xh uint64, x []uint64, bucketDistances []int, heapBucketIDs []uint64, distances []int, indices []int) (int, int) {
	dataHeap := heap.MakeMax[int](distances, indices)
	exactBucket := buckets[xh]
	numExamined := exactBucket.Length
	nearestWideInBucket(data, exactBucket, k, x, &distances[0], &dataHeap)

	// stop early for 1-NN
	if k == 1 && dataHeap.Len() == k {
		return k, exactBucket.Length
	}

	// otherwise, determine the k nearest buckets and find the k nearest neighbors in these buckets.
	bucketHeap := heap.MakeMax[uint64](bucketDistances, heapBucketIDs)
	nearestBuckets(bucketIDs, k, xh, &bucketDistances[0], &bucketHeap)
	n := nearestWideInBuckets(data, heapBucketIDs[:bucketHeap.Len()], buckets, k, x, xh, &distances[0], &dataHeap)

	return dataHeap.Len(), numExamined + n
}

func nearestWideInBucket(data [][]uint64, b slice.IndexRange, k int, x []uint64, distance0 *int, heap *heap.Max[int]) {
	if b.Length == 0 {
		return
	}

	end := b.Offset + b.Length
	end0 := b.Offset + min(b.Length, k)

	for i := b.Offset; i < end0; i++ {
		d := data[i]
		dist := 0
		for j, d := range d {
			dist += bits.OnesCount64(x[j] ^ d)
		}
		heap.Push(dist, i)
	}

	if b.Length < k {
		return
	}

	maxDist := *distance0
	for i := b.Offset + k; i < end; i++ {
		d := data[i]
		dist := 0
		for j, d := range d {
			dist += bits.OnesCount64(x[j] ^ d)
		}
		if dist >= maxDist {
			continue
		}
		heap.PushPop(dist, i)
		maxDist = *distance0
	}
}

func nearestWideInBuckets(data [][]uint64, inBuckets []uint64, buckets map[uint64]slice.IndexRange, k int, x []uint64, xh uint64, distance0 *int, heap *heap.Max[int]) int {
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
				d := data[i]
				dist := 0
				for j1, d := range d {
					dist += bits.OnesCount64(x[j1] ^ d)
				}
				if dist >= maxDist {
					continue
				}
				heap.PushPop(dist, i)
				maxDist = *distance0
			}
			continue
		}
		for i := b.Offset; i < end; i++ {
			d := data[i]
			dist := 0
			for j1, d := range d {
				dist += bits.OnesCount64(x[j1] ^ d)
			}
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
