package lsh

import (
	"cmp"
	"slices"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/slice"
)

// Model represents an LSH k-NN model, mapping points to buckets based on a locality-sensitive hash function.
type Model struct {
	*bitknn.Model
	Hash Hash // LSH function mapping points to bucket IDs.

	BucketIDs []uint64                    // Bucket IDs.
	Buckets   map[uint64]slice.IndexRange // Bucket contents for each hash (offset+length in Data).

	HeapBucketDistances []int
	HeapBucketIDs       []uint64
}

// PreallocateHeap allocates memory for the nearest neighbor heap.
func (me *Model) PreallocateHeap(k int) {
	me.HeapBucketDistances = slice.OrAlloc(me.HeapBucketDistances, k+1)
	me.HeapBucketIDs = slice.OrAlloc(me.HeapBucketIDs, k+1)
	me.HeapDistances = slice.OrAlloc(me.HeapDistances, k+1)
	me.HeapIndices = slice.OrAlloc(me.HeapIndices, k+1)
}

// Fit creates and fits an LSH k-NN model using the provided data, labels, and hash function.
// It groups points into buckets using the LSH hash function.
func Fit(data []uint64, labels []int, hash Hash, opts ...bitknn.Option) *Model {
	knnModel := bitknn.Fit(data, labels, opts...)
	values := knnModel.Values
	buckets := make([]uint64, len(data))
	hash.Hash(data, buckets)

	indices := make([]int, len(data))
	for i := range indices {
		indices[i] = i
	}

	// Sort data by bucket id so that each bucket's data slice is contiguous.	slices.SortStableFunc(indices, func(a, b int) int {
	slices.SortStableFunc(indices, func(a, b int) int {
		return cmp.Compare(buckets[a], buckets[b])
	})

	// Reorder all data-indexed slices to match the bucket sort order.
	slice.ReorderInPlace(func(i, j int) {
		buckets[i], buckets[j] = buckets[j], buckets[i]
		data[i], data[j] = data[j], data[i]
		labels[i], labels[j] = labels[j], labels[i]
		if values != nil {
			values[i], values[j] = values[j], values[i]
		}
	}, indices)

	bucketData, bucketIDs := slice.GroupSorted(data, buckets)

	return &Model{
		Model:     knnModel,
		Hash:      hash,
		BucketIDs: bucketIDs,
		Buckets:   bucketData,
	}
}

// Predict1 predicts the label for a single input using the LSH model.
func (me *Model) Predict1(k int, x uint64, votes []float64) int {
	me.HeapBucketDistances = slice.OrAlloc(me.HeapBucketDistances, k+1)
	me.HeapBucketIDs = slice.OrAlloc(me.HeapBucketIDs, k+1)
	me.HeapDistances = slice.OrAlloc(me.HeapDistances, k+1)
	me.HeapIndices = slice.OrAlloc(me.HeapIndices, k+1)
	return me.Predict1Into(k, x, votes, me.HeapBucketDistances, me.HeapBucketIDs, me.HeapDistances, me.HeapIndices)
}

// Predicts the label of a single input point. Each call allocates three new slices of length [k]+1 for the neighbor heaps.
func (me *Model) Predict1Alloc(k int, x uint64, votes []float64) int {
	bucketDistances := make([]int, k+1)
	bucketIDs := make([]uint64, k+1)
	distances := make([]int, k+1)
	indices := make([]int, k+1)

	return me.Predict1Into(k, x, votes, bucketDistances, bucketIDs, distances, indices)
}

// Predict1Into predicts the label for a single input using the given slices (of length [k]+1 each) for the neighbor heaps.
func (me *Model) Predict1Into(k int, x uint64, votes []float64, bucketDistances []int, bucketIDs []uint64, distances []int, indices []int) int {
	xp := me.Hash.Hash1(x)
	k, n := Nearest(me.Data, me.BucketIDs, me.Buckets, k, xp, x, bucketDistances, bucketIDs, distances, indices)

	clear(votes)
	switch me.DistanceWeighting {
	case bitknn.DistanceWeightingNone:
		if me.Values == nil {
			me.Votes1(k, indices, votes)
		} else {
			me.Votes1v(k, indices, votes)
		}
	case bitknn.DistanceWeightingLinear:
		if me.Values == nil {
			me.Votes1l(k, indices, votes, distances)
		} else {
			me.Votes1vl(k, indices, votes, distances)
		}
	case bitknn.DistanceWeightingQuadratic:
		if me.Values == nil {
			me.Votes1q(k, indices, votes, distances)
		} else {
			me.Votes1vq(k, indices, votes, distances)
		}
	case bitknn.DistanceWeightingCustom:
		f := me.DistanceWeightingFunc
		if me.Values == nil {
			me.Votes1c(k, indices, votes, f, distances)
		} else {
			me.Votes1vc(k, indices, votes, f, distances)
		}
	}
	return n
}
