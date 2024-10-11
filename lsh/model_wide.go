package lsh

import (
	"cmp"
	"slices"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/slice"
)

// WideModel is an LSH k-NN model, mapping points to buckets based on a locality-sensitive hash function.
type WideModel struct {
	*bitknn.WideModel
	Hash HashWide // LSH function mapping points to bucket IDs.

	BucketIDs []uint64                    // Bucket IDs.
	Buckets   map[uint64]slice.IndexRange // Bucket contents for each hash (offset+length in Data).

	HeapBucketDistances []int
	HeapBucketIDs       []uint64
}

// PreallocateHeap allocates memory for the nearest neighbor heap.
func (me *WideModel) PreallocateHeap(k int) {
	me.HeapBucketDistances = slice.OrAlloc(me.HeapBucketDistances, k+1)
	me.HeapBucketIDs = slice.OrAlloc(me.HeapBucketIDs, k+1)
	me.WideModel.PreallocateHeap(k)
}

// Fit creates and fits an LSH k-NN model using the provided data, labels, and hash function.
// It groups points into buckets using the LSH hash function.
func FitWide(data [][]uint64, labels []int, hash HashWide, opts ...bitknn.Option) *WideModel {
	knnModel := bitknn.FitWide(data, labels, opts...)
	values := knnModel.Narrow.Values
	buckets := make([]uint64, len(data))
	hash.HashWide(data, buckets)

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

	return &WideModel{
		WideModel: knnModel,
		Hash:      hash,
		BucketIDs: bucketIDs,
		Buckets:   bucketData,
	}
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the pre-allocated slices.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *WideModel) Find(k int, x []uint64) ([]int, []int) {
	me.PreallocateHeap(k)
	return me.FindInto(k, x, me.HeapBucketDistances, me.HeapBucketIDs, me.Narrow.HeapDistances, me.Narrow.HeapIndices)
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the provided slices.
// The slices should be pre-allocated to length k+1.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *WideModel) FindInto(k int, x []uint64, bucketDistances []int, bucketIDs []uint64, distances []int, indices []int) ([]int, []int) {
	xp := me.Hash.Hash1Wide(x)
	k, _ = NearestWide(me.WideData, me.BucketIDs, me.Buckets, k, xp, x, bucketDistances, bucketIDs, distances, indices)
	return distances[:k], indices[:k]
}

// Predict predicts the label for a single input using the LSH model.
func (me *WideModel) Predict(k int, x []uint64, votes bitknn.VoteCounter) int {
	me.PreallocateHeap(k)
	return me.PredictInto(k, x, votes, me.HeapBucketDistances, me.HeapBucketIDs, me.Narrow.HeapDistances, me.Narrow.HeapIndices)
}

// Predicts the label of a single input point. Each call allocates three new slices of length [k]+1 for the neighbor heaps.
func (me *WideModel) PredictAlloc(k int, x []uint64, votes bitknn.VoteCounter) int {
	bucketDistances := make([]int, k+1)
	bucketIDs := make([]uint64, k+1)
	distances := make([]int, k+1)
	indices := make([]int, k+1)

	return me.PredictInto(k, x, votes, bucketDistances, bucketIDs, distances, indices)
}

// PredictInto predicts the label for a single input using the given slices (of length [k]+1 each) for the neighbor heaps.
func (me *WideModel) PredictInto(k int, x []uint64, votes bitknn.VoteCounter, bucketDistances []int, bucketIDs []uint64, distances []int, indices []int) int {
	xp := me.Hash.Hash1Wide(x)
	k0, _ := NearestWide(me.WideData, me.BucketIDs, me.Buckets, k, xp, x, bucketDistances, bucketIDs, distances, indices)
	me.WideModel.Narrow.Vote(k0, distances, indices, votes)
	return k0
}
