package bitknn

// Create a k-NN model for the given data points and labels.
func FitWide(data [][]uint64, labels []int, opts ...Option) *WideModel {
	m := &WideModel{
		Narrow:   Fit(nil, labels, opts...),
		WideData: data,
	}
	return m
}

// A k-NN model for slices of uint64s.
type WideModel struct {
	Narrow *Model

	// Input data points.
	WideData [][]uint64
}

func (me *WideModel) PreallocateHeap(k int) {
	me.Narrow.PreallocateHeap(k)
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the pre-allocated slices.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *WideModel) Find(k int, x []uint64) ([]int, []int) {
	me.PreallocateHeap(k)
	return me.FindInto(k, x, me.Narrow.HeapDistances, me.Narrow.HeapIndices)
}

// FindV is [WideModel.Find], but vectorizable (currently only on ARM64 with NEON instructions).
// The provided [batch] slice must have length >=k and is used to pre-compute batches of distances.
func (me *WideModel) FindV(k int, x []uint64, batch []uint32) ([]int, []int) {
	me.PreallocateHeap(k)
	return me.FindIntoV(k, x, batch, me.Narrow.HeapDistances, me.Narrow.HeapIndices)
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the provided slices.
// The slices should be pre-allocated to length k+1.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *WideModel) FindInto(k int, x []uint64, distances []int, indices []int) ([]int, []int) {
	k = NearestWide(me.WideData, k, x, distances, indices)
	return distances[:k], indices[:k]
}

// FindIntoV is [WideModel.FindInto], but vectorizable (currently only on ARM64 with NEON instructions).
// The provided [batch] slice must have length >=k and is used to pre-compute batches of distances.
func (me *WideModel) FindIntoV(k int, x []uint64, batch []uint32, distances []int, indices []int) ([]int, []int) {
	k = NearestWideV(me.WideData, k, x, batch, distances, indices)
	return distances[:k], indices[:k]
}

// Predicts the label of a single input point. Reuses two slices of length K+1 for the neighbor heap.
// Returns the number of neighbors found.
func (me *WideModel) Predict(k int, x []uint64, votes VoteCounter) int {
	me.PreallocateHeap(k)
	return me.PredictInto(k, x, me.Narrow.HeapDistances, me.Narrow.HeapIndices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
// Returns the number of neighbors found.
func (me *WideModel) PredictInto(k int, x []uint64, distances []int, indices []int, votes VoteCounter) int {
	k = NearestWide(me.WideData, k, x, distances, indices)
	me.Narrow.Vote(k, distances, indices, votes)
	return k
}

// PredictV is [WideModel.Predict], but vectorizable (currently only on ARM64 with NEON instructions).
// The provided [batch] slice must have length >=k and is used to pre-compute batches of distances.
func (me *WideModel) PredictV(k int, x []uint64, batch []uint32, votes VoteCounter) int {
	me.PreallocateHeap(k)
	return me.PredictIntoV(k, x, batch, me.Narrow.HeapDistances, me.Narrow.HeapIndices, votes)
}

// PredictIntoV is [WideModel.PredictInto], but vectorizable (currently only on ARM64 with NEON instructions).
// The provided [batch] slice must have length >=k and is used to pre-compute batches of distances.
func (me *WideModel) PredictIntoV(k int, x []uint64, batch []uint32, distances []int, indices []int, votes VoteCounter) int {
	k = NearestWideV(me.WideData, k, x, batch, distances, indices)
	me.Narrow.Vote(k, distances, indices, votes)
	return k
}
