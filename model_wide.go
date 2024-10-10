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

// Predicts the label of a single input point. Reuses two slices of length K+1 for the neighbor heap.
// Returns the number of neighbors found.
func (me *WideModel) Predict1(k int, x []uint64, votes Votes) int {
	me.Narrow.PreallocateHeap(k)
	return me.Predict1Into(k, x, me.Narrow.HeapDistances, me.Narrow.HeapIndices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
// Returns the number of neighbors found.
func (me *WideModel) Predict1Into(k int, x []uint64, distances []int, indices []int, votes Votes) int {
	k = NearestWide(me.WideData, k, x, distances, indices)
	me.Narrow.Vote(k, distances, indices, votes)
	return k
}
