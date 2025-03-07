// Package bitknn provides a fast exact k-nearest neighbors (k-NN) implementation for binary feature vectors.
package bitknn

import (
	"github.com/keilerkonzept/bitknn/internal/slice"
)

// Create a k-NN model for the given data points and labels.
func Fit(data []uint64, labels []int, opts ...Option) *Model {
	m := &Model{
		Data:              data,
		Labels:            labels,
		DistanceWeighting: DistanceWeightingNone,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// A k-NN model for uint64s.
type Model struct {
	// Input data points.
	Data []uint64
	// Class labels for each data point.
	Labels []int
	// Vote values for each data point.
	Values []float64

	// Distance weighting function.
	DistanceWeighting DistanceWeighting
	// Custom function when [Model.DistanceWeighting] is [DistanceWeightingCustom].
	DistanceWeightingFunc func(int) float64

	HeapDistances []int
	HeapIndices   []int
}

func (me *Model) PreallocateHeap(k int) {
	me.HeapDistances = slice.OrAlloc(me.HeapDistances, k+1)
	me.HeapIndices = slice.OrAlloc(me.HeapIndices, k+1)
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the pre-allocated slices.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *Model) Find(k int, x uint64) ([]int, []int) {
	me.PreallocateHeap(k)
	return me.FindInto(k, x, me.HeapDistances, me.HeapIndices)
}

// Finds the nearest neighbors of the given point.
// Writes their distances and indices in the dataset into the provided slices.
// The slices should be pre-allocated to length k+1.
// Returns the distance and index slices, truncated to the actual number of neighbors found.
func (me *Model) FindInto(k int, x uint64, distances []int, indices []int) ([]int, []int) {
	k = Nearest(me.Data, k, x, distances, indices)
	return distances[:k], indices[:k]
}

// Predicts the label of a single input point. Each call allocates two new slices of length K+1 for the neighbor heap.
func (me *Model) PredictAlloc(k int, x uint64, votes VoteCounter) {
	distances, indices := make([]int, k+1), make([]int, k+1)
	me.PredictInto(k, x, distances, indices, votes)
}

// Predicts the label of a single input point. Reuses two slices of length K+1 for the neighbor heap.
func (me *Model) Predict(k int, x uint64, votes VoteCounter) {
	me.HeapDistances = slice.OrAlloc(me.HeapDistances, k+1)
	me.HeapIndices = slice.OrAlloc(me.HeapIndices, k+1)
	me.PredictInto(k, x, me.HeapDistances, me.HeapIndices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
func (me *Model) PredictInto(k int, x uint64, distances []int, indices []int, votes VoteCounter) {
	k = Nearest(me.Data, k, x, distances, indices)
	me.Vote(k, distances, indices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
func (me *Model) Vote(k int, distances []int, indices []int, votes VoteCounter) {
	votes.Clear()
	switch me.DistanceWeighting {
	case DistanceWeightingNone:
		if me.Values == nil {
			me.votes1(k, indices, votes)
		} else {
			me.votes1v(k, indices, votes)
		}
	case DistanceWeightingLinear:
		if me.Values == nil {
			me.votes1l(k, indices, votes, distances)
		} else {
			me.votes1vl(k, indices, votes, distances)
		}
	case DistanceWeightingQuadratic:
		if me.Values == nil {
			me.votes1q(k, indices, votes, distances)
		} else {
			me.votes1vq(k, indices, votes, distances)
		}
	case DistanceWeightingCustom:
		f := me.DistanceWeightingFunc
		if me.Values == nil {
			me.votes1c(k, indices, votes, f, distances)
		} else {
			me.votes1vc(k, indices, votes, f, distances)
		}
	}
}

func (me *Model) votes1vc(k int, indices []int, votes VoteCounter, f func(int) float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, f(distances[i])*me.Values[index])
	}
}

func (me *Model) votes1c(k int, indices []int, votes VoteCounter, f func(int) float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, f(distances[i]))
	}
}

func (me *Model) votes1vq(k int, indices []int, votes VoteCounter, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, DistanceWeightingFuncQuadratic(distances[i])*me.Values[index])
	}
}

func (me *Model) votes1q(k int, indices []int, votes VoteCounter, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, DistanceWeightingFuncQuadratic(distances[i]))
	}
}

func (me *Model) votes1vl(k int, indices []int, votes VoteCounter, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, DistanceWeightingFuncLinear(distances[i])*me.Values[index])
	}
}

func (me *Model) votes1l(k int, indices []int, votes VoteCounter, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, DistanceWeightingFuncLinear(distances[i]))
	}
}

func (me *Model) votes1v(k int, indices []int, votes VoteCounter) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, me.Values[index])
	}
}

func (me *Model) votes1(k int, indices []int, votes VoteCounter) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes.Add(label, 1)
	}
}

type distanceWeighting int
type DistanceWeighting distanceWeighting

const (
	DistanceWeightingNone DistanceWeighting = iota
	DistanceWeightingLinear
	DistanceWeightingQuadratic
	DistanceWeightingCustom
)

func (me DistanceWeighting) String() string {
	switch me {
	case DistanceWeightingNone:
		return "none"
	case DistanceWeightingLinear:
		return "linear"
	case DistanceWeightingQuadratic:
		return "quadratic"
	case DistanceWeightingCustom:
		return "custom"
	}
	return "unknown"
}

func DistanceWeightingFuncLinear(dist int) float64    { return 1.0 / float64(1+dist) }
func DistanceWeightingFuncQuadratic(dist int) float64 { return 1.0 / float64(1+(dist*dist)) }
