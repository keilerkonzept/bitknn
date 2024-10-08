package bitknn

import "github.com/keilerkonzept/bitknn/internal/slice"

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

// Predicts the label of a single input point. Each call allocates two new slices of length K+1 for the neighbor heap.
func (me *Model) Predict1Alloc(k int, x uint64, votes []float64) {
	distances, indices := make([]int, k+1), make([]int, k+1)
	me.Predict1Into(k, x, distances, indices, votes)
}

// Predicts the label of a single input point. Reuses two slices of length K+1 for the neighbor heap.
func (me *Model) Predict1(k int, x uint64, votes []float64) {
	me.HeapDistances = slice.OrAlloc(me.HeapDistances, k+1)
	me.HeapIndices = slice.OrAlloc(me.HeapIndices, k+1)
	me.Predict1Into(k, x, me.HeapDistances, me.HeapIndices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
func (me *Model) Predict1Into(k int, x uint64, distances []int, indices []int, votes []float64) {
	k = Nearest(me.Data, k, x, distances, indices)
	clear(votes)
	switch me.DistanceWeighting {
	case DistanceWeightingNone:
		if me.Values == nil {
			me.Votes1(k, indices, votes)
		} else {
			me.Votes1v(k, indices, votes)
		}
	case DistanceWeightingLinear:
		if me.Values == nil {
			me.Votes1l(k, indices, votes, distances)
		} else {
			me.Votes1vl(k, indices, votes, distances)
		}
	case DistanceWeightingQuadratic:
		if me.Values == nil {
			me.Votes1q(k, indices, votes, distances)
		} else {
			me.Votes1vq(k, indices, votes, distances)
		}
	case DistanceWeightingCustom:
		f := me.DistanceWeightingFunc
		if me.Values == nil {
			me.Votes1c(k, indices, votes, f, distances)
		} else {
			me.Votes1vc(k, indices, votes, f, distances)
		}
	}
}

func (me *Model) Votes1vc(k int, indices []int, votes []float64, f func(int) float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += f(distances[i]) * me.Values[index]
	}
}

func (me *Model) Votes1c(k int, indices []int, votes []float64, f func(int) float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += f(distances[i])
	}
}

func (me *Model) Votes1vq(k int, indices []int, votes []float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += DistanceWeightingFuncQuadratic(distances[i]) * me.Values[index]
	}
}

func (me *Model) Votes1q(k int, indices []int, votes []float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += DistanceWeightingFuncQuadratic(distances[i])
	}
}

func (me *Model) Votes1vl(k int, indices []int, votes []float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += DistanceWeightingFuncLinear(distances[i]) * me.Values[index]
	}
}

func (me *Model) Votes1l(k int, indices []int, votes []float64, distances []int) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += DistanceWeightingFuncLinear(distances[i])
	}
}

func (me *Model) Votes1v(k int, indices []int, votes []float64) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label] += me.Values[index]
	}
}

func (me *Model) Votes1(k int, indices []int, votes []float64) {
	for i := range k {
		index := indices[i]
		label := me.Labels[index]
		votes[label]++
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
