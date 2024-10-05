package bitknn

// The k-NN model state.
type Model struct {
	// Number of nearest neighbors to consider.
	k int
	// Input data points.
	data []uint64
	// Class labels for each data point.
	labels []int
	// Optional vote values for each data point.
	values []float64

	distanceWeightFuncAny   bool
	distanceWeightFunc      func(int) float64
	distanceWeightLinear    bool
	distanceWeightQuadratic bool

	distances []int     // len = k+1
	indices   []int     // len = k+1
	votes     []float64 // len = k+1
}

// The type of options for the k-NN model.
type Option func(*Model)

func linearDecay(dist int) float64    { return 1 / float64(1+dist) }
func quadraticDecay(dist int) float64 { return 1 / float64(1+dist*dist) }

// Set linear decay as the distance weight function.
func WithLinearDecay() Option {
	return func(m *Model) {
		m.distanceWeightFuncAny = true
		m.distanceWeightFunc = nil
		m.distanceWeightLinear = true
		m.distanceWeightQuadratic = false
	}
}

// Set quadratic decay as the distance weight function.
func WithQuadraticDecay() Option {
	return func(m *Model) {
		m.distanceWeightFuncAny = true
		m.distanceWeightFunc = nil
		m.distanceWeightLinear = false
		m.distanceWeightQuadratic = true

	}
}

// Set a custom distance weight function.
func WithDistanceWeightFunc(f func(distance int) float64) Option {
	return func(m *Model) {
		m.distanceWeightFuncAny = f != nil
		m.distanceWeightFunc = f
		m.distanceWeightLinear = false
		m.distanceWeightQuadratic = false
	}
}

// Set vote values for each data point.
func WithValues(values []float64) Option {
	return func(m *Model) {
		m.values = values
	}
}

// Build a k-NN model from the given data and options.
func Fit(data []uint64, labels []int, k int, opts ...Option) *Model {
	m := &Model{
		k:      k,
		data:   data,
		labels: labels,

		distances: make([]int, k+1),
		indices:   make([]int, k+1),
		votes:     make([]float64, k+1),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Predicts the label of a single input point, using the scratch space pre-allocated with the [Model] for the neighbor heap.
func (me *Model) Predict1(x uint64, votes []float64) {
	me.Predict1Into(x, me.distances, me.indices, votes)
}

// Predicts the label of a single input point, using the given slices for the neighbor heap.
func (me *Model) Predict1Into(x uint64, distances []int, indices []int, votes []float64) {
	k := Nearest(me.data, me.k, x, distances, indices)

	clear(votes)
	if me.values == nil {
		me.countWeightedVotes(k, indices, distances, votes)
		return
	}

	me.sumWeightedValues(k, indices, distances, votes)
}

func (me *Model) sumWeightedValues(k int, indices []int, distances []int, votes []float64) {
	w := me.distanceWeightFunc
	switch {
	case !me.distanceWeightFuncAny:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + me.values[index]
			votes[label] = v
		}
	case me.distanceWeightLinear:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + me.values[index]*linearDecay(distances[i])
			votes[label] = v
		}
	case me.distanceWeightQuadratic:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + me.values[index]*quadraticDecay(distances[i])
			votes[label] = v
		}
	case w != nil:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + me.values[index]*w(distances[i])
			votes[label] = v
		}
	}
}

func (me *Model) countWeightedVotes(k int, indices []int, distances []int, votes []float64) {
	w := me.distanceWeightFunc
	switch {
	case !me.distanceWeightFuncAny:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			votes[label]++
		}
	case me.distanceWeightLinear:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + linearDecay(distances[i])
			votes[label] = v
		}
	case me.distanceWeightQuadratic:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + quadraticDecay(distances[i])
			votes[label] = v
		}
	case w != nil:
		for i := range k {
			index := indices[i]
			label := me.labels[index]
			v := votes[label] + w(distances[i])
			votes[label] = v
		}
	}
}
