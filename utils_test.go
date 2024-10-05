package bitknn_test

import "math/rand/v2"

var randSource = rand.New(rand.NewPCG(0xB0, 0xA4))

func randomData(size int) []uint64 {
	data := make([]uint64, size)
	for i := range data {
		data[i] = randSource.Uint64()
	}
	return data
}

func randomLabels(size int) []int {
	labels := make([]int, size)
	for i := range labels {
		labels[i] = int(randSource.Uint32N(256))
	}
	return labels
}

func randomValues(size int) []float64 {
	labels := make([]float64, size)
	for i := range labels {
		labels[i] = randSource.Float64()
	}
	return labels
}
