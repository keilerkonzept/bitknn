package testrandom

import "math/rand/v2"

var Source = rand.New(rand.NewPCG(0xB0, 0xA4))

func Query() uint64 {
	return Source.Uint64()
}

func WideQuery(dim int) []uint64 {
	return Data(dim)
}

func Data(size int) []uint64 {
	data := make([]uint64, size)
	for i := range data {
		data[i] = Source.Uint64()
	}
	return data
}

func WideData(dim int, size int) [][]uint64 {
	data := make([][]uint64, size)
	for i := range data {
		data[i] = make([]uint64, dim)
		for j := range dim {
			data[i][j] = Source.Uint64()
		}
	}
	return data
}

func Labels(size int) []int {
	labels := make([]int, size)
	for i := range labels {
		labels[i] = int(Source.Uint32N(256))
	}
	return labels
}

func Values(size int) []float64 {
	labels := make([]float64, size)
	for i := range labels {
		labels[i] = Source.Float64()
	}
	return labels
}
