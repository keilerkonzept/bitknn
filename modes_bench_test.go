package bitknn_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/keilerkonzept/bitknn"
)

func Benchmark_Model_Predict1(b *testing.B) {
	votes := make([]float64, 256)
	type bench struct {
		dataSize []int
		k        []int
	}
	benches := []bench{
		{dataSize: []int{100}, k: []int{3, 10}},
		{dataSize: []int{1000, 1_000_000}, k: []int{3, 10, 100}},
	}
	for _, bench := range benches {
		for _, dataSize := range bench.dataSize {
			for _, k := range bench.k {
				b.Run(fmt.Sprintf("N=%d_k=%d", dataSize, k), func(b *testing.B) {
					data := randomData(dataSize)
					labels := randomLabels(dataSize)
					model := bitknn.Fit(data, labels)
					query := rand.Uint64()

					model.PreallocateHeap(k)
					b.ResetTimer()
					for n := 0; n < b.N; n++ {
						model.Predict1(k, query, votes)
					}
				})
			}
		}
	}
}

func Benchmark_Model_Predict1V(b *testing.B) {
	votes := make([]float64, 256)
	type bench struct {
		dataSize []int
		k        []int
	}
	benches := []bench{
		{dataSize: []int{100}, k: []int{3, 10, 100}},
	}
	for _, bench := range benches {
		for _, dataSize := range bench.dataSize {
			for _, k := range bench.k {
				b.Run(fmt.Sprintf("N=%d_k=%d", dataSize, k), func(b *testing.B) {
					data := randomData(dataSize)
					labels := randomLabels(dataSize)
					values := randomValues(dataSize)
					model := bitknn.Fit(data, labels, bitknn.WithValues(values))
					query := rand.Uint64()

					model.PreallocateHeap(k)
					b.ResetTimer()
					for n := 0; n < b.N; n++ {
						model.Predict1(k, query, votes)
					}
				})
			}
		}
	}
}

func Benchmark_Model_Predict1D(b *testing.B) {
	votes := make([]float64, 256)
	type bench struct {
		dataSize []int
		k        []int
	}
	benches := []bench{
		{dataSize: []int{100}, k: []int{3, 10, 100}},
	}
	for _, d := range []bitknn.DistanceWeighting{bitknn.DistanceWeightingLinear, bitknn.DistanceWeightingQuadratic, bitknn.DistanceWeightingCustom} {
		for _, bench := range benches {
			for _, dataSize := range bench.dataSize {
				for _, k := range bench.k {
					b.Run(fmt.Sprintf("DistFunc=%v_N=%d_k=%d", d, dataSize, k), func(b *testing.B) {
						data := randomData(dataSize)
						labels := randomLabels(dataSize)
						model := bitknn.Fit(data, labels)
						model.DistanceWeighting = d
						model.DistanceWeightingFunc = func(d int) float64 { return 1 / float64(1+d) }
						query := rand.Uint64()

						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Predict1(k, query, votes)
						}
					})
				}
			}
		}
	}
}

func Benchmark_Model_Predict1DV(b *testing.B) {
	votes := make([]float64, 256)
	type bench struct {
		dataSize []int
		k        []int
	}
	benches := []bench{
		{dataSize: []int{100}, k: []int{3, 10, 100}},
	}
	for _, d := range []bitknn.DistanceWeighting{bitknn.DistanceWeightingLinear, bitknn.DistanceWeightingQuadratic, bitknn.DistanceWeightingCustom} {
		for _, bench := range benches {
			for _, dataSize := range bench.dataSize {
				for _, k := range bench.k {
					b.Run(fmt.Sprintf("DistFunc=%v_N=%d_k=%d", d, dataSize, k), func(b *testing.B) {
						data := randomData(dataSize)
						labels := randomLabels(dataSize)
						values := randomValues(dataSize)
						model := bitknn.Fit(data, labels, bitknn.WithValues(values))
						model.DistanceWeighting = d
						model.DistanceWeightingFunc = func(d int) float64 { return 1 / float64(1+d) }
						query := rand.Uint64()

						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Predict1(k, query, votes)
						}
					})
				}
			}
		}
	}
}
