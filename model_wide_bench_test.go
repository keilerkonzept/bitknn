package bitknn_test

import (
	"fmt"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/testrandom"
)

func Benchmark_WideModel_Predict(b *testing.B) {
	type bench struct {
		dim      []int
		dataSize []int
		k        []int
	}
	benches := []bench{
		{dim: []int{1, 2, 10}, dataSize: []int{100}, k: []int{3, 10}},
		{dim: []int{1, 2, 10}, dataSize: []int{1000, 1_000_000}, k: []int{3, 10, 100}},
	}
	for _, bench := range benches {
		for _, dim := range bench.dim {
			for _, dataSize := range bench.dataSize {
				for _, k := range bench.k {
					b.Run(fmt.Sprintf("dim=%d_N=%d_k=%d", dim*64, dataSize, k), func(b *testing.B) {
						data := testrandom.WideData(dim, dataSize)
						labels := testrandom.Labels(dataSize)
						model := bitknn.FitWide(data, labels)
						query := testrandom.WideQuery(dim)

						model.PreallocateHeap(k)
						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Predict(k, query, bitknn.DiscardVotes)
						}
					})
				}
			}
		}
	}
}
