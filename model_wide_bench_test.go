package bitknn_test

import (
	"fmt"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/testrandom"
	"github.com/keilerkonzept/bitknn/pack"
)

func BenchmarkWideModel(b *testing.B) {
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
					data := testrandom.WideData(dim, dataSize)
					pack.ReallocateFlat(data)
					labels := testrandom.Labels(dataSize)
					model := bitknn.FitWide(data, labels)
					query := testrandom.WideQuery(dim)
					b.Run(fmt.Sprintf("Op=Predict_bits=%d_N=%d_k=%d", dim*64, dataSize, k), func(b *testing.B) {
						model.PreallocateHeap(k)
						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Predict(k, query, bitknn.DiscardVotes)
						}
					})
					b.Run(fmt.Sprintf("Op=Find_bits=%d_N=%d_k=%d", dim*64, dataSize, k), func(b *testing.B) {
						model.PreallocateHeap(k)
						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Find(k, query)
						}
					})
				}
			}
		}
	}
}
