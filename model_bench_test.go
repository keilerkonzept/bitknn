package bitknn_test

import (
	"fmt"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/testrandom"
)

func BenchmarkModel(b *testing.B) {
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
				data := testrandom.Data(dataSize)
				labels := testrandom.Labels(dataSize)
				model := bitknn.Fit(data, labels)
				query := testrandom.Query()

				b.Run(fmt.Sprintf("Op=Predict_bits=64_N=%d_k=%d", dataSize, k), func(b *testing.B) {
					model.PreallocateHeap(k)
					b.ResetTimer()
					for n := 0; n < b.N; n++ {
						model.Predict(k, query, bitknn.DiscardVotes)
					}
				})
				b.Run(fmt.Sprintf("Op=Find_bits=64_N=%d_k=%d", dataSize, k), func(b *testing.B) {
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
