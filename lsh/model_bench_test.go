package lsh_test

import (
	"fmt"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/internal/testrandom"
	"github.com/keilerkonzept/bitknn/lsh"
)

func BenchmarkModel(b *testing.B) {
	type bench struct {
		hashes   []lsh.Hash
		dataSize []int
		k        []int
	}
	hashes := []lsh.Hash{
		lsh.ConstantHash{}, // should be only a bit slower than exact KNN
	}
	benches := []bench{
		{hashes: hashes, dataSize: []int{100}, k: []int{1, 3, 10}},
		{hashes: hashes, dataSize: []int{1_000_000}, k: []int{3, 10, 100}},
	}
	for _, bench := range benches {
		for _, dataSize := range bench.dataSize {
			data := testrandom.Data(dataSize)
			labels := testrandom.Labels(dataSize)
			query := testrandom.Query()
			for _, k := range bench.k {
				for _, hash := range bench.hashes {
					b.Run(fmt.Sprintf("Op=Predict_hash=%T_N=%d_k=%d", hash, dataSize, k), func(b *testing.B) {
						model := lsh.Fit(data, labels, hash)
						model.PreallocateHeap(k)
						b.ResetTimer()
						for n := 0; n < b.N; n++ {
							model.Predict(k, query, bitknn.DiscardVotes)
						}
					})
					b.Run(fmt.Sprintf("Op=Find_hash=%T_N=%d_k=%d", hash, dataSize, k), func(b *testing.B) {
						model := lsh.Fit(data, labels, hash)
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
