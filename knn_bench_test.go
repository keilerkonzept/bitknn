package bitknn_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/keilerkonzept/bitknn"
)

func BenchmarkPredict1(b *testing.B) {
	votes := make([]float64, 256)
	for _, dataSize := range []int{100, 1000, 1_000_000} {
		for _, k := range []int{3, 10, 100} {
			b.Run(fmt.Sprintf("N=%d_k=%d", dataSize, k), func(b *testing.B) {
				data := randomData(dataSize)
				labels := randomLabels(dataSize)
				model := bitknn.Fit(data, labels, k)
				query := rand.Uint64()

				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					model.Predict1(query, votes)
				}
			})
		}
	}
}
