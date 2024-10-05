package bitknn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/keilerkonzept/bitknn"
)

func TestNearest(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	k := 2
	x := uint64(0b0001)
	distances := make([]int, k+1)
	indices := make([]int, k+1)

	count := bitknn.Nearest(data, k, x, distances, indices)
	distances = distances[:count]
	indices = indices[:count]

	if count != k {
		t.Errorf("Expected count %d, got %d", k, count)
	}

	expectedDistances := []int{1, 1}
	if diff := cmp.Diff(expectedDistances, distances); diff != "" {
		t.Error(diff)
	}
	expectedIndices := []int{2, 0}
	if diff := cmp.Diff(expectedIndices, indices); diff != "" {
		t.Error(diff)
	}
}

func BenchmarkNearest(b *testing.B) {
	dataSize := 10_000
	k := 5
	query := rand.Uint64()
	data := randomData(dataSize)
	distances := make([]int, k+1)
	indices := make([]int, k+1)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		bitknn.Nearest(data, k, query, distances, indices)
	}
}
