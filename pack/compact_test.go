package pack_test

import (
	"reflect"
	"slices"
	"testing"

	"github.com/keilerkonzept/bitknn/pack"
	"pgregory.net/rapid"
)

func TestPackReallocateFlat(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		dims := rapid.IntRange(3, 100).Draw(t, "dims")
		n := rapid.IntRange(0, 1000).Draw(t, "n")
		data := rapid.SliceOfN(rapid.SliceOfN(rapid.Uint64(), dims, dims), n, n).Draw(t, "data")

		dataCopy := make([][]uint64, len(data))
		for i := range dataCopy {
			dataCopy[i] = slices.Clone(data[i])
		}
		pack.ReallocateFlat(data)
		if !reflect.DeepEqual(data, dataCopy) {
			t.Fatalf("Original: %v, Packed: %v", dataCopy, data)
		}
	})
}
