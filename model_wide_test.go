package bitknn_test

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"pgregory.net/rapid"
)

func Test_Model_64bit_Equal_To_Narrow(t *testing.T) {
	id := func(a uint64) uint64 { return a }
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(3, 1001).Draw(t, "k")
		data := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 10_000, id).Draw(t, "data")
		dataWide := make([][]uint64, len(data))
		for i := range data {
			dataWide[i] = []uint64{data[i]}
		}
		labels := rapid.SliceOfN(rapid.IntRange(0, 3), len(data), len(data)).Draw(t, "labels")
		queries := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 64, id).Draw(t, "queries")
		wideVotes := make([]float64, 4)
		narrowVotes := make([]float64, 4)
		type pair struct {
			name   string
			Narrow *bitknn.Model
			Wide   *bitknn.WideModel
		}
		pairs := []pair{
			{
				"",
				bitknn.Fit(data, labels),
				bitknn.FitWide(dataWide, labels),
			},
		}
		const eps = 1e-9
		for _, pair := range pairs {
			narrow := pair.Narrow
			wide := pair.Wide
			narrow.PreallocateHeap(k)
			wide.PreallocateHeap(k)
			for _, q := range queries {
				narrow.Predict1(k, q, bitknn.VoteSlice(narrowVotes))
				wide.Predict1(k, []uint64{q}, bitknn.VoteSlice(wideVotes))
				slices.Sort(narrow.HeapDistances[:k])
				slices.Sort(wide.Narrow.HeapDistances[:k])
				if !reflect.DeepEqual(narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k]) {
					t.Fatal("Wide KNN should result in the same distances for the nearest neighbors: ", narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k])
				}
				if !reflect.DeepEqual(narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k]) {
					t.Fatal("Wide ANN should result in the same indices for the nearest neighbors: ", narrow.HeapIndices[:k], wide.Narrow.HeapIndices[:k])
				}
				for i, vk := range narrowVotes {
					va := wideVotes[i]
					if math.Abs(vk-va) > eps {
						t.Fatalf("%s: %v: %v %v", pair.name, q, narrowVotes, wideVotes)
					}
				}
			}
		}

	})
}
