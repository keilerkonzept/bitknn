package lsh_test

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/lsh"
	"pgregory.net/rapid"
)

func Test_WideModel_64bit_Equal_To_Narrow(t *testing.T) {
	id := func(a uint64) uint64 { return a }
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(1, 1001).Draw(t, "k")
		data := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 1000, id).Draw(t, "data")
		dataWide := make([][]uint64, len(data))
		for i := range data {
			dataWide[i] = []uint64{data[i]}
		}
		labels := rapid.SliceOfN(rapid.IntRange(0, 3), len(data), len(data)).Draw(t, "labels")
		values := rapid.SliceOfN(rapid.Float64(), len(data), len(data)).Draw(t, "values")
		queries := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 64, id).Draw(t, "queries")
		wideVotes := make([]float64, 4)
		narrowVotes := make([]float64, 4)
		type pair struct {
			name   string
			Narrow *lsh.Model
			Wide   *lsh.WideModel
		}
		pairs := []pair{
			{
				"",
				lsh.Fit(data, labels, lsh.ConstantHash{}, bitknn.WithValues(values)),
				lsh.FitWide(dataWide, labels, lsh.ConstantHash{}, bitknn.WithValues(values)),
			},
		}
		const eps = 1e-9
		for _, pair := range pairs {
			narrow := pair.Narrow
			wide := pair.Wide
			narrow.PreallocateHeap(k)
			wide.PreallocateHeap(k)
			for _, q := range queries {
				nd, ni := narrow.Find(k, q)
				wd, wi := wide.Find(k, []uint64{q})
				if !reflect.DeepEqual(nd, wd) {
					t.Fatal("Wide KNN should result in the same distances for the nearest neighbors: ", nd, wd)
				}
				if !reflect.DeepEqual(ni, wi) {
					t.Fatal("Wide ANN should result in the same indices for the nearest neighbors: ", ni, wi)
				}
				narrow.Predict(k, q, bitknn.VoteSlice(narrowVotes))
				wide.Predict(k, []uint64{q}, bitknn.VoteSlice(wideVotes))
				if !reflect.DeepEqual(narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k]) {
					t.Fatal("Wide KNN should result in the same distances for the nearest neighbors: ", narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k])
				}
				if !reflect.DeepEqual(narrow.HeapIndices[:k], wide.Narrow.HeapIndices[:k]) {
					t.Fatal("Wide ANN should result in the same indices for the nearest neighbors: ", narrow.HeapIndices[:k], wide.Narrow.HeapIndices[:k])
				}
				for i, vk := range narrowVotes {
					va := wideVotes[i]
					if math.Abs(vk-va) > eps {
						t.Fatalf("%s: %v: %v %v", pair.name, q, narrowVotes, wideVotes)
					}
				}
				wide.PredictAlloc(k, []uint64{q}, bitknn.VoteSlice(wideVotes))
				slices.Sort(narrow.HeapDistances[:k])
				slices.Sort(wide.Narrow.HeapDistances[:k])
				if !reflect.DeepEqual(narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k]) {
					t.Fatal("Wide KNN should result in the same distances for the nearest neighbors: ", narrow.HeapDistances[:k], wide.Narrow.HeapDistances[:k])
				}
				slices.Sort(narrow.HeapIndices[:k])
				slices.Sort(wide.Narrow.HeapIndices[:k])
				if !reflect.DeepEqual(narrow.HeapIndices[:k], wide.Narrow.HeapIndices[:k]) {
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
