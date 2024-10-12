package bitknn_test

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"pgregory.net/rapid"
)

func TestModel_64bitWideEquivNarrow(t *testing.T) {
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
				nd, ni := narrow.Find(k, q)
				wd, wi := wide.Find(k, []uint64{q})
				if !reflect.DeepEqual(nd, wd) {
					t.Fatal("Wide model should result in the same distances for the nearest neighbors as the narrow model: ", nd, wd)
				}
				if !reflect.DeepEqual(ni, wi) {
					t.Fatal("Wide model should result in the same indices for the nearest neighbors as the narrow model: ", ni, wi)
				}
				narrow.Predict(k, q, bitknn.VoteSlice(narrowVotes))
				wide.Predict(k, []uint64{q}, bitknn.VoteSlice(wideVotes))
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

func TestModel_FindV_Equiv_Find_0Remainder(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(1, 100).Draw(t, "k")
		n := rapid.IntRange(2, 100).Draw(t, "n")
		dims := rapid.IntRange(1, 10).Draw(t, "dims")
		data := rapid.SliceOfN(rapid.SliceOfN(rapid.Uint64(), dims, dims), n*k, n*k).Draw(t, "data")
		q := rapid.SliceOfN(rapid.Uint64(), dims, dims).Draw(t, "q")
		batchSize := k
		m1 := bitknn.FitWide(data, nil)
		m2 := bitknn.FitWide(data, nil)
		batch := make([]uint32, batchSize)
		vds, vis := m1.FindV(k, q, batch)
		ds, is := m2.Find(k, q)
		if !reflect.DeepEqual(vds, ds) {
			t.Fatal(vds, ds)
		}
		if !reflect.DeepEqual(vis, is) {
			t.Fatal(vis, is)
		}
	})
}

func TestModel_FindV_Equiv_Find(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(0, 1000).Draw(t, "k")
		dims := rapid.IntRange(1, 10_000).Draw(t, "dims")
		data := rapid.SliceOf(rapid.SliceOfN(rapid.Uint64(), dims, dims)).Draw(t, "data")
		batchSizes := []int{0, len(data), len(data) - 1, len(data) - 2, 2048, 100_000}
		q := rapid.SliceOfN(rapid.Uint64(), dims, dims).Draw(t, "q")
		for _, batchSize := range batchSizes {
			batchSize = max(k, batchSize)
			m1 := bitknn.FitWide(data, nil)
			m2 := bitknn.FitWide(data, nil)
			batch := make([]uint32, batchSize)
			vds, vis := m1.FindV(k, q, batch)
			ds, is := m2.Find(k, q)
			if !reflect.DeepEqual(vds, ds) {
				t.Fatal(vds, ds)
			}
			if !reflect.DeepEqual(vis, is) {
				t.Fatal(vis, is)
			}
			batchAll := make([]uint32, batchSize)
			vds, vis = m1.FindV(k, q, batchAll)
			if !reflect.DeepEqual(vds, ds) {
				t.Fatal(vds, ds)
			}
			if !reflect.DeepEqual(vis, is) {
				t.Fatal(vis, is)
			}
		}
	})
}
