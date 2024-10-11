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

func Test_Model_NoHash_IsExact(t *testing.T) {
	var h lsh.NoHash
	var h0 lsh.ConstantHash
	id := func(a uint64) uint64 { return a }
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(1, 1001).Draw(t, "k")
		data := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 1000, id).Draw(t, "data")
		labels := rapid.SliceOfN(rapid.IntRange(0, 3), len(data), len(data)).Draw(t, "labels")
		values := rapid.SliceOfN(rapid.Float64(), len(data), len(data)).Draw(t, "values")
		queries := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 64, id).Draw(t, "queries")
		knnVotes := make([]float64, 4)
		annVotes := make([]float64, 4)
		type pair struct {
			name string
			KNN  *bitknn.Model
			ANN  *lsh.Model
			ANN0 *lsh.Model
		}
		pairs := []pair{
			{
				"V",
				bitknn.Fit(data, labels, bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithValues(values)),
				lsh.Fit(data, labels, h0, bitknn.WithValues(values)),
			},
			{
				"LV",
				bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h0, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values)),
			},
			{
				"QV",
				bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h0, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values)),
			},
			{
				"CV",
				bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h0, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear), bitknn.WithValues(values)),
			},
			{
				"0",
				bitknn.Fit(data, labels),
				lsh.Fit(data, labels, h),
				lsh.Fit(data, labels, h0),
			},
			{
				"L",
				bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting()),
				lsh.Fit(data, labels, h, bitknn.WithLinearDistanceWeighting()),
				lsh.Fit(data, labels, h0, bitknn.WithLinearDistanceWeighting()),
			},
			{
				"Q",
				bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting()),
				lsh.Fit(data, labels, h, bitknn.WithQuadraticDistanceWeighting()),
				lsh.Fit(data, labels, h0, bitknn.WithQuadraticDistanceWeighting()),
			},
			{
				"C",
				bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear)),
				lsh.Fit(data, labels, h, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear)),
				lsh.Fit(data, labels, h0, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear)),
			},
		}
		const eps = 1e-8
		for _, pair := range pairs {
			knn := pair.KNN
			ann := pair.ANN
			ann0 := pair.ANN0
			knn.PreallocateHeap(k)
			ann.PreallocateHeap(k)
			for _, q := range queries {
				knn.Predict(k, q, bitknn.VoteSlice(knnVotes))
				ann.Predict(k, q, bitknn.VoteSlice(annVotes))
				slices.Sort(knn.HeapDistances[:k])
				slices.Sort(ann.HeapDistances[:k])
				if !reflect.DeepEqual(knn.HeapDistances[:k], ann.HeapDistances[:k]) {
					t.Fatal("NoHash ANN should result in the same distances for the nearest neighbors: ", knn.HeapDistances[:k], ann.HeapDistances[:k], knn.HeapIndices[:k], ann.HeapIndices[:k])
				}

				kd, ki := knn.Find(k, q)
				ad, ai := ann.Find(k, q)
				slices.Sort(kd)
				slices.Sort(ad)
				if !reflect.DeepEqual(kd, ad) {
					t.Fatal("NoHash ANN should result in the same distances for the nearest neighbors: ", kd, ad)
				}
				slices.Sort(ki)
				slices.Sort(ai)
				if !reflect.DeepEqual(ki, ai) {
					t.Fatal("NoHash ANN should result in the same indices for the nearest neighbors: ", ki, ai)
				}

				ann0.PredictAlloc(k, q, bitknn.VoteSlice(annVotes))
				for i, vk := range knnVotes {
					va := annVotes[i]
					if math.Abs(vk-va) > eps {
						t.Fatalf("ANN: %s: %v: %v %v", pair.name, q, knnVotes, annVotes)
					}
				}
				ann0.Predict(k, q, bitknn.VoteSlice(annVotes))
				for i, vk := range knnVotes {
					va := annVotes[i]
					if math.Abs(vk-va) > eps {
						t.Fatalf("ANN0: %s: %v: %v %v", pair.name, q, knnVotes, annVotes)
					}
				}
			}
		}

	})
}
