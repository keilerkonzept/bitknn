package lsh_test

import (
	"reflect"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/lsh"
	"pgregory.net/rapid"
)

func Test_Model_NoHash_IsExact(t *testing.T) {
	var h lsh.NoHash
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(1, 100).Draw(t, "k")
		data := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 64, func(a uint64) uint64 { return a }).Draw(t, "data")
		labels := rapid.SliceOfN(rapid.IntRange(0, 3), len(data), len(data)).Draw(t, "labels")
		values := rapid.SliceOfN(rapid.Float64(), len(data), len(data)).Draw(t, "values")
		queries := rapid.SliceOfN(rapid.Uint64(), 3, 64).Draw(t, "queries")
		knnVotes := make([]float64, 4)
		annVotes := make([]float64, 4)
		type pair struct {
			KNN *bitknn.Model
			ANN *lsh.Model
		}
		pairs := []pair{
			{
				bitknn.Fit(data, labels, bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithValues(values)),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values)),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values)),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear), bitknn.WithValues(values)),
				lsh.Fit(data, labels, h, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear), bitknn.WithValues(values)),
			},
			{
				bitknn.Fit(data, labels),
				lsh.Fit(data, labels, h),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting()),
				lsh.Fit(data, labels, h, bitknn.WithLinearDistanceWeighting()),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting()),
				lsh.Fit(data, labels, h, bitknn.WithQuadraticDistanceWeighting()),
			},
			{
				bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear)),
				lsh.Fit(data, labels, h, bitknn.WithDistanceWeightingFunc(bitknn.DistanceWeightingFuncLinear)),
			},
		}
		for _, pair := range pairs {
			knn := pair.KNN
			ann := pair.ANN
			for _, q := range queries {
				knn.PreallocateHeap(k)
				knn.Predict1(k, q, knnVotes)
				ann.PreallocateHeap(k)
				ann.Predict1(k, q, annVotes)
				if !reflect.DeepEqual(knnVotes, annVotes) {
					t.Fatalf("%v %v", knnVotes, annVotes)
				}
				knn.Predict1Alloc(k, q, knnVotes)
				ann.Predict1Alloc(k, q, annVotes)
				if !reflect.DeepEqual(knnVotes, annVotes) {
					t.Fatalf("%v %v", knnVotes, annVotes)
				}
			}
		}

	})
}
