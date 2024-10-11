package bitknn_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/keilerkonzept/bitknn"
)

func Test_DistanceWeighting_String(t *testing.T) {
	ds := []bitknn.DistanceWeighting{
		bitknn.DistanceWeightingNone,
		bitknn.DistanceWeightingLinear,
		bitknn.DistanceWeightingQuadratic,
		bitknn.DistanceWeightingCustom,
		-1, // invalid
	}
	names := []string{"none", "linear", "quadratic", "custom", "unknown"}
	for i, d := range ds {
		if d.String() != names[i] {
			t.Errorf("%q != %q", d.String(), names[i])
		}
	}
}

func Test_Model_Predict_PredictRealloc(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 2

	model := bitknn.Fit(data, labels)

	x := uint64(0b0010)
	votes := make([]float64, k)
	model.PreallocateHeap(k)
	{
		model.Predict(k, x, bitknn.VoteSlice(votes))

		expectedVotes := []float64{1, 1}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}
	{
		model.PredictAlloc(k, x, bitknn.VoteSlice(votes))

		expectedVotes := []float64{1, 1}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}
}

func Test_Model_Reslice_Predict(t *testing.T) {
	data := []uint64{0b0000, 0b11111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 4.0}

	model := bitknn.Fit(data, labels, bitknn.WithValues(values))

	x := uint64(0b0010)
	votes := make([]float64, 2)
	model.PreallocateHeap(3)
	model.Predict(2, x, bitknn.VoteSlice(votes))
	{
		expectedVotes := []float64{1, 3}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}
	model.Predict(3, x, bitknn.VoteSlice(votes))
	{
		expectedVotes := []float64{5, 3}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}
	model.Predict(2, x, bitknn.VoteSlice(votes))

	{
		expectedVotes := []float64{1, 3}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}

	{
		model.Predict(10, x, bitknn.VoteSlice(votes))
		expectedVotes := []float64{5, 5}
		if diff := cmp.Diff(expectedVotes, votes); diff != "" {
			t.Error(diff)
		}
	}
}

func Test_Model_PredictV(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 4.0}
	k := 2

	model := bitknn.Fit(data, labels, bitknn.WithValues(values))

	x := uint64(0b0010)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{1, 3}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictD(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 3

	model := bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting())

	x := uint64(0b0001)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{1, 0.5}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictVL(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 3.0}
	k := 3

	model := bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting(), bitknn.WithValues(values))

	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{2, 1}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictVQ(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 4.0, 5.0}
	k := 3

	model := bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting(), bitknn.WithValues(values))
	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{2, 0.8}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictQ(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 3

	model := bitknn.Fit(data, labels, bitknn.WithQuadraticDistanceWeighting())
	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{1.2, 0.2}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictVC(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 3.0}
	k := 3

	f := func(d int) float64 {
		if d <= 2 {
			return 1.0
		} else {
			return 0.0
		}
	}
	model := bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(f), bitknn.WithValues(values))

	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{4, 3}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func Test_Model_PredictC(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 2, 0}
	k := 3

	f := func(d int) float64 {
		if d <= 2 {
			return 1.0
		} else {
			return 0.0
		}
	}
	model := bitknn.Fit(data, labels, bitknn.WithDistanceWeightingFunc(f))

	x := uint64(0b0000)
	votes := make([]float64, 3)
	model.PreallocateHeap(k)
	model.Predict(k, x, bitknn.VoteSlice(votes))

	expectedVotes := []float64{2, 0, 1}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}
