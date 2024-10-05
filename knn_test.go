package bitknn_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/keilerkonzept/bitknn"
)

func TestPredict1(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 2

	model := bitknn.Fit(data, labels, k)

	x := uint64(0b0010)
	votes := make([]float64, k)
	model.Predict1(x, votes)

	expectedVotes := []float64{1, 1}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithValues(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 4.0}
	k := 2

	model := bitknn.Fit(data, labels, k, bitknn.WithValues(values))

	x := uint64(0b0010)
	votes := make([]float64, k)
	model.Predict1(x, votes)

	expectedVotes := []float64{1, 3}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithLinearDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithLinearDecay())

	x := uint64(0b0001)
	votes := make([]float64, 2)
	model.Predict1(x, votes)

	expectedVotes := []float64{1, 0.5}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithValuesAndLinearDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 3.0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithValues(values), bitknn.WithLinearDecay())

	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.Predict1(x, votes)

	expectedVotes := []float64{2, 1}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithValuesAndQuadraticDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 4.0, 5.0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithValues(values), bitknn.WithQuadraticDecay())
	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.Predict1(x, votes)

	expectedVotes := []float64{2, 0.8}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithQuadraticDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithQuadraticDecay())
	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.Predict1(x, votes)

	expectedVotes := []float64{1.2, 0.2}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithValuesAndCustomDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 1, 0}
	values := []float64{1.0, 2.0, 3.0, 3.0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithValues(values), bitknn.WithDistanceWeightFunc(func(d int) float64 {
		if d <= 2 {
			return 1.0
		} else {
			return 0.0
		}
	}))

	x := uint64(0b0000)
	votes := make([]float64, 2)
	model.Predict1(x, votes)

	expectedVotes := []float64{4, 3}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}

func TestPredict1WithCustomDecay(t *testing.T) {
	data := []uint64{0b0000, 0b1111, 0b0011, 0b0101}
	labels := []int{0, 1, 2, 0}
	k := 3

	model := bitknn.Fit(data, labels, k, bitknn.WithDistanceWeightFunc(func(d int) float64 {
		if d <= 2 {
			return 1.0
		} else {
			return 0.0
		}
	}))

	x := uint64(0b0000)
	votes := make([]float64, 3)
	model.Predict1(x, votes)

	expectedVotes := []float64{2, 0, 1}
	if diff := cmp.Diff(expectedVotes, votes); diff != "" {
		t.Error(diff)
	}
}
