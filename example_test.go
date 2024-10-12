package bitknn_test

import (
	"fmt"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/pack"
)

func Example() {
	// feature vectors packed into uint64s
	data := []uint64{0b101010, 0b111000, 0b000111}
	// class labels
	labels := []int{0, 1, 1}

	model := bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting())

	// one vote counter per class
	votes := make([]float64, 2)

	k := 2
	model.Predict(k, 0b101011, bitknn.VoteSlice(votes))
	// or, just return the nearest neighbor's distances and indices:
	// distances,indices := model.Find(k, 0b101011)

	fmt.Println("Votes:", votes)

	// you can also use a map for the votes.
	// this is good if you have a very large number of different labels:
	votesMap := make(map[int]float64)
	model.Predict(k, 0b101011, bitknn.VoteMap(votesMap))
	fmt.Println("Votes for 0:", votesMap[0])
	// Output:
	// Votes: [0.5 0.25]
	// Votes for 0: 0.5
}

func ExampleFitWide() {
	// feature vectors packed into uint64s
	data := [][]uint64{
		pack.String("foo"),
		pack.String("bar"),
		pack.String("baz"),
	}
	// class labels
	labels := []int{0, 1, 1}

	model := bitknn.FitWide(data, labels, bitknn.WithLinearDistanceWeighting())

	// one vote counter per class
	votes := make([]float64, 2)

	k := 2
	query := pack.String("fob")
	model.Predict(k, query, bitknn.VoteSlice(votes))

	fmt.Println("Votes:", votes)

	// Output:
	// Votes: [0.25 0.16666666666666666]
}
