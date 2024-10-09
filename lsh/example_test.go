package lsh_test

import (
	"fmt"

	"github.com/keilerkonzept/bitknn"
	"github.com/keilerkonzept/bitknn/lsh"
)

func Example() {
	// feature vectors packed into uint64s
	data := []uint64{0b101010, 0b111000, 0b000111}
	// class labels
	labels := []int{0, 1, 1}

	// Define a hash function
	hash := lsh.BitSample(0xF0F0F0)

	// Fit an LSH model
	model := lsh.Fit(data, labels, hash, bitknn.WithLinearDistanceWeighting())

	// one vote counter per class
	votes := make([]float64, 2)

	k := 2
	model.Predict1(k, 0b101011, bitknn.VoteSlice(votes))

	fmt.Println("Votes:", bitknn.VoteSlice(votes))

	// you can also use a map for the votes
	votesMap := make(map[int]float64)
	model.Predict1(k, 0b101011, bitknn.VoteMap(votesMap))
	fmt.Println("Votes for 0:", votesMap[0])
	// Output:
	// Votes: [0.5 0.25]
	// Votes for 0: 0.5
}
