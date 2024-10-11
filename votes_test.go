package bitknn_test

import (
	"math"
	"testing"

	"github.com/keilerkonzept/bitknn"
	"pgregory.net/rapid"
)

const eps = 1e-9

func TestVoteDiscard(t *testing.T) {
	discard := bitknn.DiscardVotes
	{
		pre := discard
		discard.Add(0, 1)
		post := discard
		if pre != post {
			t.Fail()
		}
	}
	if discard.Get(0) != 0 {
		t.Fatal()
	}
	if discard.ArgMax() != 0 {
		t.Fatal()
	}
	if discard.Max() != 0 {
		t.Fatal()
	}
	{
		pre := discard
		discard.Clear()
		post := discard
		if pre != post {
			t.Fail()
		}
	}
}

func TestVoteSlice_Clear(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		length := rapid.IntRange(0, 100).Draw(t, "length")
		voteSlice := make(bitknn.VoteSlice, length)
		for i := 0; i < length; i++ {
			voteSlice[i] = rapid.Float64().Draw(t, "vote")
		}

		voteSlice.Clear()
		for i := 0; i < length; i++ {
			if voteSlice[i] != 0 {
				t.Fatalf("expected voteSlice[%d] to be 0, got %f", i, voteSlice[i])
			}
		}
	})
}

func TestVoteSlice_ArgMax(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		length := rapid.IntRange(0, 100).Draw(t, "length")
		voteSlice := make(bitknn.VoteSlice, length)
		for i := 0; i < length; i++ {
			voteSlice[i] = rapid.Float64Range(-1000, 1000).Draw(t, "vote")
		}

		maxIdx := voteSlice.ArgMax()
		if length == 0 && maxIdx != 0 {
			t.Fatal()
		}

		for i := range voteSlice {
			if voteSlice[i] > voteSlice[maxIdx] {
				t.Fatalf("expected max index to be %d, but found a larger value at index %d", maxIdx, i)
			}
		}
	})
}

func TestVoteSlice_Max(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		length := rapid.IntRange(1, 100).Draw(t, "length")
		voteSlice := make(bitknn.VoteSlice, length)
		for i := 0; i < length; i++ {
			voteSlice[i] = rapid.Float64Range(-1000, 1000).Draw(t, "vote")
		}

		maxValue := voteSlice.Max()

		expectedMax := voteSlice[0]
		for _, val := range voteSlice[1:] {
			if val > expectedMax {
				expectedMax = val
			}
		}
		if math.Abs(expectedMax-maxValue) > eps {
			t.Fatalf("expected max value %f, got %f", expectedMax, maxValue)
		}
	})
}

func TestVoteSlice_Add(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		length := rapid.IntRange(1, 100).Draw(t, "length")
		voteSlice := make(bitknn.VoteSlice, length)
		class := rapid.IntRange(0, length-1).Draw(t, "class")
		delta := rapid.Float64().Draw(t, "delta")

		oldValue := voteSlice[class]
		voteSlice.Add(class, delta)
		newValue := voteSlice[class]

		if math.Abs((oldValue+delta)-newValue) > eps {
			t.Fatalf("expected new value to be %f, got %f", oldValue+delta, newValue)
		}
	})
}

func TestVoteSlice_Get(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		length := rapid.IntRange(1, 100).Draw(t, "length")
		voteSlice := make(bitknn.VoteSlice, length)
		for i := 0; i < length; i++ {
			voteSlice[i] = rapid.Float64().Draw(t, "vote")
		}
		class := rapid.IntRange(0, length-1).Draw(t, "class")

		value := voteSlice.Get(class)
		if math.Abs(voteSlice[class]-value) > eps {
			t.Fatalf("expected value %f, got %f", voteSlice[class], value)
		}
	})
}

func TestVoteMap_Clear(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		voteMap := make(bitknn.VoteMap)
		for i := 0; i < rapid.IntRange(0, 100).Draw(t, "length"); i++ {
			voteMap[i] = rapid.Float64().Draw(t, "vote")
		}

		voteMap.Clear()
		if len(voteMap) != 0 {
			t.Fatalf("expected VoteMap to be empty after Clear")
		}
	})
}

func TestVoteMap_ArgMax(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		voteMap := make(bitknn.VoteMap)
		length := rapid.IntRange(0, 100).Draw(t, "length")
		for i := 0; i < length; i++ {
			voteMap[i] = rapid.Float64Range(-1000, 1000).Draw(t, "vote")
		}

		maxIdx := voteMap.ArgMax()
		if length == 0 && maxIdx != 0 {
			t.Fatal()
		}

		for i, v := range voteMap {
			if v > voteMap[maxIdx] {
				t.Fatalf("expected max index to be %d, but found a larger value at index %d", maxIdx, i)
			}
		}
	})
}

func TestVoteMap_Max(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		voteMap := make(bitknn.VoteMap)
		length := rapid.IntRange(1, 100).Draw(t, "length")
		for i := 0; i < length; i++ {
			voteMap[i] = rapid.Float64Range(-1000, 1000).Draw(t, "vote")
		}

		maxValue := voteMap.Max()

		var expectedMax float64
		first := true
		for _, v := range voteMap {
			if first || v > expectedMax {
				expectedMax = v
				first = false
			}
		}
		if math.Abs(expectedMax-maxValue) > eps {
			t.Fatalf("expected max value %f, got %f", expectedMax, maxValue)
		}
	})
}

func TestVoteMap_Add(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		voteMap := make(bitknn.VoteMap)
		class := rapid.IntRange(0, 100).Draw(t, "class")
		delta := rapid.Float64().Draw(t, "delta")

		oldValue := voteMap[class]
		voteMap.Add(class, delta)
		newValue := voteMap[class]

		if math.Abs((oldValue+delta)-newValue) > eps {
			t.Fatalf("expected new value to be %f, got %f", oldValue+delta, newValue)
		}
	})
}

func TestVoteMap_Get(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		voteMap := make(bitknn.VoteMap)
		for i := 0; i < rapid.IntRange(0, 100).Draw(t, "length"); i++ {
			voteMap[i] = rapid.Float64().Draw(t, "vote")
		}
		class := rapid.IntRange(0, 100).Draw(t, "class")

		value := voteMap.Get(class)
		if math.Abs(voteMap[class]-value) > eps {
			t.Fatalf("expected value %f, got %f", voteMap[class], value)
		}
	})
}
