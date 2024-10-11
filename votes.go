package bitknn

import "slices"

// VoteCounter is a k-NN vote counter interface.
type VoteCounter interface {
	// Clear removes all votes.
	Clear()

	// ArgMax returns the label with the highest vote count.
	// If there are no votes, it returns 0.
	ArgMax() int

	// Max returns the highest vote count.
	Max() float64

	// Get returns the vote count for the given label.
	Get(label int) float64

	// Add adds the specified delta to the vote count for the given label.
	Add(label int, delta float64)
}

type discardVotes int

// DiscardVotes is a no-op vote counter.
const DiscardVotes = discardVotes(0)

func (me discardVotes) Clear()                       {}
func (me discardVotes) ArgMax() int                  { return 0 }
func (me discardVotes) Max() float64                 { return 0 }
func (me discardVotes) Get(label int) float64        { return 0 }
func (me discardVotes) Add(label int, delta float64) {}

// VoteSlice is a dense vote counter that stores votes in a slice.
// It is efficient for small sets of class labels.
type VoteSlice []float64

// Clear resets all the votes in the slice to zero.
func (me VoteSlice) Clear() {
	clear(me)
}

// ArgMax returns the index (label) of the highest vote.
// If there are no votes, it returns 0.
func (me VoteSlice) ArgMax() int {
	if len(me) == 0 {
		return 0
	}
	var max struct {
		index int
		value float64
	}
	max.value = me[0]
	for i, x := range me[1:] {
		if x >= max.value {
			max.index = i + 1
			max.value = x
		}
	}
	return max.index
}

// Max returns the highest vote count in the slice.
func (me VoteSlice) Max() float64 {
	return slices.Max(me)
}

// Add adds the specified delta to the vote count for the given label.
func (me VoteSlice) Add(label int, delta float64) {
	me[label] += delta
}

// Get retrieves the vote count for the given label.
func (me VoteSlice) Get(label int) float64 {
	return me[label]
}

// VoteMap is a sparse vote counter that stores votes in a map.
// Good for large sets of class labels.
type VoteMap map[int]float64

// Clear resets all the votes in the map.
func (me VoteMap) Clear() {
	clear(me)
}

// ArgMax returns the label with the highest vote count.
// If there are no votes, it returns 0.
func (me VoteMap) ArgMax() int {
	if len(me) == 0 {
		return 0
	}
	var out struct {
		index int
		value float64
		any   bool
	}
	for i, x := range me {
		if !out.any || x >= out.value {
			out.index = i
			out.value = x
			out.any = true
		}
	}
	return out.index
}

// Max returns the highest vote count in the map.
func (me VoteMap) Max() float64 {
	var out struct {
		value float64
		any   bool
	}
	for _, x := range me {
		if !out.any {
			out.value = x
			out.any = true
			continue
		}
		out.value = max(x, out.value)
	}
	return out.value
}

// Add adds the specified delta to the vote count for the given label.
func (me VoteMap) Add(label int, delta float64) {
	me[label] += delta
}

// Get retrieves the vote count for the given label.
func (me VoteMap) Get(label int) float64 {
	return me[label]
}
