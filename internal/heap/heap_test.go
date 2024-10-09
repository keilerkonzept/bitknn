package heap

import (
	"testing"
)

func TestMakeNeighborHeap(t *testing.T) {
	distances := []int{10, 20, 30}
	values := []int{1, 2, 3}
	heap := MakeMax(distances, values)

	// Check if lastDistance and lastValue are pointing to the correct elements
	if *heap.lastDistance != 30 {
		t.Errorf("Expected lastDistance to be 30, got %d", *heap.lastDistance)
	}
	if *heap.lastValue != 3 {
		t.Errorf("Expected lastValue to be 3, got %d", *heap.lastValue)
	}
}

func TestNeighborHeapSwap(t *testing.T) {
	heap := Max[int]{
		distances: []int{10, 20, 30},
		values:    []int{1, 2, 3},
	}

	heap.swap(0, 2)

	if heap.distances[0] != 30 || heap.distances[2] != 10 {
		t.Errorf("Swap failed on distances, got %v", heap.distances)
	}
	if heap.values[0] != 3 || heap.values[2] != 1 {
		t.Errorf("Swap failed on values, got %v", heap.values)
	}
}

func TestNeighborHeapLess(t *testing.T) {
	heap := Max[int]{
		distances: []int{10, 20, 30},
		values:    []int{1, 2, 3},
	}

	if !heap.less(2, 0) {
		t.Errorf("Expected less(2, 0) to be true, got false")
	}

	if heap.less(0, 2) {
		t.Errorf("Expected less(0, 2) to be false, got true")
	}
}

func TestNeighborHeapPushPop(t *testing.T) {
	distances := []int{30, 20, 10, 0}
	values := []int{1, 2, 3, 0}
	heap := MakeMax(distances, values)
	heap.len = 3

	heap.PushPop(25, 4)

	if heap.Len() != 3 {
		t.Error("Expected length not to change")
	}

	// Check if heap is reordered correctly
	expectedDistances := []int{25, 20, 10,
		30,
	}
	expectedIndices := []int{4, 2, 3,
		1,
	}
	for i := range expectedDistances {
		if heap.distances[i] != expectedDistances[i] {
			t.Errorf("Expected distance at %d to be %d, got %d", i, expectedDistances[i], heap.distances[i])
		}
		if heap.values[i] != expectedIndices[i] {
			t.Errorf("Expected value at %d to be %d, got %d", i, expectedIndices[i], heap.values[i])
		}
	}
}

func TestNeighborHeapPush(t *testing.T) {
	heap := MakeMax(
		make([]int, 4),
		make([]int, 4),
	)

	heap.Push(10, 3)
	heap.Push(15, 5)
	heap.Push(25, 6)
	heap.PushPop(9, 3)
	heap.PushPop(7, 2)
	heap.PushPop(8, 1)
	heap.PushPop(6, 0)

	if heap.distances[0] != 8 {
		t.Errorf("Expected root distance to be 25, got %d", heap.distances[0])
	}
	if heap.values[0] != 1 {
		t.Errorf("Expected root value to be 6, got %d", heap.values[0])
	}
}
