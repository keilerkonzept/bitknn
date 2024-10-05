package bitknn

import (
	"testing"
)

func TestMakeNeighborHeap(t *testing.T) {
	distances := []int{10, 20, 30}
	indices := []int{1, 2, 3}
	heap := makeNeighborHeap(distances, indices)

	// Check if lastDistance and lastIndex are pointing to the correct elements
	if *heap.lastDistance != 30 {
		t.Errorf("Expected lastDistance to be 30, got %d", *heap.lastDistance)
	}
	if *heap.lastIndex != 3 {
		t.Errorf("Expected lastIndex to be 3, got %d", *heap.lastIndex)
	}
}

func TestNeighborHeapSwap(t *testing.T) {
	heap := neighborHeap{
		distances: []int{10, 20, 30},
		indices:   []int{1, 2, 3},
	}

	heap.swap(0, 2)

	if heap.distances[0] != 30 || heap.distances[2] != 10 {
		t.Errorf("Swap failed on distances, got %v", heap.distances)
	}
	if heap.indices[0] != 3 || heap.indices[2] != 1 {
		t.Errorf("Swap failed on indices, got %v", heap.indices)
	}
}

func TestNeighborHeapLess(t *testing.T) {
	heap := neighborHeap{
		distances: []int{10, 20, 30},
		indices:   []int{1, 2, 3},
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
	indices := []int{1, 2, 3, 0}
	heap := makeNeighborHeap(distances, indices)
	heap.len = 3

	heap.pushpop(25, 4)

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
		if heap.indices[i] != expectedIndices[i] {
			t.Errorf("Expected index at %d to be %d, got %d", i, expectedIndices[i], heap.indices[i])
		}
	}
}

func TestNeighborHeapPush(t *testing.T) {
	heap := makeNeighborHeap(
		make([]int, 4),
		make([]int, 4),
	)

	heap.push(10, 3)
	heap.push(15, 5)
	heap.push(25, 6)
	heap.pushpop(9, 3)
	heap.pushpop(7, 2)
	heap.pushpop(8, 1)
	heap.pushpop(6, 0)

	if heap.distances[0] != 8 {
		t.Errorf("Expected root distance to be 25, got %d", heap.distances[0])
	}
	if heap.indices[0] != 1 {
		t.Errorf("Expected root index to be 6, got %d", heap.indices[0])
	}
}
