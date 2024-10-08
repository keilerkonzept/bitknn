package slice_test

import (
	"reflect"
	"testing"

	"github.com/keilerkonzept/bitknn/internal/slice"
)

func TestCountUniqueInSorted(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		expected int
	}{
		{"Empty slice", []int{}, 0},
		{"Single element", []int{1}, 1},
		{"All unique", []int{1, 2, 3, 4, 5}, 5},
		{"Some duplicates", []int{1, 1, 2, 3, 3, 4, 5, 5}, 5},
		{"All duplicates", []int{1, 1, 1, 1, 1}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := slice.CountUniqueInSorted(tt.input)
			if result != tt.expected {
				t.Errorf("CountUniqueInSorted(%v) = %d, want %d", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGroupSorted(t *testing.T) {
	tests := []struct {
		name           string
		input          []int
		keys           []string
		expectedGroups map[string]slice.IndexRange
		expectedKeys   []string
	}{
		{
			name:           "Empty slices",
			input:          []int{},
			keys:           []string{},
			expectedGroups: map[string]slice.IndexRange{},
			expectedKeys:   []string{},
		},
		{
			name:           "Single group",
			input:          []int{1, 2, 3},
			keys:           []string{"a", "a", "a"},
			expectedGroups: map[string]slice.IndexRange{"a": {Offset: 0, Length: 3}},
			expectedKeys:   []string{"a"},
		},
		{
			name:  "Multiple groups",
			input: []int{1, 2, 3, 4, 5, 6},
			keys:  []string{"a", "a", "b", "b", "c", "c"},
			expectedGroups: map[string]slice.IndexRange{
				"a": {Offset: 0, Length: 2},
				"b": {Offset: 2, Length: 2},
				"c": {Offset: 4, Length: 2},
			},
			expectedKeys: []string{"a", "b", "c"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			groups, keys := slice.GroupSorted(tt.input, tt.keys)
			if !reflect.DeepEqual(groups, tt.expectedGroups) {
				t.Errorf("GroupSorted() groups = %v, want %v", groups, tt.expectedGroups)
			}
			if !reflect.DeepEqual(keys, tt.expectedKeys) {
				t.Errorf("GroupSorted() keys = %v, want %v", keys, tt.expectedKeys)
			}
		})
	}
}

func TestOrAlloc(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		n        int
		expected []int
	}{
		{"Empty slice, n=0", []int{}, 0, []int{}},
		{"Empty slice, n>0", []int{}, 3, []int{0, 0, 0}},
		{"Slice shorter than n, reuse", []int{1, 2, 3, 4}[:2], 4, []int{1, 2, 3, 4}},
		{"Slice shorter than n, realloc", []int{1, 2}, 4, []int{0, 0, 0, 0}},
		{"Slice longer than n", []int{1, 2, 3, 4, 5}, 3, []int{1, 2, 3}},
		{"Slice equal to n", []int{1, 2, 3}, 3, []int{1, 2, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := slice.OrAlloc(tt.input, tt.n)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("OrAlloc(%v, %d) = %v, want %v", tt.input, tt.n, result, tt.expected)
			}
		})
	}
}

func TestReorderInPlace(t *testing.T) {
	tests := []struct {
		name     string
		input    []int
		indices  []int
		expected []int
	}{
		{"Empty slice", []int{}, []int{}, []int{}},
		{"No reordering", []int{1, 2, 3}, []int{0, 1, 2}, []int{1, 2, 3}},
		{"Simple reordering", []int{1, 2, 3}, []int{2, 0, 1}, []int{3, 1, 2}},
		{"Complex reordering", []int{1, 2, 3, 4, 5}, []int{4, 3, 2, 1, 0}, []int{5, 4, 3, 2, 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]int, len(tt.input))
			copy(input, tt.input)
			slice.ReorderInPlace(func(i, j int) {
				input[i], input[j] = input[j], input[i]
			}, tt.indices)
			if !reflect.DeepEqual(input, tt.expected) {
				t.Errorf("ReorderInPlace() result = %v, want %v", input, tt.expected)
			}
		})
	}
}
