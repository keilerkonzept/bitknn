package slice

// ReorderInPlace reorders elements in a slice based on the provided indices.
// The swap function is used to perform the actual swapping of elements.
func ReorderInPlace(swap func(i, j int), indices []int) {
	for i, targetIdx := range indices {
		for targetIdx < i {
			targetIdx = indices[targetIdx]
		}
		swap(i, targetIdx)
	}
}
