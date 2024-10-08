package slice

func ReorderInPlace(swap func(i, j int), indices []int) {
	for i, targetIdx := range indices {
		for targetIdx < i {
			targetIdx = indices[targetIdx]
		}
		swap(i, targetIdx)
	}
}
