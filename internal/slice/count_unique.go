package slice

// CountUniqueInSorted counts the number of unique elements in a sorted slice.
// It assumes the input slice is already sorted.
func CountUniqueInSorted[T comparable](s []T) int {
	out := 0
	var previous T
	for i, b := range s {
		if i == 0 {
			previous = b
			out = 1
			continue
		}
		if b != previous {
			out++
			previous = b
		}
	}
	return out
}

type IndexRange struct {
	Offset int
	Length int
}

func GroupSorted[E any, K comparable](s []E, sKeys []K) (map[K]IndexRange, []K) {
	numGroups := CountUniqueInSorted(sKeys)
	groups := make(map[K]IndexRange, numGroups)
	keys := make([]K, numGroups)
	{
		var previous K
		var previousIdx int
		j := 0

		for i, b := range sKeys {
			if i == 0 {
				keys[0] = b
				previous = b
				previousIdx = 0
				j = 1
				continue
			}
			if b != previous {
				groups[previous] = IndexRange{
					Offset: previousIdx,
					Length: i - previousIdx,
				}
				keys[j] = b
				j++
				previous = b
				previousIdx = i
			}
		}
		if len(s) > 0 {
			groups[previous] = IndexRange{
				Offset: previousIdx,
				Length: len(s) - previousIdx,
			}
		}
	}
	return groups, keys
}
