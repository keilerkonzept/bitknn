package slice

// OrAlloc ensures that a slice has the specified length.
// If the input slice is shorter, it's extended if possible, and reallocated otherwise.
// If it's longer, it's truncated.
func OrAlloc[T any](s []T, n int) []T {
	if len(s) == n {
		return s
	}
	if len(s) > n {
		return s[:n]
	}
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}
