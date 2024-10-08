package slice

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
