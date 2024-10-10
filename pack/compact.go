package pack

// ReallocateFlat re-allocates the given 2d slice with a flat backing slice.
func ReallocateFlat[T any](d [][]T) {
	n := 0
	for _, d := range d {
		n += len(d)
	}
	flat := make([]T, n)
	j := 0
	for i, row := range d {
		copy(flat[j:], row)
		d[i] = flat[j : j+len(row)]
		j += len(row)
	}
}
