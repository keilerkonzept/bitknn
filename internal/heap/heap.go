package heap

import "unsafe"

// Max is a max-heap used to keep track of nearest neighbors.
type Max[T int | uint64] struct {
	distances    []int
	lastDistance *int
	values       []T
	lastValue    *T
	len          int
}

const unsafeSizeofInt = unsafe.Sizeof(int(0))

func MakeMax[T int | uint64](distances []int, value []T) Max[T] {
	return Max[T]{
		distances:    distances,
		lastDistance: (*int)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(distances)), unsafeSizeofInt*uintptr(len(distances)-1))),
		values:       value,
		lastValue:    (*T)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(value)), unsafe.Sizeof(T(0))*uintptr(len(value)-1))),
	}
}

func (me *Max[T]) swap(i, j int) {
	me.distances[i], me.distances[j] = me.distances[j], me.distances[i]
	me.values[i], me.values[j] = me.values[j], me.values[i]
}

func (me *Max[T]) less(i, j int) bool {
	return me.distances[i] > me.distances[j]
}

func (me *Max[T]) PushPop(dist int, value T) {
	n := me.len
	*me.lastDistance = dist
	*me.lastValue = value
	me.up(n)
	me.swap(0, n)

	// me.down(0, n)
	i := 0
	for {
		l := 2*i + 1         // Left child
		if l >= n || l < 0 { // If no left child, break
			break
		}
		j := l
		if r := l + 1; r < n && me.less(r, l) { // If right child exists and is smaller, select right child
			j = r
		}
		if !me.less(j, i) { // If parent is smaller than selected child, break
			break
		}
		me.swap(i, j) // Swap parent with child
		i = j         // Continue pushing down
	}
}

func (me *Max[T]) Push(dist int, value T) {
	n := me.len
	me.distances[n] = dist
	me.values[n] = value
	me.len = n + 1
	me.up(n)
}

func (me *Max[T]) up(i int) {
	for {
		p := (i - 1) / 2              // Parent index
		if p == i || !me.less(i, p) { // If parent is larger or i is root, stop
			break
		}
		me.swap(p, i) // Swap child with parent
		i = p         // Continue moving up
	}
}
