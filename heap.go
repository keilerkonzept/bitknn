package bitknn

import "unsafe"

// neighborHeap is a max-heap that stores distances and their corresponding indices.
// The heap is used to keep track of nearest neighbors.
type neighborHeap struct {
	distances    []int
	lastDistance *int
	indices      []int
	lastIndex    *int
	len          int
}

const unsafeSizeofInt = unsafe.Sizeof(int(0))

func makeNeighborHeap(distances, indices []int) neighborHeap {
	return neighborHeap{
		distances:    distances,
		lastDistance: (*int)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(distances)), unsafeSizeofInt*uintptr(len(distances)-1))),
		indices:      indices,
		lastIndex:    (*int)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(indices)), unsafeSizeofInt*uintptr(len(indices)-1))),
	}
}

func (me *neighborHeap) swap(i, j int) {
	me.distances[i], me.distances[j] = me.distances[j], me.distances[i]
	me.indices[i], me.indices[j] = me.indices[j], me.indices[i]
}

func (me *neighborHeap) less(i, j int) bool {
	return me.distances[i] > me.distances[j]
}

func (me *neighborHeap) pushpop(value int, index int) {
	n := me.len
	*me.lastDistance = value
	*me.lastIndex = index
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

func (me *neighborHeap) push(value int, index int) {
	n := me.len
	me.distances[n] = value
	me.indices[n] = index
	me.len = n + 1
	me.up(n)
}

func (me *neighborHeap) up(i int) {
	for {
		p := (i - 1) / 2              // Parent index
		if p == i || !me.less(i, p) { // If parent is larger or i is root, stop
			break
		}
		me.swap(p, i) // Swap child with parent
		i = p         // Continue moving up
	}
}
