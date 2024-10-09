package lsh_test

import (
	"reflect"
	"testing"

	"github.com/keilerkonzept/bitknn/lsh"
	"pgregory.net/rapid"
)

func Test_Nearest_64bit_Equal_To_Narrow(t *testing.T) {
	id := func(a uint64) uint64 { return a }
	rapid.Check(t, func(t *rapid.T) {
		k := rapid.IntRange(3, 2001).Draw(t, "k")
		data := rapid.SliceOfNDistinct(rapid.Uint64(), 3, 1000, id).Draw(t, "data")
		labels := rapid.SliceOfN(rapid.IntRange(0, 3), len(data), len(data)).Draw(t, "labels")
		dataWide := make([][]uint64, len(data))
		for i := range data {
			dataWide[i] = []uint64{data[i]}
		}

		type hash struct {
			narrow lsh.Hash
			wide   lsh.HashWide
		}
		hashes := []hash{
			{
				narrow: lsh.ConstantHash{},
				wide:   lsh.ConstantHash{},
			},
			{
				narrow: lsh.BitSample(0xF0F0F0F0F0F0F0F0),
				wide: &lsh.HashWide1{
					Single: lsh.BitSample(0xF0F0F0F0F0F0F0F0),
				},
			},
		}
		for _, h := range hashes {
			m := lsh.Fit(data, labels, h.narrow)
			m.PreallocateHeap(k)
			mw := lsh.FitWide(dataWide, labels, h.wide)
			mw.PreallocateHeap(k)

			x := rapid.Uint64().Draw(t, "query")
			xw := []uint64{x}
			xh := m.Hash.Hash1(x)
			xwh := mw.Hash.Hash1Wide(xw)

			lsh.Nearest(data, m.BucketIDs, m.Buckets, k, xh, x, m.HeapBucketDistances, m.HeapBucketIDs, m.HeapDistances, m.HeapIndices)
			lsh.NearestWide(dataWide, mw.BucketIDs, mw.Buckets, k, xwh, xw, mw.HeapBucketDistances, mw.HeapBucketIDs, mw.Narrow.HeapDistances, mw.Narrow.HeapIndices)

			if !reflect.DeepEqual(m.HeapIndices, mw.Narrow.HeapIndices) {
				t.Fatal(m.HeapIndices, mw.Narrow.HeapIndices)
			}
			if !reflect.DeepEqual(m.HeapDistances, mw.Narrow.HeapDistances) {
				t.Fatal(m.HeapDistances, mw.Narrow.HeapDistances)
			}
		}
	})
}
