package lsh_test

import (
	"testing"

	"github.com/keilerkonzept/bitknn/internal/slice"
	"github.com/keilerkonzept/bitknn/lsh"
)

func TestNearest(t *testing.T) {
	t.Run("Nearest_<k_points", func(t *testing.T) {
		data := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
		bucketIDs := []uint64{0, 1}
		buckets := map[uint64]slice.IndexRange{
			0: {Offset: 0, Length: 4},
			1: {Offset: 4, Length: 4},
		}
		k := 10
		distances := make([]int, k+1)
		bucketDistances := make([]int, k+1)
		heapBucketIDs := make([]uint64, k+1)
		indices := make([]int, k+1)

		x := uint64(5)
		xh := uint64(1)
		k, n := lsh.Nearest(data, bucketIDs, buckets, k, x, xh, bucketDistances, heapBucketIDs, distances, indices)

		if 8 != k {
			t.Fatal(k)
		}
		if 8 != n {
			t.Fatal(n)
		}
	})
	t.Run("Nearest_<k_buckets", func(t *testing.T) {
		data := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
		bucketIDs := []uint64{0, 1}
		buckets := map[uint64]slice.IndexRange{
			0: {Offset: 0, Length: 4},
			1: {Offset: 4, Length: 4},
		}
		k := 3
		distances := make([]int, k+1)
		bucketDistances := make([]int, k+1)
		heapBucketIDs := make([]uint64, k+1)
		indices := make([]int, k+1)

		x := uint64(5)
		xh := uint64(1)
		k, n := lsh.Nearest(data, bucketIDs, buckets, k, x, xh, bucketDistances, heapBucketIDs, distances, indices)

		if 3 != k {
			t.Fatal(k)
		}
		if 8 != n {
			t.Fatal(n)
		}
	})
	t.Run("Nearest_>=k_buckets", func(t *testing.T) {
		data := []uint64{1, 2, 3, 4, 5, 6, 7, 8}
		bucketIDs := []uint64{0, 1, 2, 3}
		buckets := map[uint64]slice.IndexRange{
			0: {Offset: 0, Length: 2},
			1: {Offset: 2, Length: 2},
			2: {Offset: 4, Length: 2},
			3: {Offset: 6, Length: 2},
		}
		k := 3
		distances := make([]int, k+1)
		bucketDistances := make([]int, k+1)
		heapBucketIDs := make([]uint64, k+1)
		indices := make([]int, k+1)

		{
			x := uint64(5)
			xh := uint64(1)
			k, n := lsh.Nearest(data, bucketIDs, buckets, k, x, xh, bucketDistances, heapBucketIDs, distances, indices)

			if 3 != k {
				t.Fatal(k)
			}
			if 6 != n {
				t.Fatal(n)
			}
		}
		{
			x := uint64(4)
			xh := uint64(2)
			k, n := lsh.Nearest(data, bucketIDs, buckets, k, x, xh, bucketDistances, heapBucketIDs, distances, indices)

			if 3 != k {
				t.Fatal(k)
			}
			if 6 != n {
				t.Fatal(n)
			}
		}
	})
}
