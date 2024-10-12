package neon

import (
	"math/bits"
	"testing"

	"pgregory.net/rapid"
)

func TestDistancesWideGeneric(t *testing.T) {
	t.Run("DistancesWideGenericEquivBits", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			dims := rapid.IntRange(0, 10_000).Draw(t, "dims")
			data := rapid.SliceOfN(rapid.SliceOfN(rapid.Uint64(), dims, dims), 16, 10_000).Draw(t, "data")
			q := rapid.SliceOfN(rapid.Uint64(), dims, dims).Draw(t, "q")
			for batchSize := range []int{0, 1, 2, len(data), len(data) - 1, len(data) * 2} {
				out := make([]uint32, batchSize)
				distancesWideGeneric(q, data[:batchSize], out)
				for i, d := range out {
					expected := 0
					for j, q := range q {
						expected += bits.OnesCount64(q ^ data[i][j])
					}
					if int(d) != expected {
						t.Fatal(d, expected)
					}
				}
			}
		})
	})
}
