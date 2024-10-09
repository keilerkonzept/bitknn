package lsh_test

import (
	"math/bits"
	"reflect"
	"testing"

	"github.com/keilerkonzept/bitknn/internal/testrandom"
	"github.com/keilerkonzept/bitknn/lsh"
	"pgregory.net/rapid"
)

func TestHashCompose(t *testing.T) {
	h1 := lsh.RandomBlurR(3, 20, testrandom.Source)
	h2 := lsh.RandomMinHash()
	h := lsh.HashCompose{h1, h2}
	rapid.Check(t, func(t *rapid.T) {
		q := rapid.Uint64().Draw(t, "q")
		qs := rapid.SliceOf(rapid.Uint64()).Draw(t, "qs")
		if h.Hash1(q) != h2.Hash1(h1.Hash1(q)) {
			t.Fatal()
		}
		out12 := make([]uint64, len(qs))
		out := make([]uint64, len(qs))
		h.Hash(qs, out)
		h1.Hash(qs, out12)
		h2.Hash(out12, out12)
		if !reflect.DeepEqual(out, out12) {
			t.Fatal()
		}
	})
}

func TestMinHash(t *testing.T) {
	t.Run("RandomMinHash", func(t *testing.T) {
		h := lsh.RandomMinHash()
		if len(h) != 64 {
			t.Errorf("RandomMinHash() returned slice of length %d; want 64", len(h))
		}

		// Check that all bit positions are represented
		var allBits uint64
		for _, m := range h {
			allBits |= m
		}
		if allBits != ^uint64(0) {
			t.Errorf("RandomMinHash() doesn't cover all bit positions")
		}
	})

	t.Run("MinHash_Hash1", func(t *testing.T) {
		h := lsh.RandomMinHash()
		testCases := []struct {
			input uint64
		}{
			{0b1000},
			{0b1100},
			{0b1111},
		}

		for _, tc := range testCases {
			got := h.Hash1(tc.input)
			if got >= 64 {
				t.Errorf("MinHash.Hash() returned %d for input %b; want value < 64", got, tc.input)
			}
		}
	})

	t.Run("MinHash_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			data := rapid.SliceOf(rapid.Uint64()).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomMinHash()
			h.Hash(data, out)
			for i, d := range data {
				if out[i] != h.Hash1(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("MinHash_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomMinHash()
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestBlur(t *testing.T) {
	t.Run("Blur_Hash1", func(t *testing.T) {
		h := &lsh.Blur{
			Masks:     []uint64{0xF0F0F0F0, 0x0F0F0F0F},
			Threshold: 4,
		}

		testCases := []struct {
			input uint64
			want  uint64
		}{
			{0xFFFFFFFF, 3},
			{0xF0F0F0F0, 2},
			{0x0F0F0F0F, 1},
			{0x00000000, 0},
		}

		for _, tc := range testCases {
			got := h.Hash1(tc.input)
			if got != tc.want {
				t.Errorf("Blur.Hash1(%x) = %d; want %d", tc.input, got, tc.want)
			}
		}
	})

	t.Run("Blur_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			bits := rapid.IntRange(0, 64).Draw(t, "bits")
			masks := rapid.IntRange(0, 10).Draw(t, "masks")
			data := rapid.SliceOf(rapid.Uint64()).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomBlur(bits, masks)
			h.Hash(data, out)
			for i, d := range data {
				if out[i] != h.Hash1(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("BoxBlur", func(t *testing.T) {
		trials := 1000
		yCloser := 0
		zCloser := 0
		for range trials {
			n := testrandom.Source.IntN(32)
			dist := func(x, y uint64) int {
				return bits.OnesCount64(x ^ y)
			}
			flipNBits := uint64(lsh.RandomBitSampleR(n, testrandom.Source))
			flip2NBits := uint64(lsh.RandomBitSampleR(2*n, testrandom.Source))
			x := testrandom.Query()
			y := x ^ flipNBits
			z := x ^ flip2NBits
			h := lsh.BoxBlur(3, 3)
			dy := dist(h.Hash1(x), h.Hash1(y))
			dz := dist(h.Hash1(x), h.Hash1(z))
			if dy < dz {
				yCloser++
			}
			if dy > dz {
				zCloser++
			}
		}

		if zCloser > yCloser {
			t.Errorf("Expected Hash1(x) to be closer to Hash1(y) more often than Hash1(x) to be closer to Hash1(z), got %d and %d", yCloser, zCloser)
		}
	})

	t.Run("Blur_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBlur(3, 10)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("BlurR_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 10_000

		for range trials {
			h := lsh.RandomBlurR(3, 10, testrandom.Source)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestBitSample(t *testing.T) {
	t.Run("RandomBitSample", func(t *testing.T) {
		for _, numBits := range []int{1, 32, 63} {
			h := lsh.RandomBitSample(numBits)
			count := bits.OnesCount64(uint64(h))
			if count != numBits {
				t.Errorf("RandomBitSample(%d) set %d bits; want %d", numBits, count, numBits)
			}
		}
	})

	t.Run("BitSample_Hash1", func(t *testing.T) {
		h := lsh.BitSample(0xF0F0F0F0)

		testCases := []struct {
			input uint64
			want  uint64
		}{
			{0xFFFFFFFF, 0xF0F0F0F0},
			{0x0F0F0F0F, 0x00000000},
			{0xAAAAAAAA, 0xA0A0A0A0},
		}

		for _, tc := range testCases {
			got := h.Hash1(tc.input)
			if got != tc.want {
				t.Errorf("BitSample.Hash1(%x) = %x; want %x", tc.input, got, tc.want)
			}
		}
	})

	t.Run("BitSample_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			bits := rapid.IntRange(0, 64).Draw(t, "bits")
			data := rapid.SliceOf(rapid.Uint64()).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomBitSample(bits)
			h.Hash(data, out)
			for i, d := range data {
				if out[i] != h.Hash1(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("BitSample_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBitSample(48)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("BitSampleR_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBitSampleR(48, testrandom.Source)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestMinHashes(t *testing.T) {
	t.Run("RandomMinHashes", func(t *testing.T) {
		h := lsh.RandomMinHashes(3)
		if len(h) != 3 {
			t.Errorf("RandomMinHashes(3) returned slice of length %d; want 3", len(h))
		}
		for _, h := range h {
			if len(h) != 64 {
				t.Errorf("RandomMinHashes(3) contains MinHash of length %d; want 64", len(h))
			}
		}
	})

	t.Run("MinHashes_Hash1", func(t *testing.T) {
		h := lsh.RandomMinHashes(3)
		input := uint64(0xFFFFFFFF)
		output := h.Hash1(input)
		if output >= 1<<18 { // 3 * 6 bits
			t.Errorf("MinHashes.Hash1(%x) = %d; want value < %d", input, output, 1<<18)
		}
	})

	t.Run("MinHashes_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			masks := rapid.IntRange(0, 10).Draw(t, "masks")
			data := rapid.SliceOf(rapid.Uint64()).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomMinHashes(masks)
			h.Hash(data, out)
			for i, d := range data {
				if out[i] != h.Hash1(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("MinHashes_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomMinHashes(3)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("MinHashesR_LS_Property", func(t *testing.T) {
		x := uint64(0b1110)
		y := uint64(0b1100)
		z := uint64(0b0001)

		xyEqual := 0
		xzEqual := 0
		trials := 10_000

		for range trials {
			h := lsh.RandomMinHashesR(3, testrandom.Source)
			if h.Hash1(x) == h.Hash1(y) {
				xyEqual++
			}
			if h.Hash1(x) == h.Hash1(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestHashFunc(t *testing.T) {
	trials := 1000

	for range trials {
		data := testrandom.Data(16)
		h := lsh.RandomBitSample(10)
		hf := lsh.HashFunc(h.Hash1)
		outh := make([]uint64, len(data))
		outhf := make([]uint64, len(data))
		x := testrandom.Query()
		if h.Hash1(x) != hf.Hash1(x) {
			t.Fatal()
		}
		h.Hash(data, outh)
		hf.Hash(data, outhf)
		if !reflect.DeepEqual(outh, outhf) {
			t.Fatal()
		}

	}
}

func TestDummyHashes(t *testing.T) {
	t.Run("NoHash", func(t *testing.T) {
		var h lsh.NoHash
		query := uint64(0x12345)
		data := []uint64{0x12345, 0x54321}
		out := make([]uint64, len(data))
		if h.Hash1(query) != query {
			t.Fatal()
		}
		h.Hash(data, out)
		if !reflect.DeepEqual(data, out) {
			t.Fatal()
		}
	})
	t.Run("ConstantHash", func(t *testing.T) {
		var h lsh.ConstantHash
		q := uint64(0x12345)
		data := []uint64{0x12345, 0x54321}
		out := make([]uint64, len(data))
		if h.Hash1(q) != 0 {
			t.Fatal()
		}
		h.Hash(data, out)
		for i := range out {
			if out[i] != 0 {
				t.Fatal()
			}
		}
	})
}
