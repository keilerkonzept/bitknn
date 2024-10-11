package lsh_test

import (
	"math/bits"
	"reflect"
	"testing"

	"github.com/keilerkonzept/bitknn/internal/testrandom"
	"github.com/keilerkonzept/bitknn/lsh"
	"pgregory.net/rapid"
)

func TestBlurWide(t *testing.T) {
	t.Run("BlurWide_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			dims := rapid.IntRange(1, 4).Draw(t, "dims")
			data := rapid.SliceOf(rapid.SliceOfN(rapid.Uint64(), dims, dims)).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomBlurWide(dims, 4, 3)
			h.HashWide(data, out)
			for i, d := range data {
				if out[i] != h.Hash1Wide(d) {
					t.Fatal()
				}
			}
		})
	})
	t.Run("RandomBlurWide", func(t *testing.T) {
		blur := lsh.RandomBlurWide(2, 4, 3)

		if len(blur.Masks) != 3 {
			t.Errorf("RandomBlurWide returned BlurWide with %d masks, expected 3", len(blur.Masks))
		}

		for i, mask := range blur.Masks {
			totalBits := 0
			for _, b := range mask {
				totalBits += bits.OnesCount64(b.M)
			}
			if totalBits != 4 {
				t.Errorf("Mask %d has %d bits set, expected 4", i, totalBits)
			}
		}

		if blur.Threshold != 3 {
			t.Errorf("RandomBlurWide returned BlurWide with threshold %d, expected 3", blur.Threshold)
		}
	})

	t.Run("BlurWide_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b11, 0b10}
		y := []uint64{0b11, 0b00}
		z := []uint64{0b00, 0b01}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBlurWide(2, 3, 100)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("BlurWideR_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b11, 0b10}
		y := []uint64{0b11, 0b00}
		z := []uint64{0b00, 0b01}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBlurWideR(2, 3, 100, testrandom.Source)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestMinHashWide(t *testing.T) {
	t.Run("MinHashWide_Hash1", func(t *testing.T) {
		h := lsh.RandomMinHashWide(3)
		input := []uint64{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF}
		output := h.Hash1Wide(input)
		if output >= 1<<6 { // 6 bits
			t.Errorf("MinHashWide.Hash1Wide(%x) = %d; want value < %d", input, output, 1<<18)
		}
	})

	t.Run("MinHashWide_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			dims := rapid.IntRange(1, 4).Draw(t, "dims")
			data := rapid.SliceOf(rapid.SliceOfN(rapid.Uint64(), dims, dims)).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomMinHashWide(dims)
			h.HashWide(data, out)
			for i, d := range data {
				if out[i] != h.Hash1Wide(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("MinHashWide_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b1110, 0b1110, 0b1100}
		y := []uint64{0b1100, 0b1100, 0b1110}
		z := []uint64{0b0001, 0b0001, 0b1100}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomMinHashWide(3)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("MinHashWideR_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b1110, 0b1110, 0b1100}
		y := []uint64{0b1100, 0b1100, 0b1110}
		z := []uint64{0b0001, 0b0001, 0b1100}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomMinHashWideR(3, testrandom.Source)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}

func TestBitSampleWide(t *testing.T) {
	t.Run("RandomBitSampleWide", func(t *testing.T) {
		for _, numBits := range []int{1, 32, 63} {
			h := lsh.RandomBitSampleWide(3, numBits)
			count := 0
			for _, ms := range h {
				for _, m := range ms {
					count += bits.OnesCount64(m)
				}
			}
			if count != numBits {
				t.Errorf("RandomBitSampleWide(%d) set %d bits; want %d", numBits, count, numBits)
			}
		}
	})

	t.Run("BitSampleWide_Hash1", func(t *testing.T) {
		h := lsh.BitSampleWide{{1 << 15, 1 << 0}}

		testCases := []struct {
			input []uint64
			want  uint64
		}{
			{[]uint64{0xFFFFFFFF}, 3},
			{[]uint64{0x0F0F0F0F}, 1},
			{[]uint64{0xAAAAAAAA}, 2},
		}

		for _, tc := range testCases {
			got := h.Hash1Wide(tc.input)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("BitSampleWide.Hash1(%x) = %x; want %x", tc.input, got, tc.want)
			}
		}
	})

	t.Run("BitSampleWide_Hash1_Equiv_Hash", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			dims := rapid.IntRange(1, 4).Draw(t, "dims")
			data := rapid.SliceOf(rapid.SliceOfN(rapid.Uint64(), dims, dims)).Draw(t, "data")
			out := make([]uint64, len(data))
			h := lsh.RandomBitSampleWide(dims, 10)
			h.HashWide(data, out)
			for i, d := range data {
				if out[i] != h.Hash1Wide(d) {
					t.Fatal()
				}
			}
		})
	})

	t.Run("BitSampleWide_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b1110, 0b1110, 0b1100}
		y := []uint64{0b1100, 0b1100, 0b1110}
		z := []uint64{0b0001, 0b0001, 0b1100}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBitSampleWide(3, 48)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})

	t.Run("BitSampleWideR_Hamming_LS_Property", func(t *testing.T) {
		x := []uint64{0b1110, 0b1110, 0b1100}
		y := []uint64{0b1100, 0b1100, 0b1110}
		z := []uint64{0b0001, 0b0001, 0b1100}

		xyEqual := 0
		xzEqual := 0
		trials := 1000

		for range trials {
			h := lsh.RandomBitSampleWideR(3, 48, testrandom.Source)
			if h.Hash1Wide(x) == h.Hash1Wide(y) {
				xyEqual++
			}
			if h.Hash1Wide(x) == h.Hash1Wide(z) {
				xzEqual++
			}
		}

		if xyEqual <= xzEqual {
			t.Errorf("Expected Hash1(x) to equal Hash1(y) more often than Hash1(x) to equal Hash1(z), got %d and %d", xyEqual, xzEqual)
		}
	})
}
