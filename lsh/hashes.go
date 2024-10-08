package lsh

import (
	"math/bits"
	"math/rand/v2"
)

// Hash is a hash function for the lsh package.
type Hash interface {
	Hash1(uint64) uint64
	Hash(in []uint64, out []uint64)
}

// HashFunc is a function type that implements the Hash interface.
type HashFunc func(uint64) uint64

// Hash1 applies the function to a single uint64 value.
func (me HashFunc) Hash1(x uint64) uint64 { return me(x) }

// Hash applies the function to a slice of uint64 values.
func (me HashFunc) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		out[i] = me(d)
	}
}

// NoHash is the identity function. Used as a dummy [Hash] for testing.
type NoHash struct{}

// Hash1 returns the given value.
func (me NoHash) Hash1(x uint64) uint64 { return x }

// Hash copies the input slice to the output slice.
func (me NoHash) Hash(data []uint64, out []uint64) {
	copy(out, data)
}

// MinHashes is a concatenation of [MinHash]es
type MinHashes []MinHash

// RandomMinHashes creates n random MinHash functions.
func RandomMinHashesR(n int, rand *rand.Rand) MinHashes {
	out := make([]MinHash, n)
	for i := range out {
		out[i] = RandomMinHashR(rand)
	}
	return out
}

// RandomMinHashes creates [n] random MinHashes.
func RandomMinHashes(n int) MinHashes {
	out := make([]MinHash, n)
	for i := range out {
		out[i] = RandomMinHash()
	}
	return out
}

// Hash1 applies each MinHash to the given uint64 value and concatenates the MinHash bits.
func (me MinHashes) Hash1(x uint64) uint64 {
	var out uint64
	for _, h := range me {
		out <<= 6 // log2(64)
		out |= h.Hash1(x)
	}
	return out
}

// Has1 applies the MinHashes to each uint64 value in the slice.
func (me MinHashes) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		var m uint64
		for _, h := range me {
			m <<= 6 // log2(64)
			m |= h.Hash1(d)
		}
		out[i] = m
	}
}

// MinHash is a MinHash function for Hamming space.
type MinHash []uint64

// RandomMinHashR returns a random [MinHash].
func RandomMinHashR(rand *rand.Rand) MinHash {
	ones := rand.Perm(64)
	out := make([]uint64, 64)
	for i := range out {
		out[i] = 1 << ones[i]
	}
	return out
}

// RandomMinHash returns a random [MinHash].
func RandomMinHash() MinHash {
	ones := rand.Perm(64)
	out := make([]uint64, 64)
	for i := range out {
		out[i] = 1 << ones[i]
	}
	return out
}

// Hash1 hashes a single uint64 value.
func (me MinHash) Hash1(x uint64) uint64 {
	for j, m := range me {
		if (x & m) != 0 {
			return uint64(j)
		}
	}
	return 0 // never reached
}

// Hash hashes a slice of uint64 values.
func (me MinHash) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		for j, m := range me {
			if (d & m) != 0 {
				out[i] = uint64(j)
			}
		}
	}
}

var boxBlur3LUT = [8]uint64{
	0, // 0b000,
	0, // 0b001,
	0, // 0b010,
	1, // 0b011,
	0, // 0b100,
	1, // 0b101,
	1, // 0b110,
	1, // 0b111,
}

func boxBlur3(x uint64) uint64 {
	var b uint64
	b = boxBlur3LUT[x&0b11]
	for i := range 61 {
		b |= boxBlur3LUT[x&0b111] << (i + 1)
		x >>= 1
	}
	return b
}

// BoxBlur3 hashes values by applying a box blur with radius 3 (each bit in the output is the average of the 3 neighboring bits in the input)
type BoxBlur3 struct{}

// Hash1 hashes a single uint64 value.
func (me BoxBlur3) Hash1(x uint64) uint64 {
	return boxBlur3(x)
}

// Hash hashes a slice of uint64 values.
func (me BoxBlur3) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		out[i] = boxBlur3(d)
	}
}

// Blur hashes values based on thresholding the number of bits in common with the given bitmasks.
// For bitmasks of consecutive set bits, this is in effect a "blur" of the bit vector.
type Blur struct {
	Masks     []uint64 // Bitmasks
	Threshold int      // Minimum number of common bits required to set the output bit
}

// Hash1 hashes a single uint64 value.
func (me Blur) Hash1(x uint64) uint64 {
	var bx uint64
	for _, b := range me.Masks {
		bx <<= 1
		if bits.OnesCount64(x&b) >= me.Threshold {
			bx |= 1
		}
	}
	return bx
}

// Hash hashes a slice of uint64 values.
func (me Blur) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		var bx uint64
		for _, b := range me.Masks {
			bx <<= 1
			if bits.OnesCount64(d&b) >= me.Threshold {
				bx |= 1
			}
		}
		out[i] = bx
	}
}

// RandomBitSample generates a Blur of [n] bitmasks with the given number [numBits] of set bits.
func RandomBlurR(numBits int, n int, rand *rand.Rand) Blur {
	bits := make([]uint64, n)
	threshold := numBits/2 + 1
	for i := range n {
		b := uint64(RandomBitSampleR(numBits, rand))
		bits[i] = b
	}
	return Blur{
		Masks:     bits,
		Threshold: threshold,
	}
}

// RandomBitSample generates a Blur of [n] bitmasks with the given number [numBits] of set bits.
func RandomBlur(numBits int, n int) Blur {
	bits := make([]uint64, n)
	threshold := numBits/2 + 1
	for i := range n {
		b := uint64(RandomBitSample(numBits))
		bits[i] = b
	}
	return Blur{
		Masks:     bits,
		Threshold: threshold,
	}
}

// BitSample is a random sampling of bits in a uint64 value.
// Only the bits set in the BitSample are kept.
type BitSample uint64

// Hash1 hashes a single uint64 value.
func (me BitSample) Hash1(x uint64) uint64 {
	return x & uint64(me)
}

// Hash hashes a slice of uint64 values.
func (me BitSample) Hash(data []uint64, out []uint64) {
	for i, d := range data {
		out[i] = d & uint64(me)
	}
}

// RandomBitSample generates a BitSample with a specified number of bits set to 1.
func RandomBitSample(numBitsSet int) BitSample {
	ones := rand.Perm(64)
	var out uint64
	for i := 0; i < numBitsSet; i++ {
		out |= uint64(1) << ones[i]
	}
	return BitSample(out)
}

// RandomBitSample generates a BitSample with a specified number of bits set to 1.
func RandomBitSampleR(numBitsSet int, rand *rand.Rand) BitSample {
	ones := rand.Perm(64)
	var out uint64
	for i := 0; i < numBitsSet; i++ {
		out |= uint64(1) << ones[i]
	}
	return BitSample(out)
}
