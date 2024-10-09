package pack_test

import (
	"bytes"
	"testing"

	"github.com/keilerkonzept/bitknn/pack"
	"pgregory.net/rapid"
)

func TestPackBytes(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		bytesInput := rapid.SliceOfN(rapid.Byte(), 0, 1024).Draw(t, "bytesInput")

		// Property 1: Length of packed []uint64 should be (len(bytes) + 7) / 8
		packed := pack.Bytes(bytesInput)
		expectedLength := (len(bytesInput) + 7) / 8
		if len(packed) != expectedLength {
			t.Fatalf("Expected packed length: %d, got: %d", expectedLength, len(packed))
		}

		// Property 2: Roundtrip
		unpacked := pack.BytesInv(packed, len(bytesInput))
		if !bytes.Equal(bytesInput, unpacked) {
			t.Fatalf("Original bytes: %v, Unpacked bytes: %v", bytesInput, unpacked)
		}
	})
}
