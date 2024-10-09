package pack_test

import (
	"testing"

	"github.com/keilerkonzept/bitknn/pack"
	"pgregory.net/rapid"
)

func TestPackString(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		data := rapid.String().Draw(t, "data")

		// Property 1: Length of packed []uint64 should be (len(data) + 7) / 8
		packed := pack.String(data)
		expectedLength := (len(data) + 7) / 8
		if len(packed) != expectedLength {
			t.Fatalf("Expected packed length: %d, got: %d", expectedLength, len(packed))
		}

		// Property 2: Roundtrip
		unpacked := pack.StringInv(packed, len(data))
		if data != unpacked {
			t.Fatalf("Original string: %v, Unpacked string: %v", data, unpacked)
		}
	})
}
