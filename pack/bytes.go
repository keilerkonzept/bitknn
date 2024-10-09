// Package pack provides helpers to pack bytes and strings into []uint64 slices.
package pack

// Bytes packs a byte slice into a uint64 slice.
// If the length of the byte slice is not a multiple of 8, it will pad the remaining bytes with zeroes.
func Bytes(data []byte) []uint64 {
	n := len(data)
	dims := (n + 7) / 8 // round up division

	out := make([]uint64, dims)

	i := 0
	for ; i+8 <= n; i += 8 {
		out[i/8] = uint64(data[i]) |
			uint64(data[i+1])<<8 |
			uint64(data[i+2])<<16 |
			uint64(data[i+3])<<24 |
			uint64(data[i+4])<<32 |
			uint64(data[i+5])<<40 |
			uint64(data[i+6])<<48 |
			uint64(data[i+7])<<56
	}

	if i < n {
		var packed uint64
		remaining := n - i
		switch remaining {
		case 7:
			packed |= uint64(data[i+6]) << 48
			fallthrough
		case 6:
			packed |= uint64(data[i+5]) << 40
			fallthrough
		case 5:
			packed |= uint64(data[i+4]) << 32
			fallthrough
		case 4:
			packed |= uint64(data[i+3]) << 24
			fallthrough
		case 3:
			packed |= uint64(data[i+2]) << 16
			fallthrough
		case 2:
			packed |= uint64(data[i+1]) << 8
			fallthrough
		case 1:
			packed |= uint64(data[i])
		}
		out[i/8] = packed
	}

	return out
}

// BytesInv unpacks a []uint64 slice as packed by [Bytes],
func BytesInv(data []uint64, originalLength int) []byte {
	byteSlice := make([]byte, originalLength)

	for i := 0; i < len(data); i++ {
		for j := 0; j < 8 && i*8+j < originalLength; j++ {
			byteSlice[i*8+j] = byte((data[i] >> (8 * j)) & 0xFF)
		}
	}

	return byteSlice
}
