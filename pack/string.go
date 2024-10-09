package pack

import "unsafe"

// String packs a string into a uint64 slice.
// If the length of the string in bytes is not a multiple of 8, it will pad the remaining bytes with zeroes.
func String(data string) []uint64 {
	b := unsafe.Slice(unsafe.StringData(data), len(data))
	return Bytes(b)
}

// StringInv unpacks a []uint64 slice as packed by [String],
func StringInv(data []uint64, originalLengthBytes int) string {
	b := BytesInv(data, originalLengthBytes)
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}