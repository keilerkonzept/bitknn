package testrandom_test

import (
	"testing"

	"github.com/keilerkonzept/bitknn/internal/testrandom"
)

func TestQuery(t *testing.T) {
	_ = testrandom.Query()
}
func TestData(t *testing.T) {
	data := testrandom.Data(123)
	if len(data) != 123 {
		t.Fatal()
	}
}
func TestLabels(t *testing.T) {
	data := testrandom.Labels(123)
	if len(data) != 123 {
		t.Fatal()
	}
}
func TestValues(t *testing.T) {
	data := testrandom.Values(123)
	if len(data) != 123 {
		t.Fatal()
	}
}
