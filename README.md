# bitknn
[![Coverage](https://img.shields.io/badge/Coverage-100.0%25-brightgreen)](https://github.com/keilerkonzept/bitknn/actions/workflows/gocover.yaml)

[![Go Reference](https://pkg.go.dev/badge/github.com/keilerkonzept/bitknn.svg)](https://pkg.go.dev/github.com/keilerkonzept/bitknn)
[![Go Report Card](https://goreportcard.com/badge/github.com/keilerkonzept/bitknn)](https://goreportcard.com/report/github.com/keilerkonzept/bitknn)


```go
import "github.com/keilerkonzept/bitknn"
```

`bitknn` is a fast [k-nearest neighbors (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) library for `uint64`s, using Hamming distance to measure similarity.

If you need to classify **binary feature vectors that fit into `uint64`s**, this library might be useful. It is fast mainly because we can use cheap bitwise ops (XOR + POPCNT) to calculate distances between `uint64` values. For smaller datasets, the performance of the [neighbor heap](heap.go) is also relevant, and so this part has been tuned here also.

You can optionally weigh class votes by distance, or specify different vote values per data point.


**Contents**
- [Usage](#usage)
- [Options](#options)
- [Benchmarks](#benchmarks)
- [License](#license)

## Usage

```go
package main

import (
    "fmt"
    "github.com/keilerkonzept/bitknn"
)

func main() {
    // feature vectors packed into uint64s
    data := []uint64{0b101010, 0b111000, 0b000111}
    // class labels
    labels := []int{0, 1, 1}

    model := bitknn.Fit(data, labels, 2, bitknn.WithLinearDecay())

    // one vote counter per class
    votes := make([]float64, 2)
    model.Predict1(0b101011, votes)

    fmt.Println("Votes:", votes)
}
```

## Options

- `WithLinearDecay()`: Apply linear distance weighting (`1 / (1 + dist)`).
- `WithQuadraticDecay()`: Apply quadratic distance weighting (`1 / (1 + dist^2)`).
- `WithDistanceWeightFunc(f func(dist int) float64)`: Use a custom distance weighting function.
- `WithValues(values []float64)`: Assign specific vote values for each data point.

## Benchmarks

```
goos: darwin
goarch: arm64
pkg: github.com/keilerkonzept/bitknn
cpu: Apple M1 Pro
```

| op         | N       | k   | iters   | ns/op        | B/op | allocs/op |
|------------|---------|-----|---------|--------------|------|-----------|
| `Predict1` | 100     | 3   | 8308794 | 121.4 ns/op  | 0    | 0         |
| `Predict1` | 100     | 10  | 4707778 | 269.7 ns/op  | 0    | 0         |
| `Predict1` | 100     | 100 | 2255380 | 549.2 ns/op  | 0    | 0         |
| `Predict1` | 1000    | 3   | 1693364 | 659.3 ns/op  | 0    | 0         |
| `Predict1` | 1000    | 10  | 1220426 | 1005 ns/op   | 0    | 0         |
| `Predict1` | 1000    | 100 | 345151  | 3560 ns/op   | 0    | 0         |
| `Predict1` | 1000000 | 3   | 2076    | 566647 ns/op | 0    | 0         |
| `Predict1` | 1000000 | 10  | 2112    | 568787 ns/op | 0    | 0         |
| `Predict1` | 1000000 | 100 | 2066    | 587827 ns/op | 0    | 0         |

## License

MIT License
