# bitknn
[![Coverage](https://img.shields.io/badge/Coverage-97.8%25-brightgreen)](https://github.com/keilerkonzept/bitknn/actions/workflows/gocover.yaml)

[![Go Reference](https://pkg.go.dev/badge/github.com/keilerkonzept/bitknn.svg)](https://pkg.go.dev/github.com/keilerkonzept/bitknn)
[![Go Report Card](https://goreportcard.com/badge/github.com/keilerkonzept/bitknn)](https://goreportcard.com/report/github.com/keilerkonzept/bitknn)


```go
import "github.com/keilerkonzept/bitknn"
```

`bitknn` is a fast [k-nearest neighbors (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) library for `uint64`s, using (bitwise) Hamming distance.

If you need to classify **binary feature vectors that fit into `uint64`s**, this library might be useful. It is fast mainly because we can use cheap bitwise ops (XOR + POPCNT) to calculate distances between `uint64` values. For smaller datasets, the performance of the [neighbor heap](heap.go) is also relevant, and so this part has been tuned here also.

You can optionally weigh class votes by distance, or specify different vote values per data point.

The sub-package [`lsh`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh) implements several [Locality-Sensitive Hashing (LSH)](https://en.m.wikipedia.org/wiki/Locality-sensitive_hashing) schemes for uint64 feature vectors.

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

    model := bitknn.Fit(data, labels, bitknn.WithLinearDistanceWeighting())

    // one vote counter per class
    votes := make([]float64, 2)

    k := 2
    model.Predict1(k, 0b101011, votes)

    fmt.Println("Votes:", votes)
}
```

## Options

- `WithLinearDistanceWeighting()`: Apply linear distance weighting (`1 / (1 + dist)`).
- `WithQuadraticDistanceWeighting()`: Apply quadratic distance weighting (`1 / (1 + dist^2)`).
- `WithDistanceWeightingFunc(f func(dist int) float64)`: Use a custom distance weighting function.
- `WithValues(values []float64)`: Assign vote values for each data point.


## Benchmarks

```
goos: darwin
goarch: arm64
pkg: github.com/keilerkonzept/bitknn
cpu: Apple M1 Pro
```

| Op         | N       | k   | Distance weighting | Vote values | sec / op     | B/op | allocs/op |
|------------|---------|-----|--------------------|-------------|--------------|------|-----------|
| `Predict1` | 100     | 3   |                    |             | 138.7n ± 22% | 0    | 0         |
| `Predict1` | 100     | 3   |                    | ☑️           | 127.8n ± 11% | 0    | 0         |
| `Predict1` | 100     | 3   | linear             |             | 137.0n ± 11% | 0    | 0         |
| `Predict1` | 100     | 3   | linear             | ☑️           | 136.7n ± 10% | 0    | 0         |
| `Predict1` | 100     | 3   | quadratic          |             | 137.2n ±  7% | 0    | 0         |
| `Predict1` | 100     | 3   | quadratic          | ☑️           | 130.4n ±  4% | 0    | 0         |
| `Predict1` | 100     | 3   | custom             |             | 140.6n ±  7% | 0    | 0         |
| `Predict1` | 100     | 3   | custom             | ☑️           | 134.9n ± 13% | 0    | 0         |
| `Predict1` | 100     | 10  |                    |             | 307.4n ± 11% | 0    | 0         |
| `Predict1` | 100     | 10  |                    | ☑️           | 297.8n ± 15% | 0    | 0         |
| `Predict1` | 100     | 10  | linear             |             | 288.2n ± 18% | 0    | 0         |
| `Predict1` | 100     | 10  | linear             | ☑️           | 302.9n ± 14% | 0    | 0         |
| `Predict1` | 100     | 10  | quadratic          |             | 283.7n ± 15% | 0    | 0         |
| `Predict1` | 100     | 10  | quadratic          | ☑️           | 290.0n ± 13% | 0    | 0         |
| `Predict1` | 100     | 10  | custom             |             | 313.1n ± 17% | 0    | 0         |
| `Predict1` | 100     | 10  | custom             | ☑️           | 316.2n ± 11% | 0    | 0         |
| `Predict1` | 100     | 100 |                    | ☑️           | 545.4n ±  4% | 0    | 0         |
| `Predict1` | 100     | 100 | linear             |             | 542.4n ±  4% | 0    | 0         |
| `Predict1` | 100     | 100 | linear             | ☑️           | 577.5n ±  4% | 0    | 0         |
| `Predict1` | 100     | 100 | quadratic          |             | 553.1n ±  3% | 0    | 0         |
| `Predict1` | 100     | 100 | quadratic          | ☑️           | 582.4n ±  6% | 0    | 0         |
| `Predict1` | 100     | 100 | custom             |             | 683.8n ±  4% | 0    | 0         |
| `Predict1` | 100     | 100 | custom             | ☑️           | 748.5n ±  2% | 0    | 0         |
| `Predict1` | 1000    | 3   |                    |             | 669.5n ±  6% | 0    | 0         |
| `Predict1` | 1000    | 10  |                    |             | 930.3n ±  7% | 0    | 0         |
| `Predict1` | 1000    | 100 |                    |             | 3.762µ ±  5% | 0    | 0         |
| `Predict1` | 1000000 | 3   |                    |             | 532.1µ ±  1% | 0    | 0         |
| `Predict1` | 1000000 | 10  |                    |             | 534.5µ ±  1% | 0    | 0         |
| `Predict1` | 1000000 | 100 |                    |             | 551.7µ ±  1% | 0    | 0         |

## License

MIT License
