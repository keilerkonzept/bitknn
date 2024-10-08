# bitknn
[![Coverage](https://img.shields.io/badge/Coverage-100.0%25-brightgreen)](https://github.com/keilerkonzept/bitknn/actions/workflows/gocover.yaml)

[![Go Reference](https://pkg.go.dev/badge/github.com/keilerkonzept/bitknn.svg)](https://pkg.go.dev/github.com/keilerkonzept/bitknn)
[![Go Report Card](https://goreportcard.com/badge/github.com/keilerkonzept/bitknn)](https://goreportcard.com/report/github.com/keilerkonzept/bitknn)


```go
import "github.com/keilerkonzept/bitknn"
```

`bitknn` is a fast [k-nearest neighbors (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) library for `uint64`s, using (bitwise) Hamming distance.

If you need to classify **binary feature vectors that fit into `uint64`s**, this library might be useful. It is fast mainly because we can use cheap bitwise ops (XOR + POPCNT) to calculate distances between `uint64` values. For smaller datasets, the performance of the [neighbor heap](heap.go) is also relevant, and so this part has been tuned here also.

If your vectors are **longer than 64 bits**, you can [pack](#packing-wide-data) them into `[]uint64` and classify them using the ["wide" model variants](#packing-wide-data).

You can optionally weigh class votes by distance, or specify different vote values per data point.

The sub-package [`lsh`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh) implements several [Locality-Sensitive Hashing (LSH)](https://en.m.wikipedia.org/wiki/Locality-sensitive_hashing) schemes for `uint64` feature vectors.

**Contents**
- [Usage](#usage)
  - [Basic usage](#basic-usage)
  - [LSH](#lsh)
  - [Packing wide data](#packing-wide-data)
- [Options](#options)
- [Benchmarks](#benchmarks)
- [License](#license)

## Usage

### Basic usage

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
    model.Predict1(k, 0b101011, bitknn.VoteSlice(votes))

    fmt.Println("Votes:", bitknn.VoteSlice(votes))

    // you can also use a map for the votes.
    // this is good if you have a very large number of different labels:
    votesMap := make(map[int]float64)
    model.Predict1(k, 0b101011, bitknn.VoteMap(votesMap))
    fmt.Println("Votes for 0:", votesMap[0])
}
```

### LSH

Locality-Sensitive Hashing (LSH) is a type of approximate k-NN search. It's faster at the expense of accuracy.

LSH works by hashing data points such that points that are close in Hamming space tend to land in the same bucket, and computing k-nearest neighbors only on the buckets with the k nearest hashes. In particular, for *k*=1 only one bucket needs to be examined.

```go
package main

import (
    "fmt"
    "github.com/keilerkonzept/bitknn/lsh"
    "github.com/keilerkonzept/bitknn"
)

func main() {
    // feature vectors packed into uint64s
    data := []uint64{0b101010, 0b111000, 0b000111}
    // class labels
    labels := []int{0, 1, 1}

    // Define a hash function (e.g., MinHash)
    hash := lsh.RandomMinHash()

    // Fit an LSH model
    model := lsh.Fit(data, labels, hash, bitknn.WithLinearDistanceWeighting())

    // one vote counter per class
    votes := make([]float64, 2)

    k := 2
    model.Predict1(k, 0b101011, bitknn.VoteSlice(votes))

    fmt.Println("Votes:", bitknn.VoteSlice(votes))

    // you can also use a map for the votes
    votesMap := make(map[int]float64)
    model.Predict1(k, 0b101011, bitknn.VoteMap(votesMap))
    fmt.Println("Votes for 0:", votesMap[0])
}
```

The model accepts anything that implements the [`lsh.Hash` interface](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#Hash) as a hash function. Several functions are pre-defined:

- [MinHash](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#MinHash): An implementation of the [MinHash scheme](https://en.m.wikipedia.org/wiki/MinHash) for bit vectors.

  Constructors: [RandomMinHash](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomMinHash), [RandomMinHashR](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomMinHashR).
- [MinHashes](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#MinHash): Concatenation of several *MinHash*es.

  Constructors: [RandomMinHashes](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomMinHashes), [RandomMinHashesR](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomMinHashesR).
- [Blur](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#Blur): A threshold-based variation on bit sampling.

  Constructors: [RandomBlur](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomBlur), [RandomBlurR](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomBlurR), [BoxBlur](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#BoxBlur), .
- [BitSample](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#BitSample): A random sampling of bits from the feature vector.

    Constructors: [RandomBitSample](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomBitSample), [RandomBitSampleR](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#RandomBitSampleR).

For datasets of vectors longer than 64 bits, the `lsh` package also provides a [`lsh.FitWide`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#FitWide) function, and "wide" versions of the hash functions ([MinHashWide](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#MinHashWide), [BlurWide](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#BlurWide), [BitSampleWide](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#BitSampleWide))

The  [`lsh.Fit`/`lsh.FitWide`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/lsh#Fit) functions accept the same [Options](#options) as the others.

### Packing wide data

If your vectors are longer than 64 bits, you can still use `bitknn` if you [pack](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack) them into `[]uint64`. The [`pack` package](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack) defines helper functions to pack `string`s and `[]byte`s into `[]uint64`s.

> It's faster to use a `[][]uint64` allocated using a flat backing slice, laid out in one contiguous memory block. If you already have a non-contiguous `[][]uint64`, you can use [`pack.ReallocateFlat`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack#ReallocateFlat) to re-allocate the dataset using a flat 1d backing slice.

The exact k-NN model in `bitknn` and the approximate-NN model in `lsh` each have a `Wide` variant that accepts slice-valued data points:

```go
package main

import (
    "fmt"

    "github.com/keilerkonzept/bitknn"
    "github.com/keilerkonzept/bitknn/pack"
)

func main() {
    // feature vectors packed into uint64s
    data := [][]uint64{
    	pack.String("foo"),
    	pack.String("bar"),
    	pack.String("baz"),
    }
    // class labels
    labels := []int{0, 1, 1}

    model := bitknn.FitWide(data, labels, bitknn.WithLinearDistanceWeighting())
    // also using LSH:
    // model := lsh.FitWide(data, labels, lsh.RandomMinHash(), bitknn.WithLinearDistanceWeighting())

    // one vote counter per class
    votes := make([]float64, 2)

    k := 2
    query := pack.String("fob")
    model.Predict1(k, query, bitknn.VoteSlice(votes))

    fmt.Println("Votes:", bitknn.VoteSlice(votes))
}
```

The wide model fitting function [`bitknn.FitWide`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#FitWide) accepts the same [Options](#options) as the "narrow" one.

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
