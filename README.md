# bitknn
[![Coverage](https://img.shields.io/badge/Coverage-100.0%25-brightgreen)](https://github.com/keilerkonzept/bitknn/actions/workflows/gocover.yaml)

[![Go Reference](https://pkg.go.dev/badge/github.com/keilerkonzept/bitknn.svg)](https://pkg.go.dev/github.com/keilerkonzept/bitknn)
[![Go Report Card](https://goreportcard.com/badge/github.com/keilerkonzept/bitknn)](https://goreportcard.com/report/github.com/keilerkonzept/bitknn)


```go
import "github.com/keilerkonzept/bitknn"
```

`bitknn` is a fast [k-nearest neighbors (k-NN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) library for `uint64`s, using (bitwise) Hamming distance.

If you need to classify **binary feature vectors that fit into `uint64`s**, this library might be useful. It is fast mainly because we can use cheap bitwise ops (XOR + POPCNT) to calculate distances between `uint64` values. For smaller datasets, the performance of the [neighbor heap](internal/heap/heap.go) is also relevant, and so this part has been tuned here also.

If your vectors are **longer than 64 bits**, you can [pack](#packing-wide-data) them into `[]uint64` and classify them using the ["wide" model variants](#packing-wide-data). On ARM64 with NEON vector instruction support, `bitknn` can be [a bit faster still](#arm64-neon-support) on wide data.

You can optionally weigh class votes by distance, or specify different vote values per data point.

**Contents**
- [Usage](#usage)
  - [Basic usage](#basic-usage)
  - [Packing wide data](#packing-wide-data)
  - [ARM64 NEON Support](#arm64-neon-support)
- [Options](#options)
- [Benchmarks](#benchmarks)
- [License](#license)

## Usage

There are just three methods you'll typically need:

- **Fit** *(data, labels, [\[options\]](#options))*: create a model from a dataset

  Variants: [`bitknn.Fit`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#Fit), [`bitknn.FitWide`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#FitWide)

- **Find** *(k, point)*: Given a point, return the *k* nearest neighbor's indices and distances.

  Variants: [`bitknn.Model.Find`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#Model.Find), [`bitknn.WideModel.Find`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.Find), [`bitknn.WideModel.FindV`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.FindV) (vectorized on ARM64 with NEON instructions)

- **Predict** *(k, point, votes)*: Predict the label for a given point based on its nearest neighbors, write the label votes into the provided vote counter.

  Variants: [`bitknn.Model.Predict`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#Model.Predict), [`bitknn.WideModel.Predict`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.Predict), [`bitknn.WideModel.PredictV`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.PredictV) (vectorized on ARM64 with NEON instructions).

Each of the above methods is available on either model type:

- [`bitknn.Model`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#Model) (64 bits)
- [`bitknn.WideModel`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel) (*N* * 64 bits)

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
    model.Predict(k, 0b101011, bitknn.VoteSlice(votes))
    // or, just return the nearest neighbor's distances and indices:
    // distances,indices := model.Find(k, 0b101011)

    fmt.Println("Votes:", bitknn.votes)

    // you can also use a map for the votes.
    // this is good if you have a very large number of different labels:
    votesMap := make(map[int]float64)
    model.Predict(k, 0b101011, bitknn.VoteMap(votesMap))
    fmt.Println("Votes for 0:", votesMap[0])
}
```

### Packing wide data

If your vectors are longer than 64 bits, you can still use `bitknn` if you [pack](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack) them into `[]uint64`. The [`pack` package](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack) defines helper functions to pack `string`s and `[]byte`s into `[]uint64`s.

> It's faster to use a `[][]uint64` allocated using a flat backing slice, laid out in one contiguous memory block. If you already have a non-contiguous `[][]uint64`, you can use [`pack.ReallocateFlat`](https://pkg.go.dev/github.com/keilerkonzept/bitknn/pack#ReallocateFlat) to re-allocate the dataset using a flat 1d backing slice.

The wide model fitting function is [`bitknn.FitWide`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#FitWide) and accepts the same [Options](#options) as the "narrow" one:


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

    // one vote counter per class
    votes := make([]float64, 2)

    k := 2
    query := pack.String("fob")
    model.Predict(k, query, bitknn.VoteSlice(votes))

    fmt.Println("Votes:", votes)
}
```

### ARM64 NEON Support

For ARM64 CPUs with NEON instructions, `bitknn` has a [vectorized distance function for `[]uint64s`s](internal/neon/distance_arm64.s) that is about twice as fast as what the compiler generates.

When run on such a CPU, the ***V** methods [`WideModel.FindV`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.FindV) and [`WideModel.PredictV`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WideModel.predictV) are  noticeably faster than  the regular `Find`/`Predict`:

| Bits  | N       | k   | `Find` s/op  | `FindV` s/op | diff                   |
|-------|---------|-----|--------------|--------------|------------------------|
| 128   | 1000    | 3   | 2.374µ ± 0%  | 1.792µ ± 0%  | -24.54% (p=0.000 n=10) |
| 128   | 1000    | 10  | 2.901µ ± 1%  | 2.028µ ± 1%  | -30.08% (p=0.000 n=10) |
| 128   | 1000    | 100 | 5.472µ ± 3%  | 4.359µ ± 1%  | -20.34% (p=0.000 n=10) |
| 128   | 1000000 | 3   | 2.273m ± 3%  | 1.380m ± 2%  | -39.27% (p=0.000 n=10) |
| 128   | 1000000 | 10  | 2.261m ± 1%  | 1.406m ± 1%  | -37.84% (p=0.000 n=10) |
| 128   | 1000000 | 100 | 2.289m ± 0%  | 1.425m ± 2%  | -37.76% (p=0.000 n=10) |
| 640   | 1000    | 3   | 6.201µ ± 1%  | 3.716µ ± 0%  | -40.07% (p=0.000 n=10) |
| 640   | 1000    | 10  | 6.728µ ± 1%  | 3.973µ ± 1%  | -40.96% (p=0.000 n=10) |
| 640   | 1000    | 100 | 10.855µ ± 2% | 6.917µ ± 1%  | -36.28% (p=0.000 n=10) |
| 640   | 1000000 | 3   | 5.832m ± 2%  | 3.337m ± 1%  | -42.78% (p=0.000 n=10) |
| 640   | 1000000 | 10  | 5.830m ± 5%  | 3.339m ± 1%  | -42.73% (p=0.000 n=10) |
| 640   | 1000000 | 100 | 5.872m ± 1%  | 3.361m ± 1%  | -42.77% (p=0.000 n=10) |
| 8192  | 1000000 | 10  | 72.66m ± 1%  | 30.96m ± 3%  | -57.39% (p=0.000 n=10) |


## Options

- [`bitknn.WithLinearDistanceWeighting()`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WithLinearDistanceWeighting): Apply linear distance weighting (`1 / (1 + dist)`).
- [`bitknn.WithQuadraticDistanceWeighting()`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WithQuadraticDistanceWeighting): Apply quadratic distance weighting (`1 / (1 + dist^2)`).
- [`bitknn.WithDistanceWeightingFunc(f func(dist int) float64)`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WithDistanceWeightingFunc): Use a custom distance weighting function.
- [`bitknn.WithValues(values []float64)`](https://pkg.go.dev/github.com/keilerkonzept/bitknn#WithValues): Assign vote values for each data point.


## Benchmarks

```
goos: darwin
goarch: arm64
pkg: github.com/keilerkonzept/bitknn
cpu: Apple M1 Pro
```

| Bits | N       | k   | Model     | Op        | s/op        | B/op | allocs/op |
|------|---------|-----|-----------|-----------|-------------|------|-----------|
| 64   | 100     | 3   | Model     | `Predict` | 99.06n ± 2% | 0    | 0         |
| 64   | 100     | 3   | WideModel | `Predict` | 191.6n ± 1% | 0    | 0         |
| 64   | 100     | 3   | Model     | `Find`    | 88.09n ± 0% | 0    | 0         |
| 64   | 100     | 3   | WideModel | `Find`    | 182.8n ± 1% | 0    | 0         |
| 64   | 100     | 10  | Model     | `Predict` | 225.1n ± 1% | 0    | 0         |
| 64   | 100     | 10  | WideModel | `Predict` | 372.0n ± 1% | 0    | 0         |
| 64   | 100     | 10  | Model     | `Find`    | 202.9n ± 1% | 0    | 0         |
| 64   | 100     | 10  | WideModel | `Find`    | 345.2n ± 0% | 0    | 0         |
| 64   | 1000    | 3   | Model     | `Predict` | 538.2n ± 1% | 0    | 0         |
| 64   | 1000    | 3   | WideModel | `Predict` | 1.469µ ± 1% | 0    | 0         |
| 64   | 1000    | 3   | Model     | `Find`    | 525.8n ± 1% | 0    | 0         |
| 64   | 1000    | 3   | WideModel | `Find`    | 1.465µ ± 1% | 0    | 0         |
| 64   | 1000    | 10  | Model     | `Predict` | 835.4n ± 1% | 0    | 0         |
| 64   | 1000    | 10  | WideModel | `Predict` | 1.880µ ± 1% | 0    | 0         |
| 64   | 1000    | 10  | Model     | `Find`    | 807.4n ± 0% | 0    | 0         |
| 64   | 1000    | 10  | WideModel | `Find`    | 1.867µ ± 2% | 0    | 0         |
| 64   | 1000    | 100 | Model     | `Predict` | 3.718µ ± 0% | 0    | 0         |
| 64   | 1000    | 100 | WideModel | `Predict` | 4.935µ ± 0% | 0    | 0         |
| 64   | 1000    | 100 | Model     | `Find`    | 3.494µ ± 0% | 0    | 0         |
| 64   | 1000    | 100 | WideModel | `Find`    | 4.701µ ± 0% | 0    | 0         |
| 64   | 1000000 | 3   | Model     | `Predict` | 458.8µ ± 0% | 0    | 0         |
| 64   | 1000000 | 3   | WideModel | `Predict` | 1.301m ± 1% | 0    | 0         |
| 64   | 1000000 | 3   | Model     | `Find`    | 457.9µ ± 1% | 0    | 0         |
| 64   | 1000000 | 3   | WideModel | `Find`    | 1.302m ± 1% | 0    | 0         |
| 64   | 1000000 | 10  | Model     | `Predict` | 456.9µ ± 0% | 0    | 0         |
| 64   | 1000000 | 10  | WideModel | `Predict` | 1.295m ± 2% | 0    | 0         |
| 64   | 1000000 | 10  | Model     | `Find`    | 457.6µ ± 1% | 0    | 0         |
| 64   | 1000000 | 10  | WideModel | `Find`    | 1.298m ± 1% | 0    | 0         |
| 64   | 1000000 | 100 | Model     | `Predict` | 474.5µ ± 1% | 0    | 0         |
| 64   | 1000000 | 100 | WideModel | `Predict` | 1.316m ± 1% | 0    | 0         |
| 64   | 1000000 | 100 | Model     | `Find`    | 466.9µ ± 0% | 0    | 0         |
| 64   | 1000000 | 100 | WideModel | `Find`    | 1.306m ± 0% | 0    | 0         |
| 128  | 100     | 3   | WideModel | `Predict` | 296.7n ± 0% | 0    | 0         |
| 128  | 100     | 3   | WideModel | `Find`    | 285.8n ± 0% | 0    | 0         |
| 128  | 100     | 10  | WideModel | `Predict` | 467.4n ± 1% | 0    | 0         |
| 128  | 100     | 10  | WideModel | `Find`    | 441.1n ± 1% | 0    | 0         |
| 640  | 100     | 3   | WideModel | `Predict` | 654.6n ± 1% | 0    | 0         |
| 640  | 100     | 3   | WideModel | `Find`    | 640.3n ± 1% | 0    | 0         |
| 640  | 100     | 10  | WideModel | `Predict` | 850.0n ± 1% | 0    | 0         |
| 640  | 100     | 10  | WideModel | `Find`    | 825.0n ± 0% | 0    | 0         |
| 128  | 1000    | 3   | WideModel | `Predict` | 2.384µ ± 0% | 0    | 0         |
| 128  | 1000    | 3   | WideModel | `Find`    | 2.374µ ± 0% | 0    | 0         |
| 128  | 1000    | 10  | WideModel | `Predict` | 2.900µ ± 0% | 0    | 0         |
| 128  | 1000    | 10  | WideModel | `Find`    | 2.901µ ± 1% | 0    | 0         |
| 128  | 1000    | 100 | WideModel | `Predict` | 5.630µ ± 1% | 0    | 0         |
| 128  | 1000    | 100 | WideModel | `Find`    | 5.472µ ± 3% | 0    | 0         |
| 128  | 1000000 | 3   | WideModel | `Predict` | 2.266m ± 0% | 0    | 0         |
| 128  | 1000000 | 3   | WideModel | `Find`    | 2.273m ± 3% | 0    | 0         |
| 128  | 1000000 | 10  | WideModel | `Predict` | 2.269m ± 0% | 0    | 0         |
| 128  | 1000000 | 10  | WideModel | `Find`    | 2.261m ± 1% | 0    | 0         |
| 128  | 1000000 | 100 | WideModel | `Predict` | 2.295m ± 1% | 0    | 0         |
| 128  | 1000000 | 100 | WideModel | `Find`    | 2.289m ± 0% | 0    | 0         |
| 640  | 1000    | 3   | WideModel | `Predict` | 6.214µ ± 2% | 0    | 0         |
| 640  | 1000    | 3   | WideModel | `Find`    | 6.201µ ± 1% | 0    | 0         |
| 640  | 1000    | 10  | WideModel | `Predict` | 6.777µ ± 1% | 0    | 0         |
| 640  | 1000    | 10  | WideModel | `Find`    | 6.728µ ± 1% | 0    | 0         |
| 640  | 1000    | 100 | WideModel | `Predict` | 11.16µ ± 2% | 0    | 0         |
| 640  | 1000    | 100 | WideModel | `Find`    | 10.85µ ± 2% | 0    | 0         |
| 640  | 1000000 | 3   | WideModel | `Predict` | 5.756m ± 4% | 0    | 0         |
| 640  | 1000000 | 3   | WideModel | `Find`    | 5.832m ± 2% | 0    | 0         |
| 640  | 1000000 | 10  | WideModel | `Predict` | 5.842m ± 1% | 0    | 0         |
| 640  | 1000000 | 10  | WideModel | `Find`    | 5.830m ± 5% | 0    | 0         |
| 640  | 1000000 | 100 | WideModel | `Predict` | 5.914m ± 6% | 0    | 0         |
| 640  | 1000000 | 100 | WideModel | `Find`    | 5.872m ± 1% | 0    | 0         |

## License

MIT License
