package bitknn

type Option func(*Model)

// Assign vote values for each data point.
func WithValues(v []float64) Option {
	return func(o *Model) { o.Values = v }
}

// Apply linear distance weighting (`1 / (1 + dist)`).
func WithLinearDistanceWeighting() Option {
	return func(o *Model) { o.DistanceWeighting = DistanceWeightingLinear }
}

// Apply quadratic distance weighting (`1 / (1 + dist^2)`).
func WithQuadraticDistanceWeighting() Option {
	return func(o *Model) { o.DistanceWeighting = DistanceWeightingQuadratic }
}

// Use a custom distance weighting function.
func WithDistanceWeightingFunc(f func(dist int) float64) Option {
	return func(o *Model) {
		o.DistanceWeighting = DistanceWeightingCustom
		o.DistanceWeightingFunc = f
	}
}
