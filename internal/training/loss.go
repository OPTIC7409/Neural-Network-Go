package training

import "math"

type LossFunction func(predicted, target []float64) float64

func MeanSquaredError(predicted, target []float64) float64 {
	if len(predicted) != len(target) {
		panic("Predicted and target slices must have the same length")
	}

	sum := 0.0
	for i := range predicted {
		diff := predicted[i] - target[i]
		sum += diff * diff
	}
	return sum / float64(len(predicted))
}

func MeanSquaredErrorPrime(predicted, target []float64) []float64 {
	if len(predicted) != len(target) {
		panic("Predicted and target slices must have the same length")
	}

	result := make([]float64, len(predicted))
	for i := range predicted {
		result[i] = 2 * (predicted[i] - target[i]) / float64(len(predicted))
	}
	return result
}

func CrossEntropy(predicted, target []float64) float64 {
	if len(predicted) != len(target) {
		panic("Predicted and target slices must have the same length")
	}

	sum := 0.0
	for i := range predicted {
		sum -= target[i] * math.Log(predicted[i])
	}
	return sum / float64(len(predicted))
}

func CrossEntropyPrime(predicted, target []float64) []float64 {
	if len(predicted) != len(target) {
		panic("Predicted and target slices must have the same length")
	}

	result := make([]float64, len(predicted))
	for i := range predicted {
		result[i] = -target[i] / predicted[i] / float64(len(predicted))
	}
	return result
}
