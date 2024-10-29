package utils

import (
	"math"
	"math/rand"
)

func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("Vectors must have the same length")
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func VectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("Vectors must have the same length")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func VectorSubtract(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("Vectors must have the same length")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func VectorMultiply(v []float64, scalar float64) []float64 {
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

func RandomNormal(mean, stdDev float64) float64 {
	return rand.NormFloat64()*stdDev + mean
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}
