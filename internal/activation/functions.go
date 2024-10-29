package activation

import (
	"math"
)

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func TanhPrime(x float64) float64 {
	return 1 - math.Pow(Tanh(x), 2)
}

func Softmax(x []float64) []float64 {
	expSum := 0.0
	result := make([]float64, len(x))

	for i, v := range x {
		exp := math.Exp(v)
		result[i] = exp
		expSum += exp
	}

	for i := range result {
		result[i] /= expSum
	}

	return result
}

func SoftmaxPrime(x []float64) []float64 {
	s := Softmax(x)
	result := make([]float64, len(x))

	for i := range result {
		result[i] = s[i] * (1 - s[i])
	}

	return result
}
