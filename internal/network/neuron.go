package network

import (
	"math/rand"
)

type Neuron struct {
	Weights []float64
	Bias    float64
	Output  float64
}

func NewNeuron(inputSize int) *Neuron {
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1
	}
	return &Neuron{
		Weights: weights,
		Bias:    rand.Float64()*2 - 1,
	}
}

func (n *Neuron) Activate(inputs []float64) float64 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	n.Output = sum
	return n.Output
}
