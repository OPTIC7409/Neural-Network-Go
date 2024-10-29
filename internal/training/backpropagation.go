package training

import (
	"neural-network/internal/network"
)

type Trainer struct {
	Network      *network.Network
	LossFunc     LossFunction
	LossPrime    func(predicted, target []float64) []float64
	LearningRate float64
}

func NewTrainer(net *network.Network, lossFunc LossFunction, lossPrime func(predicted, target []float64) []float64, learningRate float64) *Trainer {
	return &Trainer{
		Network:      net,
		LossFunc:     lossFunc,
		LossPrime:    lossPrime,
		LearningRate: learningRate,
	}
}

func (t *Trainer) Train(input, target []float64) float64 {
	// Forward pass
	activations := [][]float64{input}
	for _, layer := range t.Network.Layers {
		output := layer.Forward(activations[len(activations)-1])
		activations = append(activations, output)
	}
	loss := t.LossFunc(activations[len(activations)-1], target)

	deltas := t.LossPrime(activations[len(activations)-1], target)
	for i := len(t.Network.Layers) - 1; i >= 0; i-- {
		layer := t.Network.Layers[i]
		nextDeltas := make([]float64, len(layer.Neurons[0].Weights))

		for j, neuron := range layer.Neurons {
			// Compute gradient
			gradient := deltas[j] * layer.ActivationPrime(neuron.Output)

			for k, input := range activations[i] {
				neuron.Weights[k] -= t.LearningRate * gradient * input
				nextDeltas[k] += gradient * neuron.Weights[k]
			}
			neuron.Bias -= t.LearningRate * gradient
		}

		deltas = nextDeltas
	}

	return loss
}

func (t *Trainer) TrainBatch(inputs, targets [][]float64, epochs int) []float64 {
	if len(inputs) != len(targets) {
		panic("Number of inputs must match number of targets")
	}

	losses := make([]float64, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		batchLoss := 0.0
		for i := range inputs {
			batchLoss += t.Train(inputs[i], targets[i])
		}
		losses[epoch] = batchLoss / float64(len(inputs))
	}

	return losses
}
