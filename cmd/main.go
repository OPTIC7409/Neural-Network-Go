package main

import (
	"fmt"
	"math/rand"
	"time"

	"neural-network/internal/activation"
	"neural-network/internal/network"
	"neural-network/internal/training"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	layerSizes := []int{2, 3, 1}

	activationFuncs := []network.ActivationFunc{activation.Sigmoid, activation.Sigmoid}
	activationPrimes := []network.ActivationFunc{activation.SigmoidPrime, activation.SigmoidPrime}

	nn := network.NewNetwork(layerSizes, activationFuncs, activationPrimes)

	trainer := training.NewTrainer(nn, training.MeanSquaredError, training.MeanSquaredErrorPrime, 0.1)

	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	epochs := 10000
	losses := trainer.TrainBatch(inputs, targets, epochs)
	fmt.Printf("Final loss: %f\n", losses[len(losses)-1])

	for i, input := range inputs {
		output := nn.Predict(input)
		fmt.Printf("Input: %v, Predicted: %v, Target: %v\n", input, output, targets[i])
	}
}
