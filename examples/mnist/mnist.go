package main

import (
	"fmt"
	"math/rand"
	"time"

	"neural-network/internal/activation"
	"neural-network/internal/network"
	"neural-network/internal/training"
	"neural-network/internal/utils"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	dataset, err := utils.LoadCSV("mnist_train.csv", []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{0})
	if err != nil {
		fmt.Printf("Error loading dataset: %v\n", err)
		return
	}

	for i := range dataset.Inputs {
		for j := range dataset.Inputs[i] {
			dataset.Inputs[i][j] /= 255.0
		}
	}

	for i := range dataset.Targets {
		oneHot := make([]float64, 10)
		oneHot[int(dataset.Targets[i][0])] = 1.0
		dataset.Targets[i] = oneHot
	}

	trainSet, testSet := utils.SplitDataSet(dataset, 0.8)

	layerSizes := []int{784, 128, 64, 10}

	activationFuncs := []network.ActivationFunc{activation.ReLU, activation.ReLU}
	activationPrimes := []network.ActivationFunc{activation.ReLUPrime, activation.ReLUPrime}

	nn := network.NewNetworkWithSliceOutput(layerSizes, activationFuncs, activationPrimes, activation.Softmax, activation.SoftmaxPrime)

	trainer := training.NewTrainer(nn, training.CrossEntropy, training.CrossEntropyPrime, 0.01)

	epochs := 10
	batchSize := 32
	for epoch := 0; epoch < epochs; epoch++ {
		utils.ShuffleDataSet(trainSet)
		totalLoss := 0.0
		for i := 0; i < len(trainSet.Inputs); i += batchSize {
			end := i + batchSize
			if end > len(trainSet.Inputs) {
				end = len(trainSet.Inputs)
			}
			batchInputs := trainSet.Inputs[i:end]
			batchTargets := trainSet.Targets[i:end]
			loss := trainer.TrainBatch(batchInputs, batchTargets, 1)[0]
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(trainSet.Inputs)/batchSize)
		fmt.Printf("Epoch %d, Average Loss: %f\n", epoch+1, avgLoss)
	}

	correct := 0
	for i := range testSet.Inputs {
		output := nn.Predict(testSet.Inputs[i])
		predictedClass := argmax(output)
		actualClass := argmax(testSet.Targets[i])
		if predictedClass == actualClass {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testSet.Inputs)) * 100

	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy)
}

func argmax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}
