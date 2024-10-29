package main

import (
	"fmt"
	"math/rand"
	"time"

	"neural-network/internal/activation"
	"neural-network/internal/network"
	"neural-network/internal/training"
)

type Word2Vec map[string][]float64

func main() {
	rand.Seed(time.Now().UnixNano())

	embedding := createEmbedding(10)

	inputSize := 20
	hiddenSize1 := 32
	hiddenSize2 := 16
	outputSize := len(embedding)

	layerSizes := []int{inputSize, hiddenSize1, hiddenSize2, outputSize}
	activationFuncs := []network.ActivationFunc{activation.ReLU, activation.ReLU, activation.Sigmoid}
	activationPrimes := []network.ActivationFunc{activation.ReLUPrime, activation.ReLUPrime, activation.SigmoidPrime}

	nn := network.NewNetwork(layerSizes, activationFuncs, activationPrimes)

	trainer := training.NewTrainerWithL2(nn, training.CrossEntropy, training.CrossEntropyPrime, 0.01, 0.0001)

	inputs, targets := createTrainingData(embedding)

	epochs := 10000
	batchSize := 32
	updateFrequency := 100 // Update every 100 epochs
	runningLossSum := 0.0
	runningLossCount := 0

	fmt.Println("Starting training...")
	startTime := time.Now()

	for i := 0; i < epochs; i++ {
		loss := trainer.TrainBatch(inputs, targets, batchSize)[0]
		runningLossSum += loss
		runningLossCount++

		if i%updateFrequency == 0 {
			avgLoss := runningLossSum / float64(runningLossCount)
			progress := float64(i) / float64(epochs) * 100
			elapsedTime := time.Since(startTime)
			estimatedTotalTime := elapsedTime * time.Duration(epochs) / time.Duration(i+1)
			remainingTime := estimatedTotalTime - elapsedTime

			fmt.Printf("\rEpoch %d/%d (%.2f%%) - Avg Loss: %.6f - Elapsed: %v - Remaining: %v",
				i, epochs, progress, avgLoss, elapsedTime.Round(time.Second), remainingTime.Round(time.Second))

			runningLossSum = 0
			runningLossCount = 0
		}

		learningRate := 0.01 / (1 + float64(i)/1000)
		trainer.SetLearningRate(learningRate)
	}

	fmt.Println("\nTraining completed.")

	// Test word pairs
	testPairs := [][]string{
		{"hello", "world"},
		{"neural", "network"},
		{"machine", "learning"},
		{"data", "science"},
		{"artificial", "intelligence"},
	}
	for _, pair := range testPairs {
		input := append(embedding[pair[0]], embedding[pair[1]]...)
		output := nn.Predict(input)
		predictedWord := getPredictedWord(output, embedding)
		fmt.Printf("Input: %s %s, Predicted: %s\n", pair[0], pair[1], predictedWord)
	}
}

func createEmbedding(dim int) Word2Vec {
	words := []string{
		"hello", "world", "neural", "network", "artificial", "intelligence",
		"machine", "learning", "data", "science", "algorithm", "model",
		"training", "prediction", "feature", "vector", "embedding",
		"representation", "neuron", "layer", "output", "input",
	}

	embedding := make(Word2Vec)
	for _, word := range words {
		vec := make([]float64, dim)
		for i := range vec {
			vec[i] = rand.Float64()*2 - 1 // Values between -1 and 1
		}
		embedding[word] = vec
	}
	return embedding
}

func createTrainingData(embedding Word2Vec) ([][]float64, [][]float64) {
	words := make([]string, 0, len(embedding))
	for w := range embedding {
		words = append(words, w)
	}

	var inputs [][]float64
	var targets [][]float64

	for i, w1 := range words {
		for j, w2 := range words {
			if i != j {
				input := append(embedding[w1], embedding[w2]...)
				inputs = append(inputs, input)

				target := make([]float64, len(embedding))
				for k := range words {
					if k != i && k != j {
						target[k] = 1.0 / float64(len(embedding)-2)
					}
				}
				targets = append(targets, target)
			}
		}
	}

	return inputs, targets
}

func getPredictedWord(output []float64, embedding Word2Vec) string {
	maxIndex := 0
	for i, val := range output {
		if val > output[maxIndex] {
			maxIndex = i
		}
	}

	words := make([]string, 0, len(embedding))
	for w := range embedding {
		words = append(words, w)
	}
	return words[maxIndex]
}
