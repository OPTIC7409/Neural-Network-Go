package network

import "math/rand"

type Network struct {
	DropoutRate float64
	Layers      []*Layer
}

func NewNetwork(layerSizes []int, activationFuncs []ActivationFunc, activationPrimes []ActivationFunc) *Network {
	if len(layerSizes) < 2 {
		panic("Network must have at least an input and output layer")
	}
	if len(activationFuncs) != len(layerSizes)-1 || len(activationPrimes) != len(layerSizes)-1 {
		panic("Must provide activation function and its derivative for each hidden and output layer")
	}

	layers := make([]*Layer, len(layerSizes)-1)
	for i := range layers {
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], activationFuncs[i], activationPrimes[i])
	}

	return &Network{
		Layers: layers,
	}
}

func NewNetworkWithSliceOutput(layerSizes []int, activationFuncs []ActivationFunc, activationPrimes []ActivationFunc, outputActivation, outputActivationPrime SliceActivationFunc) *Network {
	if len(layerSizes) < 2 {
		panic("Network must have at least an input and output layer")
	}
	if len(activationFuncs) != len(layerSizes)-2 || len(activationPrimes) != len(layerSizes)-2 {
		panic("Must provide activation function and its derivative for each hidden layer")
	}

	layers := make([]*Layer, len(layerSizes)-1)
	for i := range layers[:len(layers)-1] {
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], activationFuncs[i], activationPrimes[i])
	}
	layers[len(layers)-1] = NewSliceLayer(layerSizes[len(layerSizes)-2], layerSizes[len(layerSizes)-1], outputActivation, outputActivationPrime)

	return &Network{
		Layers: layers,
	}
}

func (n *Network) Predict(inputs []float64) []float64 {
	return n.Forward(inputs, true)
}

func NewNetworkWithDropout(layerSizes []int, activationFuncs, activationPrimes []ActivationFunc, dropoutRate float64) *Network {
	if len(layerSizes) < 2 {
		panic("Network must have at least an input and output layer")
	}
	if len(activationFuncs) != len(layerSizes)-1 || len(activationPrimes) != len(layerSizes)-1 {
		panic("Must provide activation function and its derivative for each hidden and output layer")
	}

	layers := make([]*Layer, len(layerSizes)-1)
	for i := range layers {
		layers[i] = NewLayer(layerSizes[i], layerSizes[i+1], activationFuncs[i], activationPrimes[i])
	}

	return &Network{
		Layers:      layers,
		DropoutRate: dropoutRate,
	}
}

func (n *Network) Forward(inputs []float64, isTraining bool) []float64 {
	currentOutput := inputs
	for i, layer := range n.Layers {
		currentOutput = layer.Forward(currentOutput)
		if isTraining && i < len(n.Layers)-1 { // Apply dropout only to hidden layers
			currentOutput = applyDropout(currentOutput, n.DropoutRate)
		}
	}
	return currentOutput
}

func applyDropout(output []float64, dropoutRate float64) []float64 {
	dropoutMask := make([]float64, len(output))
	for i := range output {
		if rand.Float64() > dropoutRate {
			dropoutMask[i] = 1.0
		} else {
			dropoutMask[i] = 0.0
		}
	}
	for i := range output {
		output[i] *= dropoutMask[i]
	}
	return output
}
