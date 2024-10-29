package network

type Network struct {
	Layers []*Layer
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

func (n *Network) Forward(inputs []float64) []float64 {
	currentOutput := inputs
	for _, layer := range n.Layers {
		currentOutput = layer.Forward(currentOutput)
	}
	return currentOutput
}

func (n *Network) Predict(inputs []float64) []float64 {
	return n.Forward(inputs)
}
