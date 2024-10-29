package network

type ActivationFunc func(float64) float64
type SliceActivationFunc func([]float64) []float64

type Layer struct {
	Neurons              []*Neuron
	ActivationFunc       ActivationFunc
	SliceActivationFunc  SliceActivationFunc
	ActivationPrime      ActivationFunc
	SliceActivationPrime SliceActivationFunc
}

func NewLayer(inputSize, neuronCount int, activationFunc ActivationFunc, activationPrime ActivationFunc) *Layer {
	neurons := make([]*Neuron, neuronCount)
	for i := range neurons {
		neurons[i] = NewNeuron(inputSize)
	}
	return &Layer{
		Neurons:         neurons,
		ActivationFunc:  activationFunc,
		ActivationPrime: activationPrime,
	}
}

func NewSliceLayer(inputSize, neuronCount int, sliceActivationFunc, sliceActivationPrime SliceActivationFunc) *Layer {
	neurons := make([]*Neuron, neuronCount)
	for i := range neurons {
		neurons[i] = NewNeuron(inputSize)
	}
	return &Layer{
		Neurons:              neurons,
		SliceActivationFunc:  sliceActivationFunc,
		SliceActivationPrime: sliceActivationPrime,
	}
}

func (l *Layer) Forward(inputs []float64) []float64 {
	outputs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activate(inputs)
	}

	if l.SliceActivationFunc != nil {
		return l.SliceActivationFunc(outputs)
	}

	for i := range outputs {
		outputs[i] = l.ActivationFunc(outputs[i])
	}
	return outputs
}
