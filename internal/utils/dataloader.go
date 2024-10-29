package utils

import (
	"encoding/csv"
	"math/rand"
	"os"
	"strconv"
)

type DataSet struct {
	Inputs  [][]float64
	Targets [][]float64
}

func LoadCSV(filename string, inputCols, targetCols []int) (*DataSet, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	dataset := &DataSet{
		Inputs:  make([][]float64, len(records)),
		Targets: make([][]float64, len(records)),
	}

	for i, record := range records {
		dataset.Inputs[i] = make([]float64, len(inputCols))
		dataset.Targets[i] = make([]float64, len(targetCols))

		for j, col := range inputCols {
			value, err := strconv.ParseFloat(record[col], 64)
			if err != nil {
				return nil, err
			}
			dataset.Inputs[i][j] = value
		}

		for j, col := range targetCols {
			value, err := strconv.ParseFloat(record[col], 64)
			if err != nil {
				return nil, err
			}
			dataset.Targets[i][j] = value
		}
	}

	return dataset, nil
}

func SplitDataSet(dataset *DataSet, trainRatio float64) (*DataSet, *DataSet) {
	trainSize := int(float64(len(dataset.Inputs)) * trainRatio)

	trainSet := &DataSet{
		Inputs:  dataset.Inputs[:trainSize],
		Targets: dataset.Targets[:trainSize],
	}

	testSet := &DataSet{
		Inputs:  dataset.Inputs[trainSize:],
		Targets: dataset.Targets[trainSize:],
	}

	return trainSet, testSet
}

func ShuffleDataSet(dataset *DataSet) {
	for i := len(dataset.Inputs) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		dataset.Inputs[i], dataset.Inputs[j] = dataset.Inputs[j], dataset.Inputs[i]
		dataset.Targets[i], dataset.Targets[j] = dataset.Targets[j], dataset.Targets[i]
	}
}
