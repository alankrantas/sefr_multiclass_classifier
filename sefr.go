// This is the multiclass classifier version of the SEFR algorithm for Go/TinyGo
// based on my Arduino C++ version.

package main

import (
	"math/rand"
	"time"
)

const (
	features   uint8 = 4  // number of features
	labels     uint8 = 3  // number of labels
	datafactor uint8 = 10 // scale factor of data
)

var (
	// dataset: the Iris dataset (times DATAFACTOR so it can be stored as integer and save space/memory)
	// target: labels of the Iris dataset
	dataset = [][features]uint16{{51, 35, 14, 2}, {49, 30, 14, 2}, {47, 32, 13, 2}, {46, 31, 15, 2}, {50, 36, 14, 2}, {54, 39, 17, 4}, {46, 34, 14, 3}, {50, 34, 15, 2}, {44, 29, 14, 2}, {49, 31, 15, 1}, {54, 37, 15, 2}, {48, 34, 16, 2}, {48, 30, 14, 1}, {43, 30, 11, 1}, {58, 40, 12, 2}, {57, 44, 15, 4}, {54, 39, 13, 4}, {51, 35, 14, 3}, {57, 38, 17, 3}, {51, 38, 15, 3}, {54, 34, 17, 2}, {51, 37, 15, 4}, {46, 36, 10, 2}, {51, 33, 17, 5}, {48, 34, 19, 2}, {50, 30, 16, 2}, {50, 34, 16, 4}, {52, 35, 15, 2}, {52, 34, 14, 2}, {47, 32, 16, 2}, {48, 31, 16, 2}, {54, 34, 15, 4}, {52, 41, 15, 1}, {55, 42, 14, 2}, {49, 31, 15, 2}, {50, 32, 12, 2}, {55, 35, 13, 2}, {49, 36, 14, 1}, {44, 30, 13, 2}, {51, 34, 15, 2}, {50, 35, 13, 3}, {45, 23, 13, 3}, {44, 32, 13, 2}, {50, 35, 16, 6}, {51, 38, 19, 4}, {48, 30, 14, 3}, {51, 38, 16, 2}, {46, 32, 14, 2}, {53, 37, 15, 2}, {50, 33, 14, 2}, {70, 32, 47, 14}, {64, 32, 45, 15}, {69, 31, 49, 15}, {55, 23, 40, 13}, {65, 28, 46, 15}, {57, 28, 45, 13}, {63, 33, 47, 16}, {49, 24, 33, 10}, {66, 29, 46, 13}, {52, 27, 39, 14}, {50, 20, 35, 10}, {59, 30, 42, 15}, {60, 22, 40, 10}, {61, 29, 47, 14}, {56, 29, 36, 13}, {67, 31, 44, 14}, {56, 30, 45, 15}, {58, 27, 41, 10}, {62, 22, 45, 15}, {56, 25, 39, 11}, {59, 32, 48, 18}, {61, 28, 40, 13}, {63, 25, 49, 15}, {61, 28, 47, 12}, {64, 29, 43, 13}, {66, 30, 44, 14}, {68, 28, 48, 14}, {67, 30, 50, 17}, {60, 29, 45, 15}, {57, 26, 35, 10}, {55, 24, 38, 11}, {55, 24, 37, 10}, {58, 27, 39, 12}, {60, 27, 51, 16}, {54, 30, 45, 15}, {60, 34, 45, 16}, {67, 31, 47, 15}, {63, 23, 44, 13}, {56, 30, 41, 13}, {55, 25, 40, 13}, {55, 26, 44, 12}, {61, 30, 46, 14}, {58, 26, 40, 12}, {50, 23, 33, 10}, {56, 27, 42, 13}, {57, 30, 42, 12}, {57, 29, 42, 13}, {62, 29, 43, 13}, {51, 25, 30, 11}, {57, 28, 41, 13}, {63, 33, 60, 25}, {58, 27, 51, 19}, {71, 30, 59, 21}, {63, 29, 56, 18}, {65, 30, 58, 22}, {76, 30, 66, 21}, {49, 25, 45, 17}, {73, 29, 63, 18}, {67, 25, 58, 18}, {72, 36, 61, 25}, {65, 32, 51, 20}, {64, 27, 53, 19}, {68, 30, 55, 21}, {57, 25, 50, 20}, {58, 28, 51, 24}, {64, 32, 53, 23}, {65, 30, 55, 18}, {77, 38, 67, 22}, {77, 26, 69, 23}, {60, 22, 50, 15}, {69, 32, 57, 23}, {56, 28, 49, 20}, {77, 28, 67, 20}, {63, 27, 49, 18}, {67, 33, 57, 21}, {72, 32, 60, 18}, {62, 28, 48, 18}, {61, 30, 49, 18}, {64, 28, 56, 21}, {72, 30, 58, 16}, {74, 28, 61, 19}, {79, 38, 64, 20}, {64, 28, 56, 22}, {63, 28, 51, 15}, {61, 26, 56, 14}, {77, 30, 61, 23}, {63, 34, 56, 24}, {64, 31, 55, 18}, {60, 30, 48, 18}, {69, 31, 54, 21}, {67, 31, 56, 24}, {69, 31, 51, 23}, {58, 27, 51, 19}, {68, 32, 59, 23}, {67, 33, 57, 25}, {67, 30, 52, 23}, {63, 25, 50, 19}, {65, 30, 52, 20}, {62, 34, 54, 23}, {59, 30, 51, 18}}
	target  = []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
)

// Sefr model object
type Sefr struct {
	Weights      [labels][features]float32 // model weights on each labels
	Bias         [labels]float32           // model bias on each labels
	TrainingTime time.Duration             // mode training time
}

// Fit (train) model with dataset and target
func (s *Sefr) Fit() {

	startTime := time.Now()

	// iterate all labels
	for l := uint8(0); l < labels; l++ {

		var countPos, countNeg uint16

		// iterate all features
		for f := uint8(0); f < features; f++ {

			var avgPos, avgNeg float32
			countPos = 0
			countNeg = 0
			for i, d := range dataset {
				if target[i] != l { // use "not the label" as positive class
					avgPos += float32(d[f])
					countPos++
				} else { // use the label as negative class
					avgNeg += float32(d[f])
					countNeg++
				}
			}
			avgPos /= (float32(countPos) * float32(datafactor))
			avgNeg /= (float32(countNeg) * float32(datafactor))

			// calculate weight of this label
			s.Weights[l][f] = (avgPos - avgNeg) / (avgPos + avgNeg)

		}

		// calculate average weighted score of data
		var avgPosW, avgNegW float32
		for i, d := range dataset {
			var weightedScore float32
			for f := uint8(0); f < features; f++ {
				weightedScore += float32(d[f]) * s.Weights[l][f]
			}
			if target[i] != l {
				avgPosW += weightedScore
			} else {
				avgNegW += weightedScore
			}
		}
		avgPosW /= (float32(countPos) * float32(datafactor))
		avgNegW /= (float32(countNeg) * float32(datafactor))

		// calculate bias of this label
		s.Bias[l] = -(float32(countNeg)*avgPosW + float32(countPos)*avgNegW) / float32(countPos+countNeg)

	}

	// calculate training time
	s.TrainingTime = time.Now().Sub(startTime)

}

// Predict label of a new data instance
func (s *Sefr) Predict(newData [features]uint16) uint8 {

	var score [labels]float32

	for l := uint8(0); l < labels; l++ {
		for f := uint8(0); f < features; f++ {
			// calculate weight of each labels
			score[l] += (float32(newData[f]) / float32(datafactor)) * s.Weights[l][f]
		}
		score[l] += s.Bias[l] // add bias of each labels
	}

	// find the min score (least possible label of "not the label")
	minScore := score[0]
	minLabel := uint8(0)
	for l, c := range score {
		if c < minScore {
			minScore = c
			minLabel = uint8(l)
		}
	}

	return minLabel // return prediction

}

// ==================================================

func main() {

	rand.Seed(42)

	sefr := Sefr{}
	sefr.Fit() // train SEFR model

	for {

		// randomly pick a random data instance in dataset as test data

		index := rand.Intn(len(dataset))
		var testData [features]uint16
		println("Test data:")
		for f := uint8(0); f < features; f++ {
			var sign float32
			if rand.Intn(2) == 0 {
				sign = 1.0
			} else {
				sign = -1.0
			}
			// randomly add or subtract 10-30% to each feature
			change := float32(rand.Intn(3)+1) / 10.0
			data := float32(dataset[index][f])
			testData[f] = uint16(data + data*change*sign)
			println(float32(testData[f]) / float32(datafactor))
		}

		// predict label
		prediction := sefr.Predict(testData)

		// compare the results
		print("Predicted label: ", prediction)
		print(" / actual label: ", target[index])
		println(" / SEFR training time: ", sefr.TrainingTime/1000000, " ms\n")

		time.Sleep(time.Millisecond * 1000)

	}

}
