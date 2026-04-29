#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "math/vector/vector.h"
#include "math/matrix/matrix.h"

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    Matrix weightsInputHidden;
    Matrix weightsHiddenOutput;

    float biasHidden;
    float biasOutput;

    float sigmoid(float x) const;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    float predict(float x, float y) const;

    Vector getWeights() const;
    void setWeights(const Vector& weights);
    int getTotalWeightsCount() const;
};

#endif