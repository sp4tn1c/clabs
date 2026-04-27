#include "neuralnetwork.h"
#include "math/random/random.h"
#include <cmath>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize),
      weightsInputHidden(hiddenSize, inputSize),
      weightsHiddenOutput(outputSize, hiddenSize),
      biasHidden(0.0f), biasOutput(0.0f)
{
    Random rng;
    
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weightsInputHidden.at(i, j) = rng.uniform(-0.5f, 0.5f);
        }
    }
    
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsHiddenOutput.at(i, j) = rng.uniform(-0.5f, 0.5f);
        }
    }
    
    biasHidden = rng.uniform(-0.5f, 0.5f);
    biasOutput = rng.uniform(-0.5f, 0.5f);
}

float NeuralNetwork::sigmoid(float x) const {
    return 1.0f / (1.0f + std::exp(-x));
}

float NeuralNetwork::predict(float x, float y) const {
    Vector input(2);
    input.at(0) = x;
    input.at(1) = y;
    
    Vector hidden = weightsInputHidden.multiply(input);
    for (int i = 0; i < hiddenSize; ++i) {
        float val = hidden.at(i) + biasHidden;
        hidden.at(i) = sigmoid(val);
    }
    
    Vector outputVec = weightsHiddenOutput.multiply(hidden);
    float output = outputVec.at(0) + biasOutput;
    
    return sigmoid(output);
}

Vector NeuralNetwork::getWeights() const {
    int total = getTotalWeightsCount();
    Vector weights(total);
    int idx = 0;
    
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights.at(idx++) = weightsInputHidden.at(i, j);
        }
    }
    
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weights.at(idx++) = weightsHiddenOutput.at(i, j);
        }
    }
    
    weights.at(idx++) = biasHidden;
    weights.at(idx++) = biasOutput;
    
    return weights;
}

void NeuralNetwork::setWeights(const Vector& weights) {
    int expected = getTotalWeightsCount();
    if (weights.size() != expected) {
        throw std::runtime_error("Wrong weights count");
    }
    
    int idx = 0;
    
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weightsInputHidden.at(i, j) = weights.at(idx++);
        }
    }
    
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weightsHiddenOutput.at(i, j) = weights.at(idx++);
        }
    }
    
    biasHidden = weights.at(idx++);
    biasOutput = weights.at(idx++);
}

int NeuralNetwork::getTotalWeightsCount() const {
    return (hiddenSize * inputSize) + (outputSize * hiddenSize) + 2;
}