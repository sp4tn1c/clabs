#include "activation/activation.h"
#include <cmath>

float sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        float exp_x = std::exp(x);
        return exp_x / (1.0f + exp_x);
    }
}

Vector sigmoid(const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result.at(i) = sigmoid(v.at(i));
    }
    return result;
}

float relu(float x) {
    return (x > 0) ? x : 0.0f;
}

Vector relu(const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result.at(i) = relu(v.at(i));
    }
    return result;
}