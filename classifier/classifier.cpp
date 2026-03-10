#include "classifier/classifier.h"
#include <cmath>

Classifier::Classifier(size_t input_size) : weights(input_size + 1) {}

void Classifier::set_weights(const Vector& w) {
    weights = w;
}

Vector Classifier::get_weights() const {
    return weights;
}

int Classifier::predict(const DataPoint& point) const {
    float score = weights.at(0) * point.x + 
                  weights.at(1) * point.y + 
                  weights.at(2);
    return (score >= 0) ? 1 : -1;
}

float Classifier::accuracy(const std::vector<DataPoint>& data) const {
    int correct = 0;
    for (const auto& p : data) {
        if (predict(p) == p.label) correct++;
    }
    return (float)correct / data.size();
}