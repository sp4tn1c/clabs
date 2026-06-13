#include "classifier/classifier.h"

Classifier::Classifier(size_t input_size)
    : weights(input_size + 1) {}  // ← Только 1 аргумент! (0.0f по умолчанию)

void Classifier::set_weights(const Vector& w) {
    weights = w;
}

int Classifier::predict(const DataPoint& point) const {
    float score = weights.at(0) * point.features[0] +
                  weights.at(1) * point.features[1] +
                  weights.at(2);
    return (score >= 0) ? 1 : -1;
}

float Classifier::accuracy(const std::vector<DataPoint>& data) const {
    if (data.empty()) return 0.0f;
    int correct = 0;
    for (const auto& point : data) {
        if (predict(point) == point.label) correct++;
    }
    return static_cast<float>(correct) / data.size();
}