#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include "../generator/generator.h"
#include "../math/vector/vector.h"
#include <vector>

class Classifier {
private:
    Vector weights;  // [w1, w2, bias]

public:
    Classifier(size_t input_size);  // input_size=2 → weights.size()=3
    void set_weights(const Vector& w);
    int predict(const DataPoint& point) const;
    float accuracy(const std::vector<DataPoint>& data) const;
};

#endif