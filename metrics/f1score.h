#ifndef F1SCORE_HPP
#define F1SCORE_HPP

#include <vector>

struct ClassificationReport {
    int true_positive = 0;
    int false_positive = 0;
    int true_negative = 0;
    int false_negative = 0;
    
    float precision() const {
        int denom = true_positive + false_positive;
        return (denom > 0) ? static_cast<float>(true_positive) / denom : 0.0f;
    }
    
    float recall() const {
        int denom = true_positive + false_negative;
        return (denom > 0) ? static_cast<float>(true_positive) / denom : 0.0f;
    }
    
    float f1() const {
        float p = precision();
        float r = recall();
        return (p + r > 0) ? (2.0f * p * r) / (p + r) : 0.0f;
    }
};

class F1Metric {
public:
    // Считает метрики для бинарной классификации
    // predicted и labels должны содержать 0 или 1
    static ClassificationReport calculate(const std::vector<int>& predicted,
                                          const std::vector<int>& labels);
};

#endif // F1SCORE_HPP