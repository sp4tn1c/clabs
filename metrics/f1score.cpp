#include "f1score.h"

ClassificationReport F1Metric::calculate(const std::vector<int>& predicted,
                                          const std::vector<int>& labels) {
    ClassificationReport report;
    
    if (predicted.size() != labels.size()) {
        return report;  // пустой отчёт при ошибке
    }
    
    for (size_t i = 0; i < labels.size(); ++i) {
        int pred = predicted[i];
        int true_label = labels[i];
        
        if (true_label == 1 && pred == 1) {
            report.true_positive++;
        } else if (true_label == 0 && pred == 1) {
            report.false_positive++;
        } else if (true_label == 0 && pred == 0) {
            report.true_negative++;
        } else if (true_label == 1 && pred == 0) {
            report.false_negative++;
        }
    }
    
    return report;
}