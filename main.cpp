#include "generator/generator.h"
#include "classifier/classifier.h"
#include "geneticalg/geneticalg.h"
#include "neural/neuralnetwork.h"
#include "neural/neural_trainer.h"
#include "data/dataloader.h"
#include "metrics/f1score.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <iomanip>

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

// Обучение линейного классификатора через ГА
Vector train_linear_classifier(const std::vector<DataPoint>& train_data) {
    GeneticAlg ga(42, 50, 100, 0.1f, 0.1f);
    return ga.train(train_data);
}

// Обучение нейросети через NeuralTrainer
void train_neural_network(NeuralNetwork& net, const std::vector<DataPoint>& train_data) {
    NeuralTrainer trainer(net, train_data, 50, 200, 0.2f, 0.2f);
    trainer.train();
}

// Предсказания для набора данных (линейный классификатор)
std::vector<int> predict_linear(const Classifier& clf, const std::vector<DataPoint>& data) {
    std::vector<int> predictions;
    for (const auto& p : data) {
        predictions.push_back(clf.predict(p));
    }
    return predictions;
}

// Предсказания для набора данных (нейросеть)
std::vector<int> predict_neural(const NeuralNetwork& net, const std::vector<DataPoint>& data) {
    std::vector<int> predictions;
    for (const auto& p : data) {
        float pred = net.predict(p.features);
        predictions.push_back((pred >= 0.5f) ? 1 : 0);
    }
    return predictions;
}

// Печать отчёта по метрикам
void print_metrics(const std::string& name, const ClassificationReport& report) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Precision: " << report.precision() << std::endl;
    std::cout << "    Recall:    " << report.recall() << std::endl;
    std::cout << "    F1:        " << report.f1() << std::endl;
}

// ==================== ОБУЧЕНИЕ И ТЕСТ НА ОДНОМ ДАТАСЕТЕ ====================

struct DatasetResult {
    float f1_linear;
    float f1_neural;
    size_t train_size;
    size_t test_size;
};

DatasetResult evaluate_dataset(const std::string& name,
                                const std::string& filepath,
                                size_t feature_count) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Evaluating: " << name << " (" << filepath << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Загрузка данных
    std::cout << "\n[1] Loading data..." << std::endl;
    auto data = DataLoader::loadFromCSV(filepath);
    if (data.empty()) {
        std::cerr << "  ERROR: No data loaded!" << std::endl;
        return {0.0f, 0.0f, 0, 0};
    }
    std::cout << "  Loaded: " << data.size() << " points" << std::endl;
    std::cout << "  Features: " << DataLoader::getFeatureCount(data) << std::endl;

    // 2. Конвертация меток (если нужно)
    // Проверяем, есть ли метки -1 (линейный классификатор работает с -1/1)
    bool has_negative_labels = false;
    for (const auto& p : data) {
        if (p.label == -1) { has_negative_labels = true; break; }
    }

    // Если все метки 0/1, оставляем как есть (для нейросети)
    // Если есть -1, конвертируем -1→0 для нейросети позже

    // 3. Split 80/20
    std::cout << "\n[2] Splitting data (80/20)..." << std::endl;
    std::vector<DataPoint> train_data, test_data;
    DataLoader::splitTrainTest(data, train_data, test_data, 0.8f, 42);
    std::cout << "  Train: " << train_data.size() << std::endl;
    std::cout << "  Test:  " << test_data.size() << std::endl;

    // 4. Линейный классификатор
    std::cout << "\n[3] Training Linear Classifier..." << std::endl;
    Vector best_weights = train_linear_classifier(train_data);
    Classifier clf(2);
    clf.set_weights(best_weights);

    auto pred_linear = predict_linear(clf, test_data);
    std::vector<int> labels_linear;
    for (const auto& p : test_data) {
        // Конвертируем 0→-1 для сравнения с линейным классификатором
        labels_linear.push_back(p.label == 0 ? -1 : p.label);
    }
    auto report_linear = F1Metric::calculate(pred_linear, labels_linear);
    print_metrics("Linear Classifier", report_linear);

    // 5. Нейросеть
    std::cout << "\n[4] Training Neural Network..." << std::endl;
    int in_size = static_cast<int>(DataLoader::getFeatureCount(data));
    NeuralNetwork net(in_size, in_size * 2, 1);  // hidden = 2*input
    net.init_weights();
    train_neural_network(net, train_data);

    auto pred_neural = predict_neural(net, test_data);
    std::vector<int> labels_neural;
    for (const auto& p : test_data) {
        labels_neural.push_back(p.label);  // 0/1 для нейросети
    }
    auto report_neural = F1Metric::calculate(pred_neural, labels_neural);
    print_metrics("Neural Network", report_neural);

    // 6. Сохранение результатов
    std::cout << "\n[5] Saving results..." << std::endl;
    std::string prefix = (name == "d1") ? "d1" : "d2";

    std::ofstream results("data/" + prefix + "_predictions.csv");
    results << "feature_0,feature_1,feature_2,feature_3,true_label,pred_linear,pred_neural\n";
    for (size_t i = 0; i < test_data.size(); ++i) {
        for (size_t f = 0; f < test_data[i].features.size(); ++f) {
            results << test_data[i].features[f] << ",";
        }
        results << test_data[i].label << "," << pred_linear[i] << "," << pred_neural[i] << "\n";
    }
    results.close();
    std::cout << "  Saved: data/" << prefix << "_predictions.csv" << std::endl;

    return {report_linear.f1(), report_neural.f1(), train_data.size(), test_data.size()};
}

// ==================== MAIN ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Lab 3-4: Neural Network Evaluation   " << std::endl;
    std::cout << "  Datasets: d1.csv (2 features)        " << std::endl;
    std::cout << "            d2.csv (4 features)        " << std::endl;
    std::cout << "========================================" << std::endl;

    // Оценка на d1
    auto result_d1 = evaluate_dataset("d1", "data/d1.csv", 2);

    // Оценка на d2
    auto result_d2 = evaluate_dataset("d2", "data/d2.csv", 4);

    // Итоговая таблица
    std::cout << "\n========================================" << std::endl;
    std::cout << "  FINAL RESULTS                         " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    printf("%-15s | %-10s | %-10s | %-10s | %-10s\n",
           "Dataset", "Train", "Test", "F1 Linear", "F1 Neural");
    printf("------------------------------------------------------------------------\n");
    printf("%-15s | %-10zu | %-10zu | %-10.3f | %-10.3f\n",
           "d1", result_d1.train_size, result_d1.test_size, result_d1.f1_linear, result_d1.f1_neural);
    printf("%-15s | %-10zu | %-10zu | %-10.3f | %-10.3f\n",
           "d2", result_d2.train_size, result_d2.test_size, result_d2.f1_linear, result_d2.f1_neural);

    // Итоговая оценка по формуле из ТЗ
    std::cout << "\n========================================" << std::endl;
    std::cout << "  FINAL SCORE (by TЗ formula)           " << std::endl;
    std::cout << "========================================" << std::endl;

    // F1 для итоговой оценки берём от нейросети (т.к. Лаба 3-4 про неё)
    float final_score = 0.5f * result_d1.f1_neural + 0.5f * result_d2.f1_neural;

    std::cout << "  Formula: 0.5 × F1(d1) + 0.5 × F1(d2)" << std::endl;
    std::cout << "  F1(d1) = " << result_d1.f1_neural << std::endl;
    std::cout << "  F1(d2) = " << result_d2.f1_neural << std::endl;
    std::cout << "\n  >>> FINAL SCORE: " << final_score << " <<<" << std::endl;

    if (final_score >= 0.55f) {
        std::cout << "\n  ✅ PASSED! (>= 0.55)" << std::endl;
    } else {
        std::cout << "\n  ❌ FAILED! (< 0.55) - Need to improve model!" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Next step: Python integration (pybind11)" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}