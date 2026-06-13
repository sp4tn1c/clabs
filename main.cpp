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
#include <memory>  // ← ← ← ДОБАВЛЕНО: для std::unique_ptr

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

void train_neural_network(NeuralNetwork& net, const std::vector<DataPoint>& train_data) {
    NeuralTrainer trainer(net, train_data, 50, 200, 0.2f, 0.2f);
    trainer.train();
}

std::vector<int> predict_neural(const NeuralNetwork& net, const std::vector<DataPoint>& data) {
    std::vector<int> predictions;
    for (const auto& p : data) {
        float pred = net.predict(p.features);
        predictions.push_back((pred >= 0.5f) ? 1 : 0);
    }
    return predictions;
}

void print_metrics(const std::string& name, const ClassificationReport& report) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  " << name << ":" << std::endl;
    std::cout << "    Precision: " << report.precision() << std::endl;
    std::cout << "    Recall:    " << report.recall() << std::endl;
    std::cout << "    F1:        " << report.f1() << std::endl;
}

// ==================== ОЦЕНКА НА ДАТАСЕТЕ (80/20 split) ====================

struct DatasetResult {
    std::string name;
    size_t feature_count;
    size_t train_size;
    size_t test_size;
    float f1_neural;
    // ← ← ← УДАЛЕНО: NeuralNetwork* trained_net; (не использовалось, сырой указатель)
};

DatasetResult evaluate_dataset(const std::string& name, const std::string& filepath) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Evaluating: " << name << " (" << filepath << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Загрузка
    std::cout << "\n[1] Loading data..." << std::endl;
    auto data = DataLoader::loadFromCSV(filepath);
    if (data.empty()) {
        std::cerr << "  ERROR: No data loaded!" << std::endl;
        return {name, 0, 0, 0, 0.0f};  // ← Убран nullptr
    }

    size_t feature_count = DataLoader::getFeatureCount(data);
    std::cout << "  Loaded: " << data.size() << " points, " << feature_count << " features" << std::endl;

    // 2. Конвертация меток (-1 → 0)
    for (auto& p : data) { if (p.label == -1) p.label = 0; }

    // 3. Split 80/20
    std::cout << "\n[2] Splitting data (80/20)..." << std::endl;
    std::vector<DataPoint> train_data, test_data;
    DataLoader::splitTrainTest(data, train_data, test_data, 0.8f, 42);
    std::cout << "  Train: " << train_data.size() << ", Test: " << test_data.size() << std::endl;

    // 4. Нейросеть
    std::cout << "\n[3] Creating & Training Neural Network..." << std::endl;
    int hidden_size = static_cast<int>(feature_count) * 2;

    // ← ← ← ИСПРАВЛЕНО: unique_ptr вместо raw pointer
    auto net = std::make_unique<NeuralNetwork>(static_cast<int>(feature_count), hidden_size, 1);
    net->init_weights();
    train_neural_network(*net, train_data);

    // 5. Тестирование
    std::cout << "\n[4] Evaluating..." << std::endl;
    auto pred = predict_neural(*net, test_data);
    std::vector<int> labels;
    for (const auto& p : test_data) labels.push_back(p.label);

    auto report = F1Metric::calculate(pred, labels);
    print_metrics("Neural Network", report);

    // 6. Сохранение
    std::cout << "\n[5] Saving results..." << std::endl;
    std::string prefix = (name == "d1") ? "d1" : (name == "d2") ? "d2" : "d3";
    std::ofstream results("data/" + prefix + "_predictions.csv");
    for (size_t f = 0; f < feature_count; ++f) results << "feature_" << f << ",";
    results << "true_label,pred_neural\n";
    for (size_t i = 0; i < test_data.size(); ++i) {
        for (size_t f = 0; f < test_data[i].features.size(); ++f)
            results << test_data[i].features[f] << ",";
        results << test_data[i].label << "," << pred[i] << "\n";
    }
    results.close();

    // ← ← ← ИСПРАВЛЕНО: не возвращаем pointer
    return {name, feature_count, train_data.size(), test_data.size(), report.f1()};
}

// ==================== ДООБУЧЕНИЕ НА D3 (FINE-TUNING) ====================

float fine_tune_and_evaluate_d3(const std::string& d3_path,
                                 NeuralNetwork& base_net,  // ← ← ← ССЫЛКА вместо указателя
                                 size_t base_feature_count) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Fine-tuning on d3 (Defense Mode)     " << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Загрузка d3
    auto d3_data = DataLoader::loadFromCSV(d3_path);
    if (d3_data.empty()) {
        std::cerr << "  ERROR: Cannot load d3!" << std::endl;
        return 0.0f;
    }
    size_t d3_features = DataLoader::getFeatureCount(d3_data);
    std::cout << "  Loaded: " << d3_data.size() << " points, " << d3_features << " features" << std::endl;

    // 2. Конвертация меток
    for (auto& p : d3_data) { if (p.label == -1) p.label = 0; }

    // 3. Split 80/20 для d3
    std::vector<DataPoint> d3_train, d3_test;
    DataLoader::splitTrainTest(d3_data, d3_train, d3_test, 0.8f, 42);
    std::cout << "  Train: " << d3_train.size() << ", Test: " << d3_test.size() << std::endl;

    // 4. Fine-tuning: если число признаков совпадает — используем base_net
    NeuralNetwork* net_for_d3 = &base_net;  // ← Ссылка на base_net
    if (d3_features != base_feature_count) {
        std::cout << "  ⚠️  Feature count mismatch (" << base_feature_count << " → " << d3_features << ")" << std::endl;
        std::cout << "  Creating new network for d3..." << std::endl;

        // ← ← ← ИСПРАВЛЕНО: unique_ptr для временной сети
        auto temp_net = std::make_unique<NeuralNetwork>(static_cast<int>(d3_features),
                                                         static_cast<int>(d3_features) * 2, 1);
        temp_net->init_weights();
        train_neural_network(*temp_net, d3_train);
        net_for_d3 = temp_net.get();  // Используем сырой указатель только для вызова (не владеем!)

        // ← ← ← НЕ НУЖЕН delete: unique_ptr сам освободит память при выходе из scope
    } else {
        // Fine-tune: меньше эпох, меньшая "агрессия"
        std::cout << "  Fine-tuning existing model..." << std::endl;
        NeuralTrainer trainer(base_net, d3_train, 30, 50, 0.1f, 0.1f);
        trainer.train();
    }

    // 5. Оценка на d3_test
    std::cout << "\n[1] Evaluating on d3 test set..." << std::endl;
    auto pred = predict_neural(*net_for_d3, d3_test);
    std::vector<int> labels;
    for (const auto& p : d3_test) labels.push_back(p.label);

    auto report = F1Metric::calculate(pred, labels);
    print_metrics("Neural Network (fine-tuned)", report);

    // 6. Сохранение
    std::ofstream results("data/d3_predictions.csv");
    for (size_t f = 0; f < d3_features; ++f) results << "feature_" << f << ",";
    results << "true_label,pred_neural\n";
    for (size_t i = 0; i < d3_test.size(); ++i) {
        for (size_t f = 0; f < d3_test[i].features.size(); ++f)
            results << d3_test[i].features[f] << ",";
        results << d3_test[i].label << "," << pred[i] << "\n";
    }
    results.close();

    // ← ← ← УДАЛЕНО: ручной delete (unique_ptr сам освободит память)

    return report.f1();
}

// ==================== MAIN ====================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Lab 3-4: Universal Neural Network    " << std::endl;
    std::cout << "  (Fine-tuning supported)              " << std::endl;
    std::cout << "========================================" << std::endl;

    // ==================== ШАГ 1: Определяем макс. число признаков ====================
    std::cout << "\n[1] Detecting max features..." << std::endl;
    size_t max_features = 0;

    auto d1_check = DataLoader::loadFromCSV("data/d1.csv");
    max_features = std::max(max_features, DataLoader::getFeatureCount(d1_check));
    std::cout << "  d1: " << DataLoader::getFeatureCount(d1_check) << " features" << std::endl;

    auto d2_check = DataLoader::loadFromCSV("data/d2.csv");
    max_features = std::max(max_features, DataLoader::getFeatureCount(d2_check));
    std::cout << "  d2: " << DataLoader::getFeatureCount(d2_check) << " features" << std::endl;

    std::ifstream d3_check("data/d3.csv");
    bool has_d3 = d3_check.good();
    d3_check.close();
    if (has_d3) {
        auto d3_check_data = DataLoader::loadFromCSV("data/d3.csv");
        max_features = std::max(max_features, DataLoader::getFeatureCount(d3_check_data));
        std::cout << "  d3: " << DataLoader::getFeatureCount(d3_check_data) << " features" << std::endl;
    }

    std::cout << "  → Max features: " << max_features << std::endl;
    std::cout << "  → Creating universal network: " << max_features << " → "
              << (max_features * 2) << " → 1" << std::endl;

    // ==================== ШАГ 2: Создаём ОДНУ универсальную сеть ====================
    // ← ← ← ИСПОЛЬЗУЕМ ЗНАЧЕНИЕ (не указатель!) — память управляется автоматически
    NeuralNetwork net(static_cast<int>(max_features),
                      static_cast<int>(max_features) * 2, 1);
    net.init_weights();

    std::vector<DatasetResult> results;

    // ==================== ШАГ 3: Оценка на d1 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Evaluating: d1                        " << std::endl;
    std::cout << "========================================" << std::endl;

    auto d1_data = DataLoader::loadFromCSV("data/d1.csv", max_features);
    for (auto& p : d1_data) { if (p.label == -1) p.label = 0; }

    std::vector<DataPoint> d1_train, d1_test;
    DataLoader::splitTrainTest(d1_data, d1_train, d1_test, 0.8f, 42);
    std::cout << "  Train: " << d1_train.size() << ", Test: " << d1_test.size() << std::endl;

    std::cout << "  Training on d1..." << std::endl;
    train_neural_network(net, d1_train);

    auto pred = predict_neural(net, d1_test);
    std::vector<int> labels;
    for (const auto& p : d1_test) labels.push_back(p.label);
    auto report = F1Metric::calculate(pred, labels);
    print_metrics("Neural Network", report);
    results.push_back({"d1", max_features, d1_train.size(), d1_test.size(), report.f1()});

    // ==================== ШАГ 4: Оценка на d2 ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Evaluating: d2                        " << std::endl;
    std::cout << "========================================" << std::endl;

    auto d2_data = DataLoader::loadFromCSV("data/d2.csv", max_features);
    for (auto& p : d2_data) { if (p.label == -1) p.label = 0; }

    std::vector<DataPoint> d2_train, d2_test;
    DataLoader::splitTrainTest(d2_data, d2_train, d2_test, 0.8f, 42);
    std::cout << "  Train: " << d2_train.size() << ", Test: " << d2_test.size() << std::endl;

    std::cout << "  Fine-tuning on d2..." << std::endl;
    NeuralTrainer trainer_d2(net, d2_train, 50, 100, 0.15f, 0.15f);
    trainer_d2.train();

    pred = predict_neural(net, d2_test);
    labels.clear();
    for (const auto& p : d2_test) labels.push_back(p.label);
    report = F1Metric::calculate(pred, labels);
    print_metrics("Neural Network (fine-tuned)", report);
    results.push_back({"d2", max_features, d2_train.size(), d2_test.size(), report.f1()});

    // ==================== ШАГ 5: Итоговая оценка ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  FINAL RESULTS (d1 + d2)              " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);

    printf("%-10s | %-6s | %-6s | %-8s\n", "Dataset", "Train", "Test", "F1 Neural");
    printf("------------------------------------------------\n");
    for (const auto& r : results) {
        printf("%-10s | %-6zu | %-6zu | %-8.3f\n",
               r.name.c_str(), r.train_size, r.test_size, r.f1_neural);
    }

    float final_score = 0.5f * results[0].f1_neural + 0.5f * results[1].f1_neural;
    std::cout << "\n  >>> FINAL SCORE: " << final_score << " <<<" << std::endl;
    if (final_score >= 0.55f) {
        std::cout << "  ✅ PASSED! (>= 0.55)" << std::endl;
    } else {
        std::cout << "  ❌ FAILED! (< 0.55)" << std::endl;
    }

    // ==================== ШАГ 6: Дообучение на d3 (защита) ====================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  DEFENSE: d3 fine-tuning               " << std::endl;
    std::cout << "========================================" << std::endl;

    if (has_d3) {
        std::cout << "\n✅ d3.csv found — starting fine-tuning..." << std::endl;

        auto d3_data = DataLoader::loadFromCSV("data/d3.csv", max_features);
        for (auto& p : d3_data) { if (p.label == -1) p.label = 0; }

        std::vector<DataPoint> d3_train, d3_test;
        DataLoader::splitTrainTest(d3_data, d3_train, d3_test, 0.8f, 42);
        std::cout << "  Train: " << d3_train.size() << ", Test: " << d3_test.size() << std::endl;

        std::cout << "  Fine-tuning on d3..." << std::endl;
        NeuralTrainer trainer_d3(net, d3_train, 30, 50, 0.1f, 0.1f);
        trainer_d3.train();

        pred = predict_neural(net, d3_test);
        labels.clear();
        for (const auto& p : d3_test) labels.push_back(p.label);
        report = F1Metric::calculate(pred, labels);
        print_metrics("Neural Network (fine-tuned on d3)", report);

        std::cout << "\n  F1(d3) = " << report.f1() << std::endl;
        if (report.f1() >= 0.55f) {
            std::cout << "  ✅ d3 PASSED! (>= 0.55)" << std::endl;
        } else {
            std::cout << "  ⚠️  d3 below threshold (< 0.55)" << std::endl;
        }
    } else {
        std::cout << "\n⚠️  d3.csv not found" << std::endl;
        std::cout << "  On defense: put d3.csv in data/ and rerun" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ready for Defense!                    " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}