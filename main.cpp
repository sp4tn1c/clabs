#include "generator/generator.h"
#include "classifier/classifier.h"
#include "geneticalg/geneticalg.h"
#include "neural/neuralnetwork.h"
#include "neural/neural_trainer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

// ==================== LAB 1: ЛИНЕЙНЫЙ КЛАССИФИКАТОР ====================
void runLab1() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  LAB 1: Binary Linear Classifier       " << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[1] Generating Data..." << std::endl;
    DataGenerator gen(42);
    auto data = gen.generate(150, 0.5f);
    gen.save_to_csv(data, "data/train.csv");

    std::cout << "  Points: " << data.size() << std::endl;
    std::cout << "  Saved: data/train.csv" << std::endl;

    std::cout << "\n[2] Training with Genetic Algorithm..." << std::endl;
    GeneticAlg ga(42, 50, 100, 0.1f, 0.1f);
    Vector best_weights = ga.train(data);

    std::cout << "\n[3] Results:" << std::endl;
    Classifier clf(2);
    clf.set_weights(best_weights);

    std::cout << "  w1 = " << best_weights.at(0) << std::endl;
    std::cout << "  w2 = " << best_weights.at(1) << std::endl;
    std::cout << "  b  = " << best_weights.at(2) << std::endl;

    float accuracy = clf.accuracy(data);
    std::cout << "  Accuracy: " << accuracy * 100 << "%" << std::endl;

    std::cout << "\n[4] Exporting..." << std::endl;
    std::ofstream results("data/results.csv");
    results << "x,y,true_label,predicted_label\n";
    for (const auto& p : data) {
        results << p.x << "," << p.y << "," << p.label << "," << clf.predict(p) << "\n";
    }
    results.close();

    std::ofstream weights_file("data/weights.csv");
    weights_file << "w1,w2,b\n" << best_weights.at(0) << "," << best_weights.at(1) << "," << best_weights.at(2) << "\n";
    weights_file.close();

    std::cout << "  Saved: data/results.csv, data/weights.csv" << std::endl;
}

// ==================== LAB 2: НЕЙРОННАЯ СЕТЬ ====================
void runLab2() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  LAB 2: Neural Network Classifier      " << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n[1] Generating Data..." << std::endl;
    DataGenerator generator(42);
    auto dataset = generator.generate(150, 0.8f);

    // Конвертируем метки -1/1 → 0/1 для нейросети
    for (auto& point : dataset) {
        if (point.label == -1) point.label = 0;
    }

    int class0 = 0, class1 = 0;
    for (const auto& p : dataset) {
        if (p.label == 0) class0++;
        else class1++;
    }
    std::cout << "  Total: " << dataset.size() << std::endl;
    std::cout << "  Class 0: " << class0 << ", Class 1: " << class1 << std::endl;

    // Train/Test split (80/20)
    std::mt19937 shuffleRng(42);
    std::shuffle(dataset.begin(), dataset.end(), shuffleRng);
    size_t splitIndex = dataset.size() * 8 / 10;

    std::vector<DataPoint> trainData(dataset.begin(), dataset.begin() + splitIndex);
    std::vector<DataPoint> testData(dataset.begin() + splitIndex, dataset.end());

    std::cout << "  Train: " << trainData.size() << std::endl;
    std::cout << "  Test: " << testData.size() << std::endl;

    std::cout << "\n[2] Creating Neural Network..." << std::endl;
    NeuralNetwork network(2, 4, 1);
    std::cout << "  Architecture: 2 -> 4 -> 1" << std::endl;
    std::cout << "  Weights: " << network.getTotalWeightsCount() << std::endl;

    std::cout << "\n[3] Training with Genetic Algorithm..." << std::endl;
    NeuralTrainer trainer(network, trainData, 100, 200, 0.2f, 0.2f);
    trainer.train();

    // Функция для расчёта точности
    auto calculateAccuracy = [](const NeuralNetwork& net, const std::vector<DataPoint>& data) -> float {
        if (data.empty()) return 0.0f;
        int correct = 0;
        for (const auto& point : data) {
            float pred = net.predict(point.x, point.y);
            int predictedClass = (pred >= 0.5f) ? 1 : 0;
            if (predictedClass == point.label) {
                correct++;
            }
        }
        return static_cast<float>(correct) / static_cast<float>(data.size());
    };

    float trainAccuracy = calculateAccuracy(network, trainData);
    float testAccuracy = calculateAccuracy(network, testData);

    std::cout << "\n[4] Results:" << std::endl;
    std::cout << "  Train Accuracy: " << trainAccuracy * 100.0f << "%" << std::endl;
    std::cout << "  Test Accuracy: " << testAccuracy * 100.0f << "%" << std::endl;

    std::cout << "\n[5] Exporting..." << std::endl;
    std::ofstream resultsFile("data/lab2_results.csv");
    resultsFile << "x,y,prediction,label\n";
    for (const auto& point : testData) {
        float pred = network.predict(point.x, point.y);
        resultsFile << point.x << "," << point.y << "," << pred << "," << point.label << "\n";
    }
    resultsFile.close();

    std::ofstream weightsFile("data/lab2_weights.csv");
    Vector weights = network.getWeights();
    for (int i = 0; i < weights.size(); ++i) {
        weightsFile << weights.at(i);
        if (i < weights.size() - 1) weightsFile << ",";
    }
    weightsFile.close();

    std::cout << "  Saved: data/lab2_results.csv, data/lab2_weights.csv" << std::endl;
}

// ==================== MAIN (ЕДИНСТВЕННАЯ ТОЧКА ВХОДА) ====================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Machine Learning Labs                 " << std::endl;
    std::cout << "  Lab 1: Linear Classifier (straight line)" << std::endl;
    std::cout << "  Lab 2: Neural Network (curved boundary)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Всегда запускаем обе лабы
    runLab1();
    runLab2();

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All Done!                             " << std::endl;
    std::cout << "  Run: python3 python/plot.py           " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;  // ← ЕДИНСТВЕННЫЙ RETURN
}