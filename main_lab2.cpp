#include "generator/generator.h"
#include "neural/neuralnetwork.h"
#include "neural/neural_trainer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

static float calculateAccuracy(const NeuralNetwork& network, const std::vector<DataPoint>& data) {
    if (data.empty()) return 0.0f;

    int correct = 0;
    for (const auto& point : data) {
        float pred = network.predict(point.x, point.y);
        int predictedClass = (pred >= 0.5f) ? 1 : 0;
        if (predictedClass == point.label) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(data.size());
}

int main() {
    std::cout << "=== Lab 2: Neural Network + Genetic Algorithm ===\n\n";
    
    DataGenerator generator(42);
    auto dataset = generator.generate(150, 0.8f);
    
    // После генерации dataset преобразую метки из -1/1 в 0/1
    for (auto& point : dataset) {
        if (point.label == -1) point.label = 0;
    }
    
    std::cout << "Generated points: " << dataset.size() << "\n";
    int class0 = 0, class1 = 0;
    for (const auto& p : dataset) {
        if (p.label == 0) class0++;
        else class1++;
    }
    std::cout << "Class 0: " << class0 << "\n";
    std::cout << "Class 1: " << class1 << "\n\n";

    // Перемешиваем и делим данные на train/test (80/20)
    std::mt19937 shuffleRng(42);
    std::shuffle(dataset.begin(), dataset.end(), shuffleRng);
    size_t splitIndex = dataset.size() * 8 / 10;

    std::vector<DataPoint> trainData(dataset.begin(), dataset.begin() + splitIndex);
    std::vector<DataPoint> testData(dataset.begin() + splitIndex, dataset.end());

    std::cout << "Train points: " << trainData.size() << "\n";
    std::cout << "Test points: " << testData.size() << "\n\n";
    
    NeuralNetwork network(2, 4, 1);
    std::cout << "Weights count: " << network.getTotalWeightsCount() << "\n\n";
    
    // Обучаемся только на train-части
    NeuralTrainer trainer(network, trainData, 100, 200, 0.2f, 0.2f);
    trainer.train();

    float trainAccuracy = calculateAccuracy(network, trainData);
    float testAccuracy = calculateAccuracy(network, testData);
    std::cout << "\nTrain accuracy: " << trainAccuracy * 100.0f << "%\n";
    std::cout << "Test accuracy: " << testAccuracy * 100.0f << "%\n";
    
    std::ofstream resultsFile("data/lab2_results.csv");
    resultsFile << "x,y,prediction,label\n";
    // Экспортируем предсказания на test-части, чтобы оценка была честной
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
    
    std::cout << "\nResults saved to data/lab2_results.csv\n";
    std::cout << "Weights saved to data/lab2_weights.csv\n";
    
    return 0;
}