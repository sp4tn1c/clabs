#include "generator/generator.h"
#include "neural/neuralnetwork.h"
#include "neural/neural_trainer.h"
#include <iostream>
#include <fstream>

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
    
    NeuralNetwork network(2, 4, 1);
    std::cout << "Weights count: " << network.getTotalWeightsCount() << "\n\n";
    
    NeuralTrainer trainer(network, dataset, 100, 200, 0.2f, 0.2f);
    trainer.train();
    
    std::ofstream resultsFile("data/lab2_results.csv");
    resultsFile << "x,y,prediction,label\n";
    for (const auto& point : dataset) {
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