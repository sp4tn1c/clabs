#include "data/dataloader.h"  // ← Путь относительно корня
#include "generator/generator.h"  // ← Для DataPoint
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Testing DataLoader ===" << std::endl;

    // Тест d1 (2 признака)
    auto d1 = DataLoader::loadFromCSV("data/d1.csv");
    std::cout << "Loaded d1: " << d1.size() << " points, " 
              << DataLoader::getFeatureCount(d1) << " features\n";
    
    // Тест d2 (4 признака)
    auto d2 = DataLoader::loadFromCSV("data/d2.csv");
    std::cout << "Loaded d2: " << d2.size() << " points, " 
              << DataLoader::getFeatureCount(d2) << " features\n";
    
    // Тест split
    std::vector<DataPoint> train, test;
    DataLoader::splitTrainTest(d1, train, test);
    std::cout << "Split d1: train=" << train.size() << ", test=" << test.size() << "\n";
    
    return 0;
}