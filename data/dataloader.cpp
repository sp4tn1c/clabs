#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

std::vector<DataPoint> DataLoader::loadFromCSV(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return data;
    }
    
    // Пропускаем заголовок (feature_0,feature_1,...,target)
    std::string header;
    std::getline(file, header);
    
    // Читаем строки данных
    std::string line;
    while (std::getline(file, line)) {
        // Пропускаем пустые строки
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;
        
        // Парсим значения, разделённые запятой
        while (std::getline(ss, token, ',')) {
            // Удаляем пробелы
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            
            if (!token.empty()) {
                try {
                    values.push_back(std::stof(token));
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not parse value: " << token << std::endl;
                }
            }
        }
        
        // Последний элемент — метка, остальные — признаки
        if (values.size() >= 2) {
            std::vector<float> features(values.begin(), values.end() - 1);
            int label = static_cast<int>(values.back());
            data.emplace_back(std::move(features), label);
        }
    }
    
    file.close();
    return data;
}

void DataLoader::splitTrainTest(const std::vector<DataPoint>& data,
                                 std::vector<DataPoint>& train,
                                 std::vector<DataPoint>& test,
                                 float train_ratio,
                                 unsigned seed) {
    // Копируем данные для перемешивания
    std::vector<DataPoint> shuffled = data;
    
    // Перемешиваем
    std::mt19937 g(seed);
    std::shuffle(shuffled.begin(), shuffled.end(), g);
    
    // Разделяем
    size_t split_idx = static_cast<size_t>(shuffled.size() * train_ratio);
    train.assign(shuffled.begin(), shuffled.begin() + split_idx);
    test.assign(shuffled.begin() + split_idx, shuffled.end());
}

size_t DataLoader::getFeatureCount(const std::vector<DataPoint>& data) {
    if (data.empty()) return 0;
    return data[0].features.size();
}