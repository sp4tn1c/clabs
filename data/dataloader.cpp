#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

std::vector<DataPoint> DataLoader::loadFromCSV(const std::string& filename,
                                                size_t max_features) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        return data;
    }

    // Пропускаем заголовок
    std::string header;
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;

        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            if (!token.empty()) {
                values.push_back(std::stof(token));
            }
        }

        if (values.size() >= 2) {
            // Последний элемент — метка, остальные — признаки
            int label = static_cast<int>(values.back());
            std::vector<float> features(values.begin(), values.end() - 1);

            // ← ← ← ДОБАВИТЬ ZERO-PADDING ← ← ←
            if (max_features > 0 && features.size() < max_features) {
                features.resize(max_features, 0.0f);  // Дополняем нулями
            }

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