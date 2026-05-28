#include "generator/generator.h"
#include <fstream>
#include <iostream>
#include <sstream>

DataGenerator::DataGenerator(unsigned int seed) : rng(seed) {}

std::vector<DataPoint> DataGenerator::generate(size_t n_per_class, float separation) {
    std::vector<DataPoint> data;
    
    // Класс +1: центр в (+separation/2, +separation/2)
    for (size_t i = 0; i < n_per_class; ++i) {
        float x = rng.normal(separation / 2.0f, 0.5f);
        float y = rng.normal(separation / 2.0f, 0.5f);
        data.push_back(DataPoint({x, y}, 1));  // ← ИСПРАВЛЕНО
    }

    // Класс -1: центр в (-separation/2, -separation/2)
    for (size_t i = 0; i < n_per_class; ++i) {
        float x = rng.normal(-separation / 2.0f, 0.5f);
        float y = rng.normal(-separation / 2.0f, 0.5f);
        data.push_back(DataPoint({x, y}, -1));  // ← ИСПРАВЛЕНО
    }

    return data;
}

void DataGenerator::save_to_csv(const std::vector<DataPoint>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }

    file << "feature_0,feature_1,target\n";  // ← Заголовок как у преподавателя
    for (const auto& p : data) {
        // ← ИСПРАВЛЕНО: используем features[0], features[1]
        file << p.features[0] << "," << p.features[1] << "," << p.label << "\n";
    }

    file.close();
    std::cout << "Saved " << data.size() << " points to " << filename << std::endl;
}

std::vector<DataPoint> DataGenerator::load_from_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return data;
    }

    // Пропускаем заголовок
    std::string header;
    std::getline(file, header);

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<float> values;

        while (std::getline(ss, token, ',')) {
            values.push_back(std::stof(token));
        }

        if (values.size() >= 3) {
            // ← ИСПРАВЛЕНО: создаём DataPoint с вектором признаков
            std::vector<float> features = {values[0], values[1]};
            int label = static_cast<int>(values[2]);
            data.push_back(DataPoint(std::move(features), label));
        }
    }

    file.close();
    return data;
}