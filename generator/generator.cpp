#include "generator/generator.h"
#include <fstream>
#include <iostream>

DataGenerator::DataGenerator(unsigned int seed) : rng(seed) {}

std::vector<DataPoint> DataGenerator::generate(size_t n_per_class, float separation) {
    std::vector<DataPoint> data;
    
    // Класс +1
    for (size_t i = 0; i < n_per_class; ++i) {
        float x = rng.normal(separation / 2.0f, 0.5f);
        float y = rng.normal(separation / 2.0f, 0.5f);
        data.push_back({x, y, 1});
    }
    
    // Класс -1
    for (size_t i = 0; i < n_per_class; ++i) {
        float x = rng.normal(-separation / 2.0f, 0.5f);
        float y = rng.normal(-separation / 2.0f, 0.5f);
        data.push_back({x, y, -1});
    }
    
    return data;
}

void DataGenerator::save_to_csv(const std::vector<DataPoint>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return;
    }
    
    file << "x,y,label\n";
    for (const auto& p : data) {
        file << p.x << "," << p.y << "," << p.label << "\n";
    }
    file.close();
    std::cout << "Saved " << data.size() << " points to " << filename << std::endl;
}

std::vector<DataPoint> DataGenerator::load_from_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << std::endl;
        return data;
    }
    
    std::string line;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t p1 = line.find(',');
        size_t p2 = line.find(',', p1 + 1);
        float x = std::stof(line.substr(0, p1));
        float y = std::stof(line.substr(p1 + 1, p2 - p1 - 1));
        int label = std::stoi(line.substr(p2 + 1));
        data.push_back({x, y, label});
    }
    
    file.close();
    std::cout << "Loaded " << data.size() << " points from " << filename << std::endl;
    return data;
}