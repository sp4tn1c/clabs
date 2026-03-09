#include "generator/generator.h"
#include <iostream>

int main() {
    std::cout << "=== LABA 1: Data Generator Test ===" << std::endl;

    DataGenerator gen(42);
    auto data = gen.generate(100, 2.0f);
    gen.save_to_csv(data, "data/train.csv");

    std::cout << "\n=== Generator Works! ===" << std::endl;
    std::cout << "Generated " << data.size() << " points" << std::endl;

    return 0;
}