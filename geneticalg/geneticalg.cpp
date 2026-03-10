#include "geneticalg/geneticalg.h"
#include <iostream>
#include <algorithm>

GeneticAlg::GeneticAlg(unsigned int seed, size_t pop, size_t generations, 
                       float mut_rate, float mut_scale)
    : rng(seed), pop_size(pop), max_generations(generations),
      mutation_rate(mut_rate), mutation_scale(mut_scale) {
    initialize_population();
}

void GeneticAlg::initialize_population() {
    population.resize(pop_size);
    for (auto& ind : population) {
        ind.genome = rng.random_vector(3, -1.0f, 1.0f);
        ind.fitness = 0.0f;
    }
}

void GeneticAlg::evaluate(const std::vector<DataPoint>& data, Classifier& clf) {
    for (auto& ind : population) {
        clf.set_weights(ind.genome);
        ind.fitness = clf.accuracy(data);
    }
}

Individual GeneticAlg::tournament_selection() {
    size_t a = (size_t)rng.uniform(0, pop_size);
    size_t b = (size_t)rng.uniform(0, pop_size);
    return population[a].fitness > population[b].fitness ? population[a] : population[b];
}

Individual GeneticAlg::crossover(const Individual& a, const Individual& b) {
    Individual child;
    child.genome = Vector(3);
    for (int i = 0; i < 3; ++i) {
        child.genome.at(i) = (rng.uniform(0, 1) < 0.5f) ? a.genome.at(i) : b.genome.at(i);
    }
    return child;
}

void GeneticAlg::evolve() {
    std::vector<Individual> new_population;
    
    auto best = *std::max_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
    new_population.push_back(best);
    
    while (new_population.size() < pop_size) {
        Individual parent1 = tournament_selection();
        Individual parent2 = tournament_selection();
        Individual child = crossover(parent1, parent2);
        child.genome = rng.mutate_vector(child.genome, mutation_rate, mutation_scale);
        new_population.push_back(child);
    }
    
    population = new_population;
}

Vector GeneticAlg::train(const std::vector<DataPoint>& data) {
    Classifier clf(2);
    
    std::cout << "\n=== Genetic Algorithm Training ===" << std::endl;
    
    for (size_t gen = 0; gen < max_generations; ++gen) {
        evaluate(data, clf);
        evolve();
        
        if (gen % 10 == 0 || gen == max_generations - 1) {
            auto best = *std::max_element(population.begin(), population.end(),
                [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
            std::cout << "Gen " << gen << ": Best fitness = " << best.fitness * 100 << "%" << std::endl;
        }
    }
    
    auto best = *std::max_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) { return a.fitness < b.fitness; });
    
    std::cout << "\nTraining complete! Final accuracy: " << best.fitness * 100 << "%" << std::endl;
    return best.genome;
}