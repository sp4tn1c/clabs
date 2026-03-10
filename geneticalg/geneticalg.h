#ifndef GENETICALG_HPP
#define GENETICALG_HPP

#include "vector/vector.h"
#include "classifier/classifier.h"
#include "generator/generator.h"
#include "random/random.h"
#include <vector>

struct Individual {
    Vector genome;
    float fitness;

    // Инициализируем genome размером 3 (w1, w2, b)
    Individual() : genome(3), fitness(0.0f) {}
};

class GeneticAlg {
private:
    std::vector<Individual> population;
    Random rng;
    size_t pop_size;
    size_t max_generations;
    float mutation_rate;
    float mutation_scale;
    
    void initialize_population();
    void evaluate(const std::vector<DataPoint>& data, Classifier& clf);
    void evolve();
    Individual tournament_selection();
    Individual crossover(const Individual& a, const Individual& b);
    
public:
    GeneticAlg(unsigned int seed = 42, size_t pop = 50, 
               size_t generations = 100, float mut_rate = 0.1f, float mut_scale = 0.1f);
    Vector train(const std::vector<DataPoint>& data);
};

#endif // GENETICALG_HPP