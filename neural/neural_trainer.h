#ifndef NEURAL_TRAINER_H
#define NEURAL_TRAINER_H

#include "neuralnetwork.h"
#include "generator/generator.h"
#include "math/random/random.h"
#include <vector>
#include <algorithm>
#include <iostream>

class NeuralTrainer {
private:
    NeuralNetwork& network;
    const std::vector<DataPoint>& dataset;
    int populationSize;
    int generations;
    float mutationRate;
    float mutationScale;
    Random rng;
    
    struct Individual {
        Vector weights;
        float fitness;
    };
    
    float calculateFitness(const Vector& weights) {
        network.setWeights(weights);
        int correct = 0;
        for (const auto& point : dataset) {
            float pred = network.predict(point.x, point.y);
            if ((pred >= 0.5f) == point.label) correct++;
        }
        return (float)correct / dataset.size();
    }
    
public:
    NeuralTrainer(NeuralNetwork& net, const std::vector<DataPoint>& data, 
                  int popSize = 100, int gen = 200, 
                  float mutRate = 0.2f, float mutScale = 0.2f)
        : network(net), dataset(data), populationSize(popSize), generations(gen),
          mutationRate(mutRate), mutationScale(mutScale) {}
    
    void train() {
        int wCount = network.getTotalWeightsCount();
        std::vector<Individual> population(populationSize);
        
        // Initialize population
        for (int i = 0; i < populationSize; ++i) {
            population[i].weights = Vector(wCount);
            for (int j = 0; j < wCount; ++j) {
                population[i].weights.at(j) = rng.uniform(-1.0f, 1.0f);
            }
        }
        
        // Evolution
        for (int gen = 0; gen < generations; ++gen) {
            // Evaluate fitness
            for (int i = 0; i < populationSize; ++i) {
                population[i].fitness = calculateFitness(population[i].weights);
            }
            
            // Sort by fitness
            std::sort(population.begin(), population.end(),
                      [](const Individual& a, const Individual& b) {
                          return a.fitness > b.fitness;
                      });
            
            std::cout << "Generation " << gen + 1 << ", best accuracy: " 
                      << population[0].fitness * 100 << "%\n";
            
            // Создание новой популяции
            std::vector<Individual> newPopulation;
            
            // Сохраняем лучшую
            newPopulation.push_back(population[0]);
            
            while (newPopulation.size() < populationSize) {
                // выбор родителей
                int p1 = rng.uniform(0, populationSize - 1);
                int p2 = rng.uniform(0, populationSize - 1);
                int p3 = rng.uniform(0, populationSize - 1);
                int p4 = rng.uniform(0, populationSize - 1);
                
                int parent1 = (population[p1].fitness > population[p2].fitness) ? p1 : p2;
                int parent2 = (population[p3].fitness > population[p4].fitness) ? p3 : p4;
                
                // кроссовер
                Individual child1, child2;
                child1.weights = Vector(wCount);
                child2.weights = Vector(wCount);
                
                int point = rng.uniform(1, wCount - 1);
                for (int i = 0; i < wCount; ++i) {
                    if (i < point) {
                        child1.weights.at(i) = population[parent1].weights.at(i);
                        child2.weights.at(i) = population[parent2].weights.at(i);
                    } else {
                        child1.weights.at(i) = population[parent2].weights.at(i);
                        child2.weights.at(i) = population[parent1].weights.at(i);
                    }
                }
                
                // мутации
                for (int i = 0; i < wCount; ++i) {
                    if (rng.uniform(0.0f, 1.0f) < mutationRate) {
                        child1.weights.at(i) += rng.normal(0.0f, mutationScale);
                    }
                    if (rng.uniform(0.0f, 1.0f) < mutationRate) {
                        child2.weights.at(i) += rng.normal(0.0f, mutationScale);
                    }
                }
                
                newPopulation.push_back(child1);
                if (newPopulation.size() < populationSize) {
                    newPopulation.push_back(child2);
                }
            }
            
            population = newPopulation;
        }
        
        for (int i = 0; i < populationSize; ++i) {
            population[i].fitness = calculateFitness(population[i].weights);
        }
        std::sort(population.begin(), population.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });
        
        network.setWeights(population[0].weights);
        std::cout << "\nBest accuracy: " << population[0].fitness * 100 << "%\n";
    }
};

#endif