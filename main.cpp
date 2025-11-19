#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

#include "boltzmann.hpp"


std::vector<Eigen::VectorXi> generate_random_data(size_t n_samples, size_t n_visible, double sparsity = 0.3) {
    std::vector<Eigen::VectorXi> data;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < n_samples; ++i) {
        Eigen::VectorXi sample(n_visible);
        for (size_t j = 0; j < n_visible; ++j) {
            sample(j) = (dist(rng) < sparsity) ? 1 : 0;
        }
        data.push_back(sample);
    }
    return data;
}

std::vector<Eigen::VectorXi> generate_bars_data(size_t n_samples, size_t grid_size = 4) {
    size_t n_visible = grid_size * grid_size;
    std::vector<Eigen::VectorXi> data;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> bar_dist(0, 2 * grid_size - 1);
    std::uniform_real_distribution<double> noise_dist(0.0, 1.0);
    
    for (size_t i = 0; i < n_samples; ++i) {
        Eigen::VectorXi sample = Eigen::VectorXi::Zero(n_visible);
        
        // Choose random horizontal or vertical bar
        size_t bar_type = bar_dist(rng);
        
        if (bar_type < grid_size) {
            // Horizontal bar
            size_t row = bar_type;
            for (size_t col = 0; col < grid_size; ++col) {
                sample(row * grid_size + col) = 1;
            }
        } else {
            // Vertical bar
            size_t col = bar_type - grid_size;
            for (size_t row = 0; row < grid_size; ++row) {
                sample(row * grid_size + col) = 1;
            }
        }
        
        // Add noise
        for (size_t j = 0; j < n_visible; ++j) {
            if (noise_dist(rng) < 0.05) { // 5% noise
                sample(j) = 1 - sample(j);
            }
        }
        
        data.push_back(sample);
    }
    return data;
}


int main() {

    size_t n_visible = 8; 
    size_t n_hidden = 4; 
    BernoulliRBM machine(n_visible, n_hidden); 

    auto data = generate_random_data(50, n_visible); 
    // auto data = generate_bars_data(20, n_visible); 
    
    auto epochs = 50000; 
    auto k = 5; 
    auto lr = 0.001; 

    machine.train(data, epochs, k, lr);
    machine.visualize_weights(); 
    return 0; 
}