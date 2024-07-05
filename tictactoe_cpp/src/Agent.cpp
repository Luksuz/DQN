#include "../include/Agent.h"
#include <algorithm> // for std::max_element
#include <iostream> // for debugging purposes
#include <iostream>
#include <vector>
#include <fstream>


// Constructor implementation
Agent::Agent(double lr, double initial_epsilon, double epsilon_decay, double min_epsilon, double discount)
    : lr(lr), epsilon(initial_epsilon), epsilon_decay(epsilon_decay), min_epsilon(min_epsilon), discount(discount), gen(std::random_device{}()) {
    // Initialize any additional setup if needed
}

// Definition of get_action method
std::array<int, 2> Agent::get_action(const std::array<int, 9>& obs, const std::vector<int>& available_indices) {
    std::array<int, 2> action;
    auto compare = [](const std::array<int, 2>& a, const std::array<int, 2>& b) {
        return a[1] < b[1];
    };

    // Ensure that the Q-values for this state are initialized
    if (q_table.find(obs) == q_table.end()) {
        // Initialize the Q-values for this state with zeros
        q_table[obs] = std::vector<int>(9, 0); // Assuming 9 possible actions with initial Q-value 0
    }

    const auto& q_values = q_table[obs];

    // Random exploration with probability epsilon
    if (std::uniform_real_distribution<double>(0.0, 1.0)(gen) < epsilon) {
        action[0] = 1;  // Assuming the first element of the action array is 1 as in Python code
        action[1] = available_indices[std::uniform_int_distribution<int>(0, available_indices.size() - 1)(gen)];
    } else {
        // Exploitation: Choose action with max q_value for given obs
        std::vector<std::array<int, 2>> valid_actions;
        for (int index : available_indices) {
            valid_actions.push_back({index, q_values[index]});
        }
        auto max_it = std::max_element(valid_actions.begin(), valid_actions.end(), compare);
        action[0] = 1;  // Again, assuming the structure of action matches Python code
        action[1] = (*max_it)[0];
    }

    return action;
}

// Definition of update method
void Agent::update(const std::array<int, 9>& obs, int action, int reward, bool terminated, const std::array<int, 9>& next_obs) {
    // Ensure that the Q-values for this state and next state are initialized
    if (q_table.find(obs) == q_table.end()) {
        q_table[obs] = std::vector<int>(9, 0);
    }
    if (q_table.find(next_obs) == q_table.end()) {
        q_table[next_obs] = std::vector<int>(9, 0);
    }

    double future_q_value = (!terminated) ? *std::max_element(q_table[next_obs].begin(), q_table[next_obs].end()) : 0.0;
    double temporal_difference = reward + discount * future_q_value - q_table[obs][action];
    q_table[obs][action] += lr * temporal_difference;
}

void Agent::decay_epsilon() {
    epsilon = std::max(min_epsilon, epsilon - epsilon_decay);
}

void Agent::save_q_table(const std::string& filename) {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        if (file.is_open()) {
            for (const auto& [state, actions] : q_table) {
                file.write(reinterpret_cast<const char*>(&state), sizeof(state));
                file.write(reinterpret_cast<const char*>(actions.data()), actions.size() * sizeof(actions[0]));
            }
            file.close();
        } else {
            std::cerr << "Unable to open file for saving Q-table" << std::endl;
        }
    }
