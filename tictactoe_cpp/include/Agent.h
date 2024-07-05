#ifndef AGENT_H
#define AGENT_H

#include <unordered_map>
#include <array>
#include <vector>
#include <random>
#include <string>

class Agent {
public:
    Agent(double lr, double initial_epsilon, double epsilon_decay, double min_epsilon, double discount = 0.95);
    std::array<int, 2> get_action(const std::array<int, 9>& obs, const std::vector<int>& available_indices);
    void update(const std::array<int, 9>& obs, int action, int reward, bool terminated, const std::array<int, 9>& next_obs);
    void decay_epsilon();
    void save_q_table(const std::string& filename);
    void load_q_table(const std::string& filename);
    double epsilon;

private:
    struct ArrayHash {
        std::size_t operator()(const std::array<int, 9>& arr) const {
            std::size_t hash = 0;
            for (int i : arr) {
                hash ^= std::hash<int>{}(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2); // Simple but effective hash combiner
            }
            return hash;
        }
    };

    std::unordered_map<std::array<int, 9>, std::vector<int>, ArrayHash> q_table;
    double lr;
    double epsilon_decay;
    double min_epsilon;
    double discount;
    std::vector<double> training_error;
    std::mt19937 gen;
};

#endif // AGENT_H