#include <iostream>
#include <vector>
#include <array>
#include <unordered_map>
#include <iomanip>
#include <tuple>  
#include <cmath> 
#include "../include/TicTacToe.h"
#include "../include/Agent.h"

using namespace std;

int main() {
    TicTacToe env;
    Agent agent(1, 1.0, 8e-7, 0.01);
    int n_episodes = 1700000;
    bool done;
    int wins = 0;
    int total_wins = 0; 
    vector<double> epsilons;

    for (int episode = 0; episode < n_episodes; ++episode) {
        std::array<int, 9> obs;
        std::unordered_map<std::string, std::array<int, 9>> info;

        if (episode % 2 == 0){
            tie(obs, info) = env.reset(true);
        }else{
            tie(obs, info) = env.reset();
        }

        done = false;
        while (!done) {
            int action_index;
            std::tie(std::ignore, action_index) = agent.get_action(obs, env.available_indices);
            std::array<int, 9> next_obs;
            double reward;
            bool terminated;
            bool truncated;
            std::tie(next_obs, reward, terminated, truncated, info) = env.step(action_index, false);

            done = terminated;
            if (done && reward == 1) {
                wins += 1;
                total_wins += 1; // Increment total wins
            }

            agent.update(obs, action_index, reward, terminated, next_obs);
            obs = next_obs;
        }
        epsilons.push_back(agent.epsilon);
        agent.decay_epsilon();

        if ((episode + 1) % 50000 == 0) {
            cout << "Episode: " << episode + 1 << " "
                 << "epsilon: " << std::fixed << std::setprecision(3) << agent.epsilon << std::endl;            
            cout << "win_rate: " << std::fixed << std::setprecision(2) << (static_cast<double>(wins) / 50000 * 100) << "%" << endl;
            wins = 0;  // Reset wins counter for the next block of 50,000 episodes
        }
    }
    std::cout << "Overall win rate: " << (static_cast<double>(total_wins) / n_episodes * 100) << "%" << std::endl;
    std::cout << "Total wins: " << total_wins << std::endl;
    agent.save_q_table("tictactoe_q_table");




    for (int episode = 0; episode < n_episodes; ++episode) {
        std::array<int, 9> obs;
        std::unordered_map<std::string, std::array<int, 9>> info;
        if (episode % 2 == 0){
            tie(obs, info) = env.reset(true);
        }else{
        tie(obs, info) = env.reset();
        }
        done = false;
        while (!done) {
            int action_index;
            std::tie(std::ignore, action_index) = agent.get_action(obs, env.available_indices);
            std::array<int, 9> next_obs;
            double reward;
            bool terminated;
            bool truncated;
            std::tie(next_obs, reward, terminated, truncated, info) = env.step(action_index, true);

            done = terminated;
            if (done && reward == 1) {
                wins += 1;
                total_wins += 1; // Increment total wins
            }

            obs = next_obs;
        }
    }




    return 0;
}