#include "../include/TicTacToe.h"
#include <algorithm>
#include <iostream>
#include <random>

TicTacToe::TicTacToe() : state{0}, available_indices{0, 1, 2, 3, 4, 5, 6, 7, 8} {}

std::tuple<std::array<int, 9>> TicTacToe::_get_obs() const {
    return std::make_tuple(state);
}

std::unordered_map<std::string, std::array<int, 9>> TicTacToe::_get_info() const {
    return {{"state", state}};
}

std::tuple<std::array<int, 9>, double, bool, bool, std::unordered_map<std::string, std::array<int, 9>>>
TicTacToe::step(int action, bool human_opponent) {
    int mark = 1; // Assuming mark for the current player is 1
    int field = action; // Extract field from action
    
    // Add mark to the game state
    state[field] = mark;
    update_available_indices();

    bool terminated;
    int winner;
    std::tie(terminated, winner) = validate_state();

    double reward = 0.0;
    if (terminated) {
        if (winner == 1) {
            reward = 1.0;
        }
        std::array<int, 9> observation;
        std::tie(observation) = _get_obs();
        std::unordered_map<std::string, std::array<int, 9>> info = _get_info();
        return std::make_tuple(observation, reward, terminated, false, info);
    }


    int agent_2_action;
    if (human_opponent) {
        display_board();
        std::cout << "Enter your move (index): ";
        std::cin >> agent_2_action; // Read action from human player
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, available_indices.size() - 1);
        int random_index = dis(gen); // Generate random index
        agent_2_action = available_indices[random_index]; // Select action from available indices
    }

    add_mark(agent_2_action, -1);
    update_available_indices();

    // Update game termination and winner after agent's action
    std::tie(terminated, winner) = validate_state();

    if (terminated) {
        if (winner == -1){
            reward = -1;
        }
        std::array<int, 9> observation;
        std::tie(observation) = _get_obs();
        std::unordered_map<std::string, std::array<int, 9>> info = _get_info();
        return std::make_tuple(observation, reward, terminated, false, info);
    }

    std::array<int, 9> observation;
    std::tie(observation) = _get_obs();
    std::unordered_map<std::string, std::array<int, 9>> info = _get_info();
    return std::make_tuple(observation, reward, terminated, false, info);
}

std::tuple<std::array<int, 9>, std::unordered_map<std::string, std::array<int, 9>>> TicTacToe::reset(bool player_2_advantage) {
    for (int i = 0; i < state.size(); i++) {
        state[i] = 0;
    }
    available_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    if (player_2_advantage){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, available_indices.size() - 1);
        int random_index = dis(gen); // Generate random index
        state[random_index] = -1;
        update_available_indices();
    }
    return std::make_tuple(state, _get_info());
}

std::tuple<bool, int> TicTacToe::validate_state() const {
    bool terminated = false;
    int winner = 0; // 1 for player 1 (X), -1 for player 2 (O)

    static const std::array<std::array<int, 3>, 8> win_conditions = {{
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, // Rows
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, // Columns
        {0, 4, 8}, {2, 4, 6}             // Diagonals
    }};

    for (const std::array<int, 3>& condition : win_conditions) {
        if (state[condition[0]] != 0 &&
            state[condition[0]] == state[condition[1]] &&
            state[condition[1]] == state[condition[2]]) {
            winner = state[condition[0]]; // Set winner based on the first matching element
            terminated = true;
            break;
        }
    }

    if (std::find(state.begin(), state.end(), 0) == state.end()) {
        terminated = true;
    }

    return std::make_tuple(terminated, winner);
}

void TicTacToe::add_mark(int action, int mark) {
    state[action] = mark;
}

void TicTacToe::update_available_indices() {
    available_indices.clear();
    for (int i = 0; i < state.size(); i++) {
        if (state[i] == 0) {
            available_indices.push_back(i);
        }
    }
}

void TicTacToe::debug(){
    std::cout << "Available indices: ";
    for (int i = 0; i < available_indices.size(); ++i) {
        std::cout << available_indices[i];
        if (i != available_indices.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

void TicTacToe::display_board() {
    // Assuming state is a 1D array of size 9 (flattened representation of a 3x3 board)
    for (int i = 0; i < 9; ++i) {
        // Print the cell value
        std::cout << state[i];
        // Print a vertical separator if it's not the last column of a row
        if ((i + 1) % 3 != 0) {
            std::cout << " | ";
        }
        // Print a newline if it's the end of a row, except after the last row
        else if (i != 8) {
            std::cout << std::endl << "---------" << std::endl;
        }
    }
    std::cout << std::endl;
}