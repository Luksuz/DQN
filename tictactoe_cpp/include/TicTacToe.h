#ifndef TICTACTOE_H
#define TICTACTOE_H

#include <array>
#include <vector>
#include <tuple>
#include <unordered_map>

class TicTacToe {
public:
    std::array<int, 9> state;
    std::vector<int> available_indices;

    TicTacToe();

    std::tuple<std::array<int, 9>> _get_obs() const;
    std::unordered_map<std::string, std::array<int, 9>> _get_info() const;
    std::tuple<std::array<int, 9>, double, bool, bool, std::unordered_map<std::string, std::array<int, 9>>>
    step(int action, bool human_opponent);
    std::tuple<std::array<int, 9>, std::unordered_map<std::string, std::array<int, 9>>> reset(bool player_2_advantage=false);
    void display_board();

private:
    std::tuple<bool, int> validate_state() const;
    void add_mark(int action, int mark);
    void update_available_indices();
    void debug();
};

#endif // TICTACTOE_H