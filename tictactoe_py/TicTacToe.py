import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.action_space = self.board.flatten()
        self.available_indices = [i for i in range(len(self.action_space)) if self.action_space[i] == 0]
        
    def add_mark(self, action):
        mark, field = action
        row, col = divmod(field, 3)

        # update game state
        self.board[row, col] = mark
        self.action_space = self.board.flatten()
        self.available_indices = [i for i in range(len(self.action_space)) if self.action_space[i] == 0]

    def reset(self, player2_advantage=None, human_opponent=False):
        self.board = np.zeros((3, 3), dtype=int)
        self.action_space = self.board.flatten()
        self.available_indices = [i for i in range(len(self.action_space)) if self.action_space[i] == 0]


        # Player makes the first move (prompt for input)
        if player2_advantage:
            self.player_move(human_opponent=human_opponent)

        obs, info = self._get_obs(), self._get_info()
        return obs, info

    def player_move(self, human_opponent):
        if human_opponent:
            agent_2_action = int(input("Enter an unoccupied index on the board: "))
        else:
            agent_2_action = (-1, np.random.choice(self.available_indices))
        self.add_mark(agent_2_action)

    def _get_obs(self):
        return tuple(self.action_space)

    def _get_info(self):
        return {"action_space": self.action_space}

    def step(self, action, human_opponent=False):
        mark, field = action
        if self.action_space[field] != 0:
            raise ValueError("Invalid action: Position already occupied.")
        self.add_mark(action)
        terminated, winner = self.validate_board()
        
        if terminated:
            reward = winner if winner else 0
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        if human_opponent:
            agent_2_action = int(input("Enter an unoccupied indec on the board: "))
        else:
            agent_2_action = (-1, np.random.choice(self.available_indices))


        self.add_mark(agent_2_action)

        terminated, winner = self.validate_board()
        reward = 1 if winner == 1 else -1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
        
    def display_interface(self):
        print(self.board[0])
        print(self.board[1])
        print(self.board[2])
        
    def validate_board(self):
        board = self.board
        terminated = False
        winner = None

        # Check rows and columns
        for I in range(3):
            if np.all(board[I] == 1) or np.all(board[I] == -1):
                winner = board[I, 0]
                terminated = True
            if np.all(board[:, I] == 1) or np.all(board[:, I] == -1):
                winner = board[0, I]
                terminated = True

        # Check diagonals
        if np.all(board.diagonal() == 1) or np.all(board.diagonal() == -1):
            winner = board[0, 0]
            terminated = True
        if np.all(np.fliplr(board).diagonal() == 1) or np.all(np.fliplr(board).diagonal() == -1):
            winner = board[0, 2]
            terminated = True

        # Check for draw
        if not terminated and 0 not in board:
            winner = 0
            terminated = True

        return terminated, winner