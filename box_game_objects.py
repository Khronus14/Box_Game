import numpy as np
import random as r
from numpy import ndarray, dtype


class Game:
    __slots__ = ('player1', 'player2', 'p1_search', 'p2_search', 'size',  'obstacles',
                 'board', 'state', "curPlayer", "running", 'p1_states', 'p2_states',
                 'p1_depth', 'p2_depth', 'timer', 'concise')

    def __init__(self, arguments):
        self.player1 = arguments[1].upper()
        self.player2 = arguments[2].upper()
        self.p1_search = self.player1[:2]
        self.p2_search = self.player2[:2]
        self.size = int(arguments[3])
        self.obstacles = int(arguments[4])
        self.board = self.create_board()
        self.state = np.copy(self.board)
        self.curPlayer = 1
        self.running = True
        self.p1_states = 0
        self.p2_states = 0
        self.p1_depth = float('inf')
        self.p2_depth = float('inf')
        self.set_depth()
        self.timer = 0
        self.concise = False if int(arguments[6]) == 0 else True

    def create_board(self) -> ndarray[int, dtype[int]]:
        """
        Method creates board and adds obstacles based on CLA.
        :return: The game board.
        """
        board = np.zeros((self.size, self.size), dtype=int)
        obstacles = []
        while len(obstacles) < self.obstacles:
            row = r.randint(0, self.size - 1)
            col = r.randint(0, self.size - 1)
            unique = True
            # ensure we create distinct obstacles
            for obs in obstacles:
                if obs[0] == row and obs[1] == col:
                    unique = False
                    break
            if unique:
                obstacles.append([row, col])
        for obs in obstacles:
            board[obs[0]][obs[1]] = 8
        # used for board replication during debug
        # board[1][3] = 8
        # board[3][1] = 8
        # board[2][0] = 8
        # board[4][4] = 8
        # board[5][5] = 8
        return board

    def set_depth(self) -> None:
        """
        Method to set search depth based on the player's search algorithm and
        evaluation method.
        :return: None
        """
        if self.player1.startswith('MM') and not self.player1.endswith('0'):
            self.p1_depth = 3
        if self.player1.startswith('AB') and not self.player1.endswith('0'):
            self.p1_depth = 5
        if self.player2.startswith('MM') and not self.player2.endswith('0'):
            self.p2_depth = 3
        if self.player2.startswith('AB') and not self.player2.endswith('0'):
            self.p2_depth = 5

    def to_move(self) -> int:
        """
        Returns the current player.
        :return: current player
        """
        return self.curPlayer

    def actions(self, state: list) -> list:
        """
        Returns the set of legal moves in state provided.
        :param state: The state under consideration.
        :return: A list of all legal moves (board index) in this state.
        """
        actions = []
        for index, x in np.ndenumerate(state):
            # index is top-left cell of play being considered
            if (x == 0 and index[0] < self.size - 1
                    and index[1] < self.size - 1
                    and state[index[0]][index[1] + 1] == 0
                    and state[index[0] + 1][index[1]] == 0
                    and state[index[0] + 1][index[1] + 1] == 0):
                actions.append(index)
        return actions

    def result(self, state: ndarray[int, dtype[int]], action: tuple, player: int):
        """
        Transition model which returns new state after taking the provided
        action.
        :param state: Current state of the game.
        :param action: Index of top-left cell of play to apply to state.
        :param player: Current turn's player.
        :return: State with action applied.
        """
        state[action[0]][action[1]] = player
        state[action[0]][action[1] + 1] = player
        state[action[0] + 1][action[1]] = player
        state[action[0] + 1][action[1] + 1] = player
        return state

    def is_valid(self, row: int, col: int) -> bool:
        if (self.state[row][col] == 8 or self.state[row + 1][col] == 8
                or self.state[row][col + 1] == 8 or self.state[row + 1][col + 1] == 8):
            return False
        return True

    def is_terminal(self, state: list) -> bool:
        """
        Determines if the provided state is terminal.
        :param state: State under consideration.
        :return: True iff state is terminal, otherwise False
        """
        for index, x in np.ndenumerate(state):
            # print(index, x)
            # if a 2x2 block of '0' exists -> not terminal
            if (x == 0 and index[0] < self.size - 1
                    and index[1] < self.size - 1
                    and state[index[0]][index[1] + 1] == 0
                    and state[index[0] + 1][index[1]] == 0
                    and state[index[0] + 1][index[1] + 1] == 0):
                return False
        return True

    def utility(self, state: list, player: int, is_terminal: bool) -> float:
        """
        Returns the value of the state to the provided player. The evaluation
        method used is based on the active player calling this method.
        :param state: State under consideration.
        :param player: Player considering the state.
        :param is_terminal: If this is a terminal state.
        :return: Value of provided state to the player.
        """
        value_p1 = 0
        value_p2 = 0
        final_value = 0
        # always use method zero to evaluate a terminal state
        if is_terminal:
            method = 0
        else:
            player_str = self.player1 if player == 1 else self.player2
            method = int(player_str[2])

        # logic masks to prevent re-checking evaluated cells
        mask_p1 = np.zeros((self.size, self.size), dtype=bool)
        mask_p2 = np.zeros((self.size, self.size), dtype=bool)
        for index, x in np.ndenumerate(state):
            if not mask_p1[index[0]][index[1]] and x == 1:
                temp_value, mask_p1 = self._cal_value(state, 1, index, 0, mask_p1, method)
                value_p1 = max(value_p1, temp_value)
            if not mask_p2[index[0]][index[1]] and x == -1:
                temp_value, mask_p2 = self._cal_value(state, -1, index, 0, mask_p2, method)
                value_p2 = min(value_p2, temp_value)

        # calculate final value of state being considered
        match method:
            # terminal state eval method binds value to 1, 0, or -1
            case 0:
                final_value = value_p1 + value_p2
                if final_value < 0:
                    final_value = -1
                elif final_value > 0:
                    final_value = 1
                else:
                    final_value = 0
            # in method 2, the p1 and p2 values had potential play values added in
            case 1 | 2:
                final_value = (value_p1 + value_p2) / 100
        return final_value

    def _cal_value(self, state: list, player: int, index: tuple, value: int,
                   mask: ndarray[int, dtype[bool]], method: int) -> tuple:
        """
        Helper function to find the largest contiguous block in a BFS manner.
        :param state: Current state to find value of.
        :param player: Player value to find in state.
        :param index: Starting location of BFS search in state.
        :param value: Current value of this state.
        :return: Updated value of state and binary mask of cells visited.
        """
        value += player
        visit_q = [[index[0], index[1]]]
        visited_q = [[index[0], index[1]]]
        mask[index[0]][index[1]] = True
        mask_positions = np.zeros((self.size, self.size), dtype=bool)

        while len(visit_q) > 0:
            cur_index = visit_q.pop(0)

            # method 2 additionally checks for potential of future plays to increase score
            if method == 2:
                self.open_edges(state, cur_index, mask_positions)

            # check up
            if cur_index[0] > 0 and state[cur_index[0] - 1][cur_index[1]] == player:
                if [cur_index[0] - 1, cur_index[1]] not in visited_q:
                    visit_q.append([cur_index[0] - 1, cur_index[1]])
                    visited_q.append([cur_index[0] - 1, cur_index[1]])
                    value += player
                    mask[cur_index[0] - 1][cur_index[1]] = True
            # check right
            if cur_index[1] < self.size - 1 and state[cur_index[0]][cur_index[1] + 1] == player:
                if [cur_index[0], cur_index[1] + 1] not in visited_q:
                    visit_q.append([cur_index[0], cur_index[1] + 1])
                    visited_q.append([cur_index[0], cur_index[1] + 1])
                    value += player
                    mask[cur_index[0]][cur_index[1] + 1] = True
            # check down
            if cur_index[0] < self.size - 1 and state[cur_index[0] + 1][cur_index[1]] == player:
                if [cur_index[0] + 1, cur_index[1]] not in visited_q:
                    visit_q.append([cur_index[0] + 1, cur_index[1]])
                    visited_q.append([cur_index[0] + 1, cur_index[1]])
                    value += player
                    mask[cur_index[0] + 1][cur_index[1]] = True
            # check left
            if cur_index[1] > 0 and state[cur_index[0]][cur_index[1] - 1] == player:
                if [cur_index[0], cur_index[1] - 1] not in visited_q:
                    visit_q.append([cur_index[0], cur_index[1] - 1])
                    visited_q.append([cur_index[0], cur_index[1] - 1])
                    value += player
                    mask[cur_index[0]][cur_index[1] - 1] = True
        # if using eval 2, calculate value of mask_positions and add to return value
        if method == 2:
            value += np.count_nonzero(mask_positions) * player
        return value, mask

    def open_edges(self, state: list, index: list, mask_positions: ndarray[int, dtype[bool]]) -> None:
        """
        Eval method that finds the number of future positions to increase the
        number of contiguous plays. The resulting number of True's in the mask
        indicate the number of possible plays that would increase the number of
        contiguous blocks.
        :param state: State under consideration.
        :param index: Cell from which to consider future plays.
        :param mask_positions: Binary mask to track all possible future plays
        :return: None
        """
        up_origin = [index[0] - 2, index[1] - 1]
        right_origin = [index[0] - 1, index[1] + 1]
        down_origin = [index[0] + 1, index[1] - 1]
        left_origin = [index[0] - 1, index[1] - 2]
        if up_origin[0] >= 0:
            if up_origin[1] >= 0:
                if (state[up_origin[0]][up_origin[1]] == 0
                        and state[up_origin[0] + 1][up_origin[1]] == 0
                        and state[up_origin[0]][up_origin[1] + 1] == 0
                        and state[up_origin[0] + 1][up_origin[1] + 1] == 0):
                    mask_positions[up_origin[0]][up_origin[1]] = True
            if (up_origin[1] + 2) < self.size:
                if (state[up_origin[0]][up_origin[1] + 1] == 0
                        and state[up_origin[0] + 1][up_origin[1] + 1] == 0
                        and state[up_origin[0]][up_origin[1] + 2] == 0
                        and state[up_origin[0] + 1][up_origin[1] + 2] == 0):
                    mask_positions[up_origin[0]][up_origin[1] + 1] = True
        if (right_origin[1] + 1) < self.size:
            if right_origin[0] >= 0:
                if (state[right_origin[0]][right_origin[1]] == 0
                        and state[right_origin[0] + 1][right_origin[1]] == 0
                        and state[right_origin[0]][right_origin[1] + 1] == 0
                        and state[right_origin[0] + 1][right_origin[1] + 1] == 0):
                    mask_positions[right_origin[0]][right_origin[1]] = True
            if (right_origin[0] + 2) < self.size:
                if (state[right_origin[0] + 1][right_origin[1]] == 0
                        and state[right_origin[0] + 2][right_origin[1]] == 0
                        and state[right_origin[0] + 1][right_origin[1] + 1] == 0
                        and state[right_origin[0] + 2][right_origin[1] + 1] == 0):
                    mask_positions[right_origin[0] + 1][right_origin[1]] = True
        if (down_origin[0] + 1) < self.size:
            if down_origin[1] >= 0:
                if (state[down_origin[0]][down_origin[1]] == 0
                        and state[down_origin[0] + 1][down_origin[1]] == 0
                        and state[down_origin[0]][down_origin[1] + 1] == 0
                        and state[down_origin[0] + 1][down_origin[1] + 1] == 0):
                    mask_positions[down_origin[0]][down_origin[1]] = True
            if (down_origin[1] + 2) < self.size:
                if (state[down_origin[0]][down_origin[1] + 1] == 0
                        and state[down_origin[0] + 1][down_origin[1] + 1] == 0
                        and state[down_origin[0]][down_origin[1] + 2] == 0
                        and state[down_origin[0] + 1][down_origin[1] + 2] == 0):
                    mask_positions[down_origin[0]][down_origin[1] + 1] = True
        if left_origin[1] >= 0:
            if left_origin[0] >= 0:
                if (state[left_origin[0]][left_origin[1]] == 0
                        and state[left_origin[0] + 1][left_origin[1]] == 0
                        and state[left_origin[0]][left_origin[1] + 1] == 0
                        and state[left_origin[0] + 1][left_origin[1] + 1] == 0):
                    mask_positions[left_origin[0]][left_origin[1]] = True
            if (left_origin[0] + 2) < self.size:
                if (state[left_origin[0] + 1][left_origin[1]] == 0
                        and state[left_origin[0] + 2][left_origin[1]] == 0
                        and state[left_origin[0] + 1][left_origin[1] + 1] == 0
                        and state[left_origin[0] + 2][left_origin[1] + 1] == 0):
                    mask_positions[left_origin[0] + 1][left_origin[1]] = True

    def update_game(self, move: tuple) -> None:
        """
        Method to update game board with provided move. The move provided is the
        actual move for the player's turn.
        :param move: Move to apply to game board.
        :return: None
        """
        self.board[move[0]][move[1]] = self.curPlayer
        self.board[move[0]][move[1] + 1] = self.curPlayer
        self.board[move[0] + 1][move[1]] = self.curPlayer
        self.board[move[0] + 1][move[1] + 1] = self.curPlayer
        self.state = np.copy(self.board)

    def display(self) -> None:
        """
        Method to print game board for debug/display.
        :return: None
        """
        for row in self.board:
            for value in row:
                print(f'{value:3}', end='')
            print()

    def final_score(self, state: list) -> tuple:
        """
        Calculates the final score for each player once no legal moves are available.
        :param state: Final state of the current game.
        :return: Score for player 1 and 2
        """
        value_p1 = 0
        value_p2 = 0
        mask = np.zeros((self.size, self.size), dtype=bool)
        for index, x in np.ndenumerate(state):
            if not mask[index[0]][index[1]]:
                if x == 1:
                    temp_value, mask_p1 = self._cal_value(state, 1, index, 0, mask, 0)
                    value_p1 = max(value_p1, temp_value)
                elif x == -1:
                    temp_value, mask_p1 = self._cal_value(state, -1, index, 0, mask, 0)
                    value_p2 = min(value_p2, temp_value)
        value_p1 //= 4
        value_p2 //= -4
        return value_p1, value_p2

    def __str__(self) -> None:
        """
        Prints what the current game is; players, board size, and obstacles.
        :return: None
        """
        return (self.player1 + ' vs. ' + self.player2 + ', ' + str(self.size) +
                'x' + str(self.size) + ' grid with ' + str(self.obstacles) +
                ' obstacle(s)')
