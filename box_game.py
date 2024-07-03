"""
box_game.py
Author: David Robinson, ddr6248@rit.edu

Program that simulates the box game with various game searches and evaluation
functions.

MM0: Minimax with no evaluation and complete game tree search
MM1: Minimax with evaluation function 1 and a search depth limit of 3
MM2: Minimax with evaluation function 2 and a search depth limit of 3
AB0: Alpha-beta with no evaluation and complete game tree search
AB1: Alpha-beta with evaluation function 1 and a depth limit of 5
AB2: Alpha-beta with evaluation function 2 and a depth limit of 5

Evaluation function 1: Maximize own score AND minimize opponents score
Evaluation function 2: Method 1 plus considers number of future plays that can
    increase score

"""
import sys
import time as t
import numpy as np
from box_game_objects import Game


def error_and_exit(error: str) -> None:
    """
    Handles error messages and exits program
    :param error: Description of error.
    :return: None
    """
    print(error)
    sys.exit("Usage: python box_game.py {player1} {player2} {size} {obstacles} {iterations} {concise}\n"
             "\tplayer1/player2 - MM0, MM1, MM2, AB0, AB1, AB2\n"
             "\tsize - integer > 3\n"
             "\tobstacles - integer >= 0\n"
             "\titerations - integer > 0\n"
             "\tconcise - 0, 1")


def argument_check() -> list:
    """
    Checks argument count.
    :return: List of CLA.
    """
    if len(sys.argv) == 6:
        sys.argv.append("0")
    if len(sys.argv) != 7:
        error_and_exit("Invalid number of arguments.")
    if int(sys.argv[3]) < 4:
        error_and_exit("Choose a size >= 4")
    if int(sys.argv[4]) < 0:
        error_and_exit("Obstacles must be >= 0")
    if int(sys.argv[5]) < 1:
        error_and_exit("Number of iterations must be > 0")
    return sys.argv


def start_game(game: Game, count: int) -> None:
    """
    Kicker function to start the game loop. Based on player type, the correct
    search algorithm is called. Also calls for updating board when a move is
    selected, displays game board if selected, and stores game time.
    :param game: The instance of the current game.
    :param count: Stores which iteration is being simulated.
    :return: None
    """
    if not game.concise:
        print(f'Game {count + 1} - {game} - Starting board')
        game.display()
    turn = 1
    game.timer = t.time()
    # continue to loop while search returns a move
    while True:
        search_algo = game.p1_search if game.curPlayer == 1 else game.p2_search
        if search_algo == 'MM':
            move = minimax_search(game, game.state)
        elif search_algo == 'AB':
            move = alpha_beta_search(game, game.state)
        else:
            # human player
            move = get_player_move(game, game.state)
        if move is None:
            game.running = False
            break
        game.update_game(move)
        # displays updated board state
        if not game.concise:
            print("Turn: " + str(turn))
            game.display()
            print(f"\tPlayer 1, {game.player1}: {game.p1_states:9.1f}\n"
                  f"\tPlayer 2, {game.player2}: {game.p2_states:9.1f}\n")
        game.curPlayer = -1 if game.curPlayer == 1 else 1
        turn += 1
    game.timer = t.time() - game.timer
    if not game.concise:
        print('End of game')


def alpha_beta_search(game: Game, state: list) -> tuple:
    """
    Kicker function to start the recursive alpha-beta search method.
    :param game: The instance of the current game.
    :param state: The current state of the game.
    :return: A valid move, or None if no moves are left (end of game).
    """
    # if player 1's turn call MAX function first, else call MIN function first
    if game.to_move() == 1:
        value, move = ab_max_value(game, state, 0, game.p1_depth, float('-inf'), float('inf'))
    else:
        value, move = ab_min_value(game, state, 0, game.p2_depth, float('-inf'), float('inf'))
    return move


def ab_max_value(game: Game, state: list, depth: int, limit: int, alpha, beta) -> tuple:
    """
    Recursive function to search game tree as the MAX player (alpha-beta pruning).
    :param game: The instance of the current game.
    :param state: The state being considered. Initially the current game state.
    :param depth: The current depth of the search.
    :param limit: The depth limit related to the current turn's player.
    :param alpha: Current best value for MAX
    :param beta: Current best value for MIN
    :return: Best value and move for MAX
    """
    is_terminal = game.is_terminal(state)
    if depth > limit or is_terminal:
        return game.utility(state, game.to_move(), is_terminal), None
    v, move = float('-inf'), None
    state_count = 0
    for action in game.actions(state):
        state_count += 1
        temp_state = np.copy(state)
        v2, a2 = ab_min_value(game, game.result(temp_state, action, 1),
                              depth + 1, limit, alpha, beta)
        if v2 > v:
            v, move = v2, action
            alpha = max(alpha, v)
        # prune if best MAX value is greater than best MIN
        if v >= beta:
            return v, move
    if game.curPlayer == 1:
        game.p1_states += state_count
    else:
        game.p2_states += state_count
    return v, move


def ab_min_value(game: Game, state: list, depth: int, limit: int, alpha, beta) -> tuple:
    """
    Recursive function to search game tree as the MIN player (alpha-beta pruning).
    :param game: The instance of the current game.
    :param state: The state being considered. Initially the current game state.
    :param depth: The current depth of the search.
    :param limit: The depth limit related to the current turn's player.
    :param alpha: Current best value for MAX
    :param beta: Current best value for MIN
    :return: Best value and move for MIN
    """
    is_terminal = game.is_terminal(state)
    if depth > limit or is_terminal:
        return game.utility(state, game.to_move(), is_terminal), None
    v, move = float('inf'), None
    state_count = 0
    for action in game.actions(state):
        state_count += 1
        temp_state = np.copy(state)
        v2, a2 = ab_max_value(game, game.result(temp_state, action, -1),
                              depth + 1, limit, alpha, beta)
        if v2 < v:
            v, move = v2, action
            beta = max(beta, v)
        # prune if best MIN value is less than best MAX
        if v <= alpha:
            return v, move
    if game.curPlayer == 1:
        game.p1_states += state_count
    else:
        game.p2_states += state_count
    return v, move


def minimax_search(game: Game, state: list) -> tuple:
    """
    Kicker function to start the recursive minimax search method.
    :param game: The instance of the current game.
    :param state: The current state of the game.
    :return: A valid move, or None if no moves are left (end of game).
    """
    # if player 1's turn call MAX function first, else call MIN function first
    if game.to_move() == 1:
        value, move = max_value(game, state, 0, game.p1_depth)
    else:
        value, move = min_value(game, state, 0, game.p2_depth)
    return move


def max_value(game: Game, state: list, depth: int, limit: int) -> tuple:
    """
    Recursive function to search game tree as the MAX player (minimax).
    :param game: The instance of the current game.
    :param state: The state being considered. Initially the current game state.
    :param depth: The current depth of the search.
    :param limit: The depth limit related to the current turn's player.
    :return:
    """
    is_terminal = game.is_terminal(state)
    if depth > limit or is_terminal:
        return game.utility(state, game.to_move(), is_terminal), None
    v, move = float('-inf'), None
    state_count = 0
    for action in game.actions(state):
        state_count += 1
        temp_state = np.copy(state)
        v2, a2 = min_value(game, game.result(temp_state, action, 1),
                           depth + 1, limit)
        if v2 > v:
            v, move = v2, action
    if game.curPlayer == 1:
        game.p1_states += state_count
    else:
        game.p2_states += state_count
    return v, move


def min_value(game: Game, state: list, depth: int, limit: int) -> tuple:
    """
    Recursive function to search game tree as the MIN player (minimax).
    :param game: The instance of the current game.
    :param state: The state being considered. Initially the current game state.
    :param depth: The current depth of the search.
    :param limit: The depth limit related to the current turn's player.
    :return:
    """
    is_terminal = game.is_terminal(state)
    if depth > limit or is_terminal:
        return game.utility(state, game.to_move(), is_terminal), None
    v, move = float('inf'), None
    state_count = 0
    for action in game.actions(state):
        state_count += 1
        temp_state = np.copy(state)
        v2, a2 = max_value(game, game.result(temp_state, action, -1),
                           depth + 1, limit)
        if v2 < v:
            v, move = v2, action
    if game.curPlayer == 1:
        game.p1_states += state_count
    else:
        game.p2_states += state_count
    return v, move


def get_player_move(game: Game, state: list) -> any:
    if game.is_terminal(state):
        return None
    # print(state)
    row, col = -1, -1
    valid_move = False
    while not valid_move:
        row = int(input("Enter row number: "))
        col = int(input("Enter col number: "))
        if col < 0 or row < 0 or col >= game.size or row >= game.size:
            print(f"({row}, {col}) is outside the game board. Please try again.")
            continue
        elif not game.is_valid(row, col):
            print(f"({row}, {col}) is blocked by an obstacle. Please try again.")
            continue
        else:
            valid_move = True
    return row, col


def game_stats(game: Game) -> int:
    """
    Function to calculate winner of current game.
    :param game: The instance of the current game.
    :return: The winning player
    """
    p1_score, p2_score = game.final_score(game.state)
    if not game.concise:
        print(f"Player 1's Score, {game.player1}: {p1_score}")
        print(f"Player 2's Score, {game.player2}: {p2_score}")
        print(f'Game time: {game.timer} seconds\n')
    winner = 0
    if p1_score > p2_score:
        winner = 1
    if p1_score < p2_score:
        winner = -1
    return winner


def final_results(game: Game, p1_states: list, p2_states: list, p1_wins: int,
                  p2_wins: int, ties: int, times: list) -> None:
    """
    Calculates and prints the results of the entire simulation.
    :param game: The instance of the current game.
    :param p1_states: A list containing the states visited by player 1 per game.
    :param p2_states: A list containing the states visited by player 2 per game.
    :param p1_wins: Number of player 1 wins over the entire simulation.
    :param p2_wins: Number of player 2 wins over the entire simulation.
    :param ties: Number of ties over the entire simulation.
    :param times: A list of the time each game took in the simulation.
    :return: None
    """
    p1_states = sum(p1_states) / len(p1_states)
    p2_states = sum(p2_states) / len(p2_states)
    avg_time = sum(times) / len(times)
    print(f"Simulation results. Total wins / average states visited, ties, and average time per game:\n"
          f"\tPlayer 1, {game.player1}: {p1_wins:2} / {p1_states:9.1f}\n"
          f"\tPlayer 2, {game.player2}: {p2_wins:2} / {p2_states:9.1f}\n"
          f"\tTies: {ties}\n"
          f"\tAverage time per game: {avg_time:.4f} seconds")


def main() -> None:
    """
    Runs the simulation and collects per game statistics.
    :return: None
    """
    arguments = argument_check()
    p1_states = []
    p2_states = []
    p1_wins = 0
    p2_wins = 0
    ties = 0
    times = []
    game = None
    print(f'Simulating {arguments[1]} vs. {arguments[2]} on a {arguments[3]}x{arguments[3]}'
          f' grid with {arguments[4]} obstacle(s) over {arguments[5]} games\n')
    for i in range(int(arguments[5])):
        game = Game(arguments)
        start_game(game, i)
        winner = game_stats(game)
        match winner:
            case 0:
                ties += 1
            case 1:
                p1_wins += 1
            case -1:
                p2_wins += 1
        p1_states.append(game.p1_states)
        p2_states.append(game.p2_states)
        times.append(game.timer)
    final_results(game, p1_states, p2_states, p1_wins, p2_wins, ties, times)


if __name__ == '__main__':
    main()
