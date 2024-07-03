# Box Game

This application is a game that implements multiple AI opponents to play against each other or against a human player.

The game is played on an n by n board.
A valid move consists of placing a 2x2 square on the board that does not overlap an
obstacle and is within the game board.
The goal is to have the highest score when no valid moves are left.
A player's score is equal to the highest number of their boxes that touch each other.
Touching is defined as two boxes sharing any portion of their edges.
So, two boxes from the same player where only the corners touch does not count.

## Running the application

This application uses NumPy. Run `pip install -r requirements.txt` to install required package.

Usage: `python box_game.py {player1} {player2} {size} {obstacles} {iterations} {concise}`

- `player1`, `player2`: MM0, MM1, MM2, AB0, AB1, AB2, or HUM
  - MMx or ABx chooses whether the AI uses minimax or alpha-beta pruning to search for a play. This only impacts the AIs speed.
  - xx0, xx1, xx2 determines what logic to use; 2 is the hardest opponent.
  - Note that `player1` always goes first.
- `size`: integer > 3
  - The size of the game board; {size} x {size}
  - A board size greater than 7 will significantly increase the time AI takes to play.
- `obstacles`: integer >= 0
  - Number of obstacles to place on the board.
- `iterations`: integer > 0
  - How many games to play before the program ends.
- `concise`: 0, 1 or leave blank
  - If 0 or left blank, only final average results are displayed. If 1, per turn information is printed.