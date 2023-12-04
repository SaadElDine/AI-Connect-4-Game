# Connect 4 AI Agent

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/6fef8394-d8e8-48f5-9ca0-277a3071a5b0)


Connect 4 AI Agent is a sophisticated implementation of the classic Connect 4 game, enhanced with artificial intelligence capabilities. The project focuses on creating an intelligent agent capable of playing Connect 4 against a human opponent, offering a challenging and dynamic gaming experience.

## Game Overview

Connect 4 is a two-player game involving strategically dropping colored discs into a vertical grid with the objective of connecting four discs of the same color either vertically, horizontally, or diagonally. The Connect 4 AI Agent takes this engaging game to the next level by incorporating advanced algorithms and heuristic pruning techniques.

## GUI

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/b01742a8-f3db-4522-9d33-67b12e2a3a01)

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/2708a398-8f35-457c-8158-0fe054cc2eef)

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/c2d360ea-13eb-452e-b7cb-02f5094ed1da)

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/46f01b78-f972-470e-9560-a2e69778115a)

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/3c4eac2b-cb9e-404f-b020-accbddb3bfbe)

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/4bbce564-ec1f-456c-9034-c0444120dff3)


## Unique State Representation

The project utilizes a unique state representation to efficiently manage the board configuration. The board state is encoded as a 64-bit number, where each column's last location and slot information are packed into a binary structure.

![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/4e00a7b4-f188-4cad-84a6-6df2f77b82a1)


- For each column, 6 bits represent each slot inside that column (0 for human and 1 for the intelligent agent).
- 3 bits for each column indicate the last empty location, known as the last location mask.
- The 64th bit is reserved for indicating whether the state is pruned or not in case of using alpha-beta pruning.

## Initialization and Possible Moves Generation

- Initialization of the Connect 4 game board with an empty configuration.
- Representation of the initial state using a unique decimal number.
- Generation of possible moves (game states) for a given player, updating the last location mask.
- Utilization of a 64-bit integer for state representation.

## Heuristic Pruning

The project implements heuristic pruning to cope with the immense branching factor of the Connect 4 game tree. Two heuristic functions are designed to assess the state of the game, providing an efficient means to evaluate positions and trim the search tree.

## Heuristics

Two heuristics are offered:

1. **Medium Level:**
   - Positive Behavior:
     - 4 consecutives (AI colors) get 4 points.
     - 3 candidates consecutive (AI colors)  gets 3 points.
     - 2 candidates consecutive (AI colors) gets  2 points.
     - stopping opponent from getting a point gets 1 point.
   - Negative Behavior: 4 consecutives (Human color) get -4 points.
      - 4 consecutives (Human color)  gets -4 points.
      - 3 candidates consecutive (Human color)  gets -3 points.
      - 2 candidates consecutive (Human color)  gets -2 points.
      - Stopping AI from getting a point gets  -1 point.

        ![image](https://github.com/SaadElDine/AI-Connect-4-Game/assets/113860522/b684903c-c0f2-4b62-bf0f-5798e9753925)

2. **Hard Level:**
   - Positive Behavior:
     -  4 consecutives (AI color) get 40 points.
     -  3 consecutive candidates (AI color) gets 17 points (next move will guarantee the point)
     -  3 consecutive candidates (AI color) gets 15 points (a column is not built yet)
     -  2 consecutive candidates (AI color) gets 4 points (next move will guarantee the point)
     -  2 consecutive candidates (AI color) gets 2 points (a column is not built yet)
     -  Stopping opponent from getting a point gets 13 points.
  
   - Negative Behavior:
     - 4 consecutives (Human color) get -40 points.
     - 3 candidates consecutive (Human color) gets -17 points (next move will guarantee the point)
     - 3 candidates consecutive (Human color) gets -15 points (a column is not built yet)
     - 2 candidates consecutive (Human color) gets -4 points (next move will guarantee the point)
     - 2 candidates consecutive (Human color) gets -2 points (a column is not built yet)
     - stopping AI from getting a point gets -13 points

## Minimax Algorithm

The minimax algorithm is a decision-making algorithm for two-player zero-sum games. It explores the entire game tree to determine the best move for the maximizing player and the worst move for the minimizing player.

## Alpha-Beta Pruning

Alpha-beta pruning is an optimization technique for the minimax algorithm. It reduces the number of nodes evaluated in the minimax tree by eliminating branches that cannot influence the final decision.

## Expected Minimax

Expected Minimax incorporates probabilistic elements into the decision-making process, considering the probability that a disc will fall into a chosen column.

## Conclusion

The Connect 4 AI Agent project successfully combines classic board gaming with advanced artificial intelligence. The integration of various algorithms and heuristic pruning techniques has resulted in a formidable Connect 4 opponent, showcasing algorithmic prowess, heuristic excellence, and efficient pruning. This project stands as a robust and well-crafted example of the exciting possibilities at the intersection of classic games and artificial intelligence.
