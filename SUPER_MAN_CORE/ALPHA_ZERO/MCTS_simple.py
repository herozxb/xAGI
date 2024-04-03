import math
import random
import copy
from TicTacToeLogic import Board



# Exploitation Term : wins / visits

# Exploration Term  : exploration_weight * sqrt(log(total_visits) / visits)

# exploitation_term = child.wins / child.visits if child.visits > 0 else 0
# exploration_term = exploration_weight * math.sqrt(math.log(child.parent.visits) / child.visits) if child.visits > 0 else float('inf')
# ucb1_value = exploitation_term + exploration_term



class TicTacToeMCTS():
    """
    Simple Monte Carlo Tree Search (MCTS) implementation for TicTacToe.
    """

    def __init__(self, exploration_weight=1.4, num_simulations=100):
        self.exploration_weight = exploration_weight
        self.num_simulations = num_simulations

    def select_move(self, board):
        """
        Selects the best move using MCTS.
        """
        root_node = Node(board)
        for _ in range(self.num_simulations):
            node = root_node
            # Selection phase
            while not node.is_leaf() and node.get_board().has_legal_moves():
                node = node.select_child(self.exploration_weight)

            # Expansion phase
            if node.get_board().has_legal_moves():
                node = node.expand()

            # Simulation phase
            winner = self.simulate(node.get_board())

            # Backpropagation phase
            while node is not None:
                node.update(winner)
                node = node.parent

        # Select the best move based on visit counts
        best_move = max(root_node.children, key=lambda x: x.visits).action
        return best_move

    def simulate(self, board):
        """
        Simulates a game until the end.
        """
        current_player = 1
        while not board.is_win(1) and not board.is_win(-1) and board.has_legal_moves():
            legal_moves = board.get_legal_moves(current_player)
            move = random.choice(legal_moves)
            board.execute_move(move, current_player)
            current_player *= -1

        if board.is_win(1):
            return 1
        elif board.is_win(-1):
            return -1
        else:
            return 0


class Node:
    """
    Node class for the MCTS tree.
    """

    def __init__(self, board, parent=None, action=None):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

    def get_board(self):
        return self.board

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, exploration_weight):
        """
        Selects a child node based on the UCB1 formula.
        """
        ucb_values = [self.ucb1(child, exploration_weight) for child in self.children]
        return self.children[ucb_values.index(max(ucb_values))]

    def ucb1(self, child, exploration_weight):
        """
        Calculates the UCB1 value for a child node.
        """
        exploitation_term = child.wins / child.visits if child.visits > 0 else 0
        exploration_term = math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
        return exploitation_term + exploration_weight * exploration_term

#    def expand(self):
        """
        Expands the current node by adding child nodes for all legal moves.
        """
#        legal_moves = self.board.get_legal_moves()
#        for move in legal_moves:
#            next_board = copy.deepcopy(self.board)
#            next_board.execute_move(move, 1)
#            self.children.append(Node(next_board, self, move))
#        return random.choice(self.children)

    def expand(self):
        """
        Expands the current node by adding child nodes for all legal moves.
        """
        legal_moves = self.board.get_legal_moves(1)  # Assuming player 1 is the one expanding
        for move in legal_moves:
            next_board = copy.deepcopy(self.board)
            next_board.execute_move(move, 1)
            self.children.append(Node(next_board, self, move))
        return random.choice(self.children)


    def update(self, winner):
        """
        Updates the node's statistics (wins and visits) based on the winner of the simulation.
        """
        self.visits += 1
        if winner == 1:
            self.wins += 1
        elif winner == -1:
            self.wins -= 1
            
            
# Initialize the MCTS agent
mcts_agent = TicTacToeMCTS()

# Assuming you have your TicTacToe board initialized
current_board = Board()

# Select the best move using MCTS
best_move = mcts_agent.select_move(current_board)

print("Best Move:", best_move)        
            

