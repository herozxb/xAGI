import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        # Ps is a policy that is a probability vector over all possible actions
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.valid_move = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

	# s = canonicalBoard.tostring()
        s = self.game.stringRepresentation(canonicalBoard)



	# get the end state
        if s not in self.Es:
            # 1 is player = 1
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)

        if self.Es[s] != 0:
            # terminal node
            # self.Es[s] = 0 is has move, self.Es[s] = 1 or self.Es[s] = -1, the game end 
            return -self.Es[s]

        if s not in self.Ps:
            # Ps is a policy that is a probability vector over all possible actions
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            print("=====================Ps==========================")
            print( s, self.Ps[s] , v)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            print( self.Ps[s], valids )
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            print(sum_Ps_s)
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            print("====valids====")
            print( valids, self.Ps[s], self.Ps[s][0] )
            self.valid_move[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.valid_move[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        
        print("===================NextState=======================")
        print( next_s, next_player )
        
        next_s = self.game.getCanonicalForm(next_s, next_player)
        print( next_s )

        # Backpropagation phase
        #while node is not None:
        #	node.update(winner)
        #	node = node.parent
        
        # Backpropagation phase
        v = self.search(next_s)

	# update( winner )
        if (s, a) in self.Qsa:
            # mean of Qsa = ( number * Qsa + v ) / ( number + 1 )
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        
        return -v
        
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
            
