import Arena
from MCTS import MCTS

from tictactoe.TicTacToeGame import TicTacToeGame
from tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet

import numpy as np
from utils import *

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
            
        print("=============RandomPlayer==============")
        print(a)
        return a


game = TicTacToeGame()

neural_net = TicTacToeKerasNNet

rp = RandomPlayer(game).play

args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
mcts = MCTS(game, neural_net(game), args)
n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

print("==============play_int====================")
print(rp)
print(n1p)

arena = Arena.Arena(n1p, rp, game)
print(arena.playGames(10, verbose=False))




player1 = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

player2 = RandomPlayer(game).play

arena = Arena.Arena(player1, player2, game)
print(arena.playGames(100, verbose=False))




