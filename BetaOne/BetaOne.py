import chess
import random
import numpy
import chess.svg
import chess.pgn
import sys, os, random, time, warnings
import multiprocessing
from multiprocessing import Pool, Manager
from scipy.stats import pearsonr

class MCTS(object):

    T = 0
    visits = {}
    differential = {}
    c = 0

    def __init__(self, manager, T=0.3, C=1.5):
        super().__init__()

        self.visits = manager.dict()
        self.differential = manager.dict()
        self.T = T
        self.C = C

    def value(self, board, playouts = 100, steps = 5):
        with Pool() as p:
            copy_fen = board.fen()
            copy_board = chess.Board()
            scores = p.map(self.playout, )
    
    def heuristicValue(board: chess.Board):
        #V + c * sqrt(ln(N) / Ni)
        N = self.visits.get("total", 1)
        Ni = self.visits.get(hash(board.fen()), 1e-9)
        V = self.differential.get(hash(board.fen()), 0) * 1.0 / Ni
        return V + self.C * (numpy.log(N)/Ni)

    def record(self, board: chess.Board, score: int):
        self.visits["total"] = self.visits.get("total", 0) + 1
        self.visits[hash(board.fen())] = self.visits.get(hash(board.fen()), 0) + 1
        self.differential[hash(board.fen())] = self.differential.get(hash(board.fen()), 0) + score

    def rolloutValue(self, board: chess.Board, expand = 150):
        if (expand == 0):
            return -0.5

        if board.is_game_over():
            result = board.result();
            sub_res = result[0 : result.find("-")]
            if (sub_res == "1"):
                record(board, -1)
                return 1
            elif (sub_res == "0"):
                record(board, 1)
                return -1
            elif (sub_res == "1/2"):
                record(board, -0.5)
                return 0.5

        for move in board.legal_moves:

            board.push(move)
            action_mapping[move] = self.heuristicValue(game)
            game.undo_move()

        chosen_action = sample(action_mapping, T=self.T)
        board.push(chosen_action)
        score = -1 * self.playout(board, expand =  expand - 1)
        board.pop()
        self.record(board, score)

        return score


def humanPlayer(board: chess.Board):
        print("enter you move: ")
        inp =  input("")

        while True:
            if (board.is_legal(board.parse_san(inp))):
                break
            print("enter you move: ")
            inp =  input("")

        board.push(board.parse_san(inp))


def main():
    board = chess.Board("rn3rk1/p5pp/2p5/3Ppb2/2q5/1Q6/PPPB2PP/R3K1NR b - - 0 1") 

    #while board.is_checkmate:
     #   print (boardValue(board))
      #  print(board)
       # print("")
        #humanPlayer(board)
        #print (boardValue(board))
        #print(board)
        #print("")
        
        
    
if __name__ == "__main__":
    main()
