import chess
import random
import numpy
import time
import chess.svg
import chess.pgn
from utils import sample
import sys, os, random, time, warnings
import multiprocessing
from multiprocessing import Pool, Manager
from scipy.stats import pearsonr


class MCTS(object):
    def __init__(self, manager, T=0.3, C=1.5):
        super().__init__()

        self.visits = manager.dict()
        self.differential = manager.dict()
        self.T = T
        self.C = C

    def copy_board(self, board):
        fen = board.fen()
        copy_fen = fen[0:len(fen) - 4] + " 0 0"
        return chess.Board(copy_fen)

    def value(self, board, playouts=50, steps=5):
        give = [playouts]
        visit_dict = [playouts]
        diff_dict = [playouts]

        for i in range(playouts):
            visit_dict[i] = {}
            diff_dict[i] = {}
            give[i] = (self.copy_board(board), visit_dict[i], diff_dict[i])

        with Pool() as p:
            scores = p.starmap(self.rollout_value, give)

        amalgam_visit = {}
        amalgam_diff = {}

        for i in range(playouts):
            temp = {key: visit_dict[i].get(key, 0) + amalgam_visit.get(key, 0)
                    for key in set(visit_dict[i]) | set(amalgam_visit)}

            amalgam_visit = temp

            temp = {key: diff_dict[i].get(key, 0) + amalgam_diff.get(key, 0)
                    for key in set(diff_dict[i]) | set(amalgam_diff)}

            amalgam_diff = temp

        return self.differential[self.hash_board(board)] * 1.0 / self.visits[self.hash_board(board)]

    def heuristic_value(self, board: chess.Board):
        # V/Ni + c * sqrt(ln(N) / Ni)
        N = self.visits.get("total", 1)
        Ni = self.visits.get(self.hash_board(board), 1e-9)
        V = self.differential.get(self.hash_board(board), 0) * 1.0 / Ni
        return V + self.C * (numpy.log(N) / Ni)

    def record(self, board: chess.Board, score: float):
        self.visits["total"] = self.visits.get("total", 0) + 1
        self.visits[self.hash_board(board)] = self.visits.get(self.hash_board(board), 0) + 1
        self.differential[self.hash_board(board)] = self.differential.get(self.hash_board(board), 0) + score

    def rollout_value(self, board: chess.Board, expand=100):
        if expand == 0:
            self.record(board, -0.5)
            return 0.5

        if board.is_game_over():
            result = board.result()
            sub_res = result[0: result.find("-")]
            if sub_res == "1":
                self.record(board, -1)
                return 1
            elif sub_res == "0":
                self.record(board, 1)
                return -1
            elif sub_res == "1/2":
                self.record(board, -0.5)
                return 0.5

        action_mapping = {}

        for move in board.legal_moves:
            board.push(move)
            action_mapping[move] = self.heuristic_value(board)
            board.pop()

        chosen_action = sample(action_mapping, T=self.T)
        board.push(chosen_action)
        score = -1 * self.rollout_value(board, expand=expand - 1)
        board.pop()
        self.record(board, score)

        return score

    def best_move(self, board, playouts=50):
        startTime = time.time()

        action_mapping = {}

        for move in board.legal_moves:
            board.push(move)
            action_mapping[move] = self.value(board, playouts=playouts)
            board.pop()

        print("Process took: " + str(time.time() - startTime) + " seconds")
        print(str((time.time() - startTime) / 60.0) + " minutes")

        return max(action_mapping, key=action_mapping.get)

    def hash_board(self, board):
        fen_string = board.fen()
        return fen_string[0:len(fen_string) - 13]


def human_player(board: chess.Board):
    print("enter you move: ")
    inp = input("")

    while True:
        if board.is_legal(board.parse_san(inp)):
            break
        print("enter you move: ")
        inp = input("")

    return board.parse_san(inp)


def main():
    board = chess.Board()

    manager = Manager()
    bot = MCTS(manager)

    print(board)
    while not board.is_game_over():
        board.push(human_player(board))
        print(board)
        board.push(bot.best_move(board))
        print(board)


if __name__ == "__main__":
    main()
