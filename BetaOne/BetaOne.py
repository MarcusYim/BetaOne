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


def init(l):
    global lock
    lock = l


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

    # total value per move
    def value(self, board, playouts=50, steps=5):
        l = multiprocessing.Lock()
        pool = multiprocessing.Pool(initializer=init, initargs=(l,))

        itr = []
        vis = {}
        diff = {}

        for i in range(0, playouts):
            itr.append((self.copy_board(board), diff, vis))

        pool.starmap(self.rollout_value, itr)
        pool.close()
        pool.join()

        for key in self.differential:
            if key in diff.keys():
                diff[key] = self.differential.get(key) + diff.get(key)

        self.differential.update(diff)

        print(len(diff))

        for key in self.visits:
            if key in vis.keys():
                vis[key] = self.visits.get(key) + vis.get(key)

        self.visits.update(vis)

        print(len(vis))

        return self.differential[self.hash_board(board)] * 1.0 / self.visits[self.hash_board(board)]

    def heuristic_value(self, board: chess.Board):
        # V/Ni + c * sqrt(ln(N) / Ni)
        N = self.visits.get("total", 1)
        Ni = self.visits.get(self.hash_board(board), 1e-9)
        V = self.differential.get(self.hash_board(board), 0) * 1.0 / Ni
        return V + self.C * (numpy.log(N) / Ni)

    def record(self, board: chess.Board, score: float, diff, vis):
        vis["total"] = vis.get("total", 0) + 1
        vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
        diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) + score

    # one playout
    def rollout_value(self, board: chess.Board, diff, vis, expand=80):
        if expand == 0:
            lock.acquire()
            self.record(board, -0.5, diff, vis)
            lock.release()
            return 0.5

        if board.is_game_over():
            result = board.result()
            sub_res = result[0: result.find("-")]
            if sub_res == "1":
                lock.acquire()
                self.record(board, -1, diff, vis)
                lock.release()
                return 1
            elif sub_res == "0":
                lock.acquire()
                self.record(board, 1, diff, vis)
                lock.release()
                return -1
            elif sub_res == "1/2":
                lock.acquire()
                self.record(board, -0.5, diff, vis)
                lock.release()
                return 0.5

        action_mapping = {}

        for move in board.legal_moves:
            board.push(move)
            action_mapping[move] = self.heuristic_value(board)
            board.pop()

        chosen_action = sample(action_mapping, T=self.T)
        board.push(chosen_action)
        score = -1 * self.rollout_value(board, diff, vis, expand=expand - 1)
        board.pop()

        lock.acquire()
        self.record(board, score, diff, vis)
        lock.release()

        return score

    # best move
    def best_move(self, board, playouts=75):
        startTime = time.time()

        action_mapping = {}

        for move in board.legal_moves:
            board.push(move)
            action_mapping[move] = self.value(board, playouts=playouts)
            print("done")
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
    # board = chess.Board("r1bqk2r/pppp1ppp/2nb1n2/1B2p3/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq")
    board = chess.Board()

    manager = Manager()
    bot = MCTS(manager)

    print(board)
    while not board.is_game_over():
        board.push(human_player(board))
        print(board)
        start = time.time()
        board.push(bot.best_move(board))
        print(board)
        print("Time taken: " + str(time.time() - start))


if __name__ == "__main__":
    main()
