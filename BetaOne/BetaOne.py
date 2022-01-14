import chess
import random
import numpy
import time
import chess.svg
import chess.pgn
import chess.polyglot
from utils import sample
import sys, os, random, time, warnings
import multiprocessing
from multiprocessing import Pool, Manager
from scipy.stats import pearsonr


def init(l):
    global lock
    lock = l


class MCTS(object):
    def __init__(self, manager, opening_book, T=0.3, C=1.5):
        super().__init__()

        self.visits = manager.dict()
        self.differential = manager.dict()
        self.opening_book = opening_book
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

        itr = [self.copy_board(board) for i in range(0, playouts)]

        pool.map(self.do_rollout_value, itr)
        pool.close()
        pool.join()

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

    def do_rollout_value(self, board, expand=80):
        diff = {}
        vis = {}

        self.rollout_value(board, diff, vis, expand)

        try:
            temp_diff = self.differential.copy()

            for key in temp_diff:
                if key in diff.keys():
                    diff[key] = temp_diff.get(key, 0) + diff.get(key, 0)

            temp_vis = self.visits.copy()

            for key in temp_vis:
                if key in vis.keys():
                    vis[key] = temp_vis.get(key, 0) + vis.get(key, 0)

            lock.acquire()
            self.differential.update(diff)
            self.visits.update(vis)
            lock.release()

        except Exception as e:
            print(repr(e))

    # one playout
    def rollout_value(self, board: chess.Board, diff, vis, expand):
        if expand == 0:
            vis["total"] = vis.get("total", 0) + 1
            vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
            diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) - 0.5
            return 0.5

        if board.is_game_over():
            result = board.result()
            sub_res = result[0: result.find("-")]
            if sub_res == "1":
                vis["total"] = vis.get("total", 0) + 1
                vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
                diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) - 1
                return 1
            elif sub_res == "0":
                vis["total"] = vis.get("total", 0) + 1
                vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
                diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) + 1
                return -1
            elif sub_res == "1/2":
                vis["total"] = vis.get("total", 0) + 1
                vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
                diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) - 0.5
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

        vis["total"] = vis.get("total", 0) + 1
        vis[self.hash_board(board)] = vis.get(self.hash_board(board), 0) + 1
        diff[self.hash_board(board)] = diff.get(self.hash_board(board), 0) + score

        return score

    # best move
    def best_move(self, board, playouts=75):
        startTime = time.time()

        opening_move = self.opening_book.get(board)

        if opening_move is not None:
            print("opening move")
            return opening_move.move

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
    board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 0")
    # board = chess.Board()
    with chess.polyglot.open_reader("data/baron30.bin") as reader:
        manager = Manager()
        bot = MCTS(manager, reader)

        print(board)
        while not board.is_game_over():
            start = time.time()
            board.push(bot.best_move(board))
            print(board)
            print("Time taken: " + str(time.time() - start))
            board.push(human_player(board))
            print(board)



if __name__ == "__main__":
    main()
